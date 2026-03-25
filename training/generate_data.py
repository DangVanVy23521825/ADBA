"""
ADBA — Synthetic Training Data Generator
=========================================
Calls GPT-4o-mini in batches to generate (instruction, context) → (plan/sql/insight)
pairs for SFT fine-tuning of Qwen2.5-Coder-7B.

Target distribution (1,500 raw → ≥1,200 valid after validate_dataset.py):
  sql        : 600  — text-to-sql samples
  python     : 400  — data-analysis samples
  supervisor : 200  — ExecutionPlan routing samples
  insight    : 150  — InsightOutput samples
  reflector  :  50  — error-reflection samples (simpler, short)
  ─────────────────
  Total raw  : 1,400  (buffer: generate ~1,500 to absorb ~7% format failures)

Output: data/raw_dataset.jsonl
  Each line is a ShareGPT-format record:
  {
    "skill_type": "text-to-sql",
    "messages": [
      {"role": "system",    "content": "<prompt template filled>"},
      {"role": "user",      "content": "<query or task>"},
      {"role": "assistant", "content": "<sql | plan json | insight json | python code>"}
    ]
  }

Usage:
  export OPENAI_API_KEY=sk-...
  export DATABASE_URL=postgresql://adba_user:adba@localhost:5432/adba_db
  python training/generate_data.py
  python training/generate_data.py --skill sql --count 50   # partial run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
# Allow `from schemas.plan_schema import ...` when running as:
#   python training/generate_data.py
# (cwd is project root but Python's import path is training/, not ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data"
PROMPT_DIR  = ROOT / "prompts"
PERCEPTION  = ROOT / "perception"
OUTPUT_FILE = DATA_DIR / "raw_dataset.jsonl"

# ── API config ─────────────────────────────────────────────────────────────────
MODEL          = "gpt-4o-mini"
# Skill-specific decoding to avoid structural drift on JSON outputs.
SKILL_TEMPERATURES: dict[str, float] = {
    "text-to-sql":        0.35,
    "data-analysis":      0.45,
    "supervisor-routing": 0.10,
    "insight-generation": 0.20,
    "error-reflection":   0.20,
}
# Avoid SQL truncation: text-to-sql gets a larger completion budget.
SKILL_MAX_TOKENS: dict[str, int] = {
    "text-to-sql":        1800,
    "data-analysis":      1200,
    "supervisor-routing": 700,
    "insight-generation": 800,
    "error-reflection":   700,
}
REQUESTS_PER_MINUTE = 450     # gpt-4o-mini tier-1 limit; stay under to avoid 429s
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 2.0          # seconds between retries
SQL_SELF_REPAIR_ATTEMPTS = 2
SUPERVISOR_SELF_REPAIR_ATTEMPTS = 2
INSIGHT_SELF_REPAIR_ATTEMPTS = 4

# ── Generation targets ────────────────────────────────────────────────────────
TARGETS: dict[str, int] = {
    "sql":        600,
    "python":     400,
    "supervisor": 200,
    "insight":    150,
    "reflector":  50,
}
SKILL_KEY_TO_TYPE: dict[str, str] = {
    "sql": "text-to-sql",
    "python": "data-analysis",
    "supervisor": "supervisor-routing",
    "insight": "insight-generation",
    "reflector": "error-reflection",
}
SKILL_TYPE_TO_KEY: dict[str, str] = {v: k for k, v in SKILL_KEY_TO_TYPE.items()}

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://adba_user:adba@localhost:5432/adba_db"
)

# =============================================================================
# INFO-BOX LOADER
# =============================================================================

def load_info_box(domain: str) -> str:
    """Load and compact info_box JSON for a domain."""
    path = PERCEPTION / f"info_box_{domain}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"info_box not found: {path}. "
            "Run perception/extract_info_box.py first."
        )
    with open(path, encoding="utf-8") as f:
        return json.dumps(json.load(f), ensure_ascii=False, separators=(",", ":"))


def load_all_info_box() -> str:
    path = PERCEPTION / "info_box_all.json"
    if not path.exists():
        raise FileNotFoundError(f"info_box_all.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.dumps(json.load(f), ensure_ascii=False, separators=(",", ":"))


def load_slim_info_box() -> str:
    """Return a compact info_box for Supervisor: table_name + column names/types only.

    Drops sample_rows, indexes, foreign_keys, row_count, primary_key.
    Reduces prompt size from ~25 K chars (6 270 tokens) → ~5 K chars (~1 200 tokens),
    well within Qwen-7B 4096-token context window.
    """
    path = PERCEPTION / "info_box_all.json"
    if not path.exists():
        raise FileNotFoundError(f"info_box_all.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    slim_tables = []
    for t in data.get("tables", []):
        slim_t: dict = {
            "table_name": t.get("table_name", t.get("name", "")),
            "columns": [],
        }
        for col in t.get("columns", []):
            slim_col: dict = {
                "name": col.get("name", ""),
                "type": col.get("data_type", col.get("type", "")),
            }
            if col.get("is_generated"):
                slim_col["generated"] = True
            slim_t["columns"].append(slim_col)
        slim_tables.append(slim_t)

    slim = {
        "tables": slim_tables,
        "cross_domain_hints": data.get("cross_domain_hints", []),
    }
    return json.dumps(slim, ensure_ascii=False, separators=(",", ":"))


# =============================================================================
# PROMPT LOADER
# =============================================================================

def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


# =============================================================================
# QUERY / TASK SEED POOLS
# These are diverse seed questions. GPT-4o-mini will be called with each
# seed to generate the corresponding SQL / plan / insight output.
#
# Target sizes (with MAX_SEED_REUSE = 3):
#   SQL_SEEDS      ≥ 200 → can generate 600 samples
#   PYTHON_SEEDS   ≥ 135 → can generate 405 samples
#   SUPERVISOR_SEEDS ≥ 70 → can generate 210 samples
#   INSIGHT_SEEDS  ≥ 50 → can generate 150 samples
# =============================================================================

# Maximum times a seed query may be reused (different LLM outputs per call due to temperature).
# Per-skill values are tuned so seed_count × max_reuse ≥ target:
#   sql        156 × 4 = 624  ≥ 600
#   python     110 × 4 = 440  ≥ 400
#   supervisor  67 × 3 = 201  ≥ 200
#   insight     50 × 3 = 150  ≥ 150  (see INSIGHT_SEEDS below for 50 seeds)
#   reflector    5 × 12 = 60  ≥  50
MAX_SEED_REUSE_PER_SKILL: dict[str, int] = {
    "sql":        4,
    "python":     4,
    "supervisor": 3,
    "insight":    4,    # 48 seeds × 4 = 192 ≥ 150
    "reflector":  12,
}

def _max_reuse(skill: str) -> int:
    return MAX_SEED_REUSE_PER_SKILL.get(skill, 3)

# Each seed is (query, domain). domain controls which info_box is injected:
#   "sales" → info_sales  "inventory" → info_inv  "hr" → info_hr  "cross" → info_all
SQL_SEEDS: list[tuple[str, str]] = [
    # ── Sales — basic aggregations ───────────────────────────────────────────
    ("Tổng doanh thu theo từng region trong Q4 2024",                                    "sales"),
    ("Top 5 khách hàng có doanh thu cao nhất năm 2024",                                  "sales"),
    ("Số đơn hàng cancelled trong tháng 11/2023 theo từng region",                       "sales"),
    ("Tỷ lệ đơn hàng refunded theo category trong Q3 2023",                              "sales"),
    ("Số lượng sản phẩm theo từng category đang active",                                 "sales"),
    ("Doanh thu theo segment khách hàng trong năm 2024",                                 "sales"),
    ("Tổng revenue theo segment khách hàng trong từng quý 2024",                         "sales"),
    ("Sản phẩm có gross margin cao nhất theo category (unit_price - cost)",               "sales"),
    ("Top 10 sản phẩm bán nhiều nhất theo quantity trong Q4 2024",                       "sales"),
    ("Khách hàng chưa đặt hàng nào trong năm 2024 nhưng đã mua trong 2023",              "sales"),
    ("Tổng số đơn hàng theo status và region trong năm 2024",                            "sales"),
    ("Revenue trung bình mỗi đơn hàng (AOV) theo region trong Q4 2024",                  "sales"),
    ("Category nào có tỷ lệ cancelled cao nhất trong Q3 2024?",                          "sales"),
    ("Top 5 khách hàng đặt hàng nhiều lần nhất trong năm 2024",                          "sales"),
    ("So sánh số lượng đơn hàng completed theo tháng 2023 vs 2024",                      "sales"),
    ("Revenue từ top 3 khách hàng mỗi region trong Q4 2024",                             "sales"),
    ("Số lượng khách hàng unique đặt hàng trong mỗi tháng của 2024",                     "sales"),
    ("Top 5 thành phố có doanh thu cao nhất năm 2024",                                   "sales"),
    ("Danh sách khách hàng đặt trên 10 đơn hàng trong Q4 2024",                          "sales"),
    ("Khách hàng có tổng doanh thu vượt 50,000,000 trong năm 2024",                      "sales"),
    ("Category nào có gross profit margin cao nhất trong năm 2024?",                     "sales"),
    ("Revenue contribution của top 10 sản phẩm chiếm bao nhiêu % tổng 2024?",           "sales"),
    # ── Sales — order status analysis ────────────────────────────────────────
    ("Số đơn hàng cancelled trong mỗi tháng của năm 2024 theo region",                   "sales"),
    ("Đơn hàng có amount > 10,000,000 VND trong Q4 2024 theo region",                    "sales"),
    ("Tổng revenue và count theo status trong Q4 2024",                                  "sales"),
    ("Tháng nào có tỷ lệ đơn hàng refunded cao nhất trong năm 2023?",                   "sales"),
    ("Danh sách đơn hàng có trạng thái pending đặt trong tháng 12/2024",                 "sales"),
    ("So sánh số đơn hàng theo từng quý năm 2023 vs 2024",                               "sales"),
    ("Region nào có tỷ lệ đơn hàng completed cao nhất trong Q4 2024?",                  "sales"),
    ("Đơn hàng nào có quantity > 10 trong tháng 11/2024?",                               "sales"),
    ("Danh sách đơn hàng có status pending đặt trước 30 ngày so với hôm nay",            "sales"),
    ("Tổng doanh thu và count theo status cho từng region năm 2024",                     "sales"),
    # ── Sales — rolling windows ───────────────────────────────────────────────
    ("Doanh thu 30 ngày gần nhất so với 30 ngày trước đó theo region",                   "sales"),
    ("Tổng doanh thu 90 ngày qua theo từng category",                                    "sales"),
    ("Đơn hàng nào được tạo trong 7 ngày qua có status pending?",                        "sales"),
    ("So sánh doanh thu 180 ngày đầu 2024 vs 180 ngày đầu 2023",                         "sales"),
    ("Tỷ lệ hoàn tiền Electronics trong 60 ngày gần nhất",                               "sales"),
    ("Khách hàng chưa đặt hàng trong 90 ngày qua nhưng đã mua năm 2023",                 "sales"),
    ("Doanh thu trung bình mỗi ngày theo tháng trong năm 2024",                          "sales"),
    ("Revenue theo từng ngày trong tháng 1/2024 cho region Miền Bắc",                   "sales"),
    ("Số lượng đơn hàng trung bình mỗi tuần theo region trong Q4 2024",                  "sales"),
    ("Phân tích trend doanh thu theo tuần trong tháng 12/2024",                          "sales"),
    # ── Sales — YoY / QoQ comparisons ─────────────────────────────────────────
    ("So sánh doanh thu theo tháng giữa 2023 và 2024 cho Miền Bắc",                      "sales"),
    ("Tăng trưởng doanh thu Q4 2024 vs Q4 2023 theo từng region",                        "sales"),
    ("Doanh thu tháng 3/2024 so với trung bình 3 tháng trước",                           "sales"),
    ("Sản phẩm Electronics nào có tỷ lệ refund cao bất thường tháng 11/2023?",           "sales"),
    ("SKU nào có effective price thấp hơn cost kể từ tháng 4/2024?",                     "sales"),
    ("So sánh revenue per customer theo segment trong Q4 2024 vs Q4 2023",               "sales"),
    ("Tốc độ tăng trưởng revenue QoQ theo category trong năm 2024",                      "sales"),
    ("Top 3 region có doanh thu tăng trưởng nhất từ Q1 sang Q4 2024",                    "sales"),
    ("So sánh doanh thu tháng 3/2024 vs tháng 3/2023 theo từng category",                "sales"),
    ("Doanh thu theo từng quý 2023 vs 2024 — waterfall comparison",                      "sales"),
    # ── Sales — product / category deep-dive ─────────────────────────────────
    ("Sản phẩm nào có revenue giảm liên tiếp 3 quý trong năm 2024?",                    "sales"),
    ("Sản phẩm có unit_price cao nhất trong từng category",                              "sales"),
    ("Top 5 sản phẩm có gross margin > 50% và quantity sold > 100 trong 2024",           "sales"),
    ("Revenue contribution % của mỗi category theo quý năm 2024",                       "sales"),
    ("Khách hàng nào mua sản phẩm từ ít nhất 3 category khác nhau trong 2024?",          "sales"),
    ("Top 5 ngày có doanh thu cao nhất trong năm 2024",                                  "sales"),
    ("Sản phẩm Clothing nào có doanh thu vượt 100,000,000 trong 2024?",                  "sales"),
    ("Danh sách sản phẩm không có đơn hàng nào trong tháng 12/2024",                    "sales"),
    ("Danh sách top 10 đơn hàng có revenue cao nhất trong năm 2024",                    "sales"),
    ("Category Food nào bán được nhiều nhất theo quantity trong Q3 2024?",               "sales"),
    # ── Sales — customer behaviour ────────────────────────────────────────────
    ("Segment khách hàng nào có tốc độ tăng trưởng order count cao nhất Q3→Q4 2024?",   "sales"),
    ("Khách hàng Enterprise và SME ở Miền Nam có khác biệt về AOV không trong Q4 2024?", "sales"),
    ("So sánh revenue per customer theo segment trong Q4 2024",                          "sales"),
    ("Khách hàng B2B có order frequency thay đổi thế nào từ 2023 sang 2024?",            "sales"),
    ("Tính RFM cơ bản: Recency, Frequency, Monetary cho top 20 khách hàng 2024",         "sales"),
    ("Revenue từ 3 khách hàng lớn nhất chiếm bao nhiêu % tổng doanh thu 2024?",          "sales"),
    ("Số lượng khách hàng Enterprise đặt hàng mỗi tháng trong Q4 2024",                 "sales"),
    ("Region nào có số lượng sản phẩm khác nhau được mua nhiều nhất trong 2024?",        "sales"),
    ("Khách hàng nào có first order và last order cách nhau trên 300 ngày trong 2024?",  "sales"),
    ("Danh sách khách hàng SME chưa đặt hàng trong Q4 2024 nhưng active Q3 2024",       "sales"),
    # ── Sales — advanced analytics ────────────────────────────────────────────
    ("Tháng nào có gross profit cao nhất: (amount - cost*quantity) trong năm 2024?",     "sales"),
    ("Sản phẩm nào có effective price bằng với cost (break-even) trong Q4 2024?",        "sales"),
    ("Phân tích basket size: số đơn hàng trung bình mỗi khách trong quý theo region",   "sales"),
    ("Revenue run rate Q4 2024 annualized (×4) theo region",                             "sales"),
    ("Đơn hàng nào có amount > mean + 3×stddev trong năm 2024?",                         "sales"),
    ("Category nào chiếm > 30% tổng doanh thu nhưng < 10% số SKU?",                     "sales"),
    ("Khách hàng Enterprise ở Miền Bắc mua sản phẩm gì nhiều nhất trong 2024?",          "sales"),
    ("So sánh doanh thu weekday vs weekend trong Q4 2024",                               "sales"),
    ("Sản phẩm nào có refund_rate tăng liên tiếp 3 tháng trong năm 2024?",               "sales"),
    ("Miền Nam có bao nhiêu đơn hàng có amount > 5,000,000 trong Q3 2024?",              "sales"),
    # ── Inventory — stock levels ───────────────────────────────────────────────
    ("Sản phẩm nào có tồn kho dưới min_threshold hiện tại?",                             "inventory"),
    ("Số lượng stock_movements theo movement_type trong Q4 2023",                         "inventory"),
    ("Kho nào có tổng write_off lớn nhất trong năm 2023?",                               "inventory"),
    ("Tổng inbound vs outbound movements theo region trong Q4 2024",                      "inventory"),
    ("Kho Miền Bắc có bao nhiêu outbound không có order link trong Q4 2024?",            "inventory"),
    ("Sản phẩm nào không có outbound movement trong 180 ngày qua?",                      "inventory"),
    ("Transfer loop nào xảy ra giữa kho Miền Bắc và Miền Trung trong Q1 2024?",         "inventory"),
    ("Tồn kho trung bình theo region trong 30 ngày qua",                                 "inventory"),
    ("Sản phẩm có inbound nhưng không có outbound trong 6 tháng qua",                   "inventory"),
    ("Sản phẩm nào có stock quantity = 0 tại bất kỳ kho nào?",                           "inventory"),
    ("Total write-off quantity theo product trong năm 2023",                             "inventory"),
    ("Kho Miền Nam nhận bao nhiêu inbound movements trong Q4 2024?",                     "inventory"),
    ("Sản phẩm Electronics có tồn kho trung bình mỗi kho trong tháng 12/2024",           "inventory"),
    # ── Inventory — movement analysis ─────────────────────────────────────────
    ("Transfer movements nào xảy ra giữa các kho trong Q1 2024?",                        "inventory"),
    ("Top 5 sản phẩm có inbound/outbound ratio cao nhất trong Q4 2024",                  "inventory"),
    ("Kho nào có số sản phẩm unique được lưu nhiều nhất?",                               "inventory"),
    ("Tổng số lượng tồn kho theo region tính đến cuối tháng 12/2024",                    "inventory"),
    ("Sản phẩm nào có min_threshold cao nhất trong mỗi kho?",                            "inventory"),
    ("Movement analysis: tỷ lệ write-off/total_outbound theo product category",          "inventory"),
    ("Kho nào có số ngày không có outbound movement nhiều nhất trong Q4 2024?",          "inventory"),
    ("Products nào được transfer qua nhiều kho nhất trong năm 2024?",                    "inventory"),
    ("Kho nào có tổng outbound / capacity ratio cao nhất trong Q4 2024?",                "inventory"),
    ("Số lượng inbound movements theo movement_type trong tháng 12/2024",                "inventory"),
    ("Tổng số lượng mỗi loại movement_type trong từng quý năm 2024",                     "inventory"),
    ("Danh sách các kho có net stock change âm (tồn kho giảm) trong Q4 2024",           "inventory"),
    # ── Inventory — capacity & planning ──────────────────────────────────────
    ("Kho nào có tỉ lệ utilization (quantity/capacity) cao nhất?",                       "inventory"),
    ("So sánh total stock value (quantity × unit_price) theo kho",                       "inventory"),
    ("Tồn kho tại kho Miền Bắc vào ngày đầu và cuối tháng 12/2024",                     "inventory"),
    ("Danh sách sản phẩm có quantity giảm liên tiếp trong 3 tháng cuối 2024",            "inventory"),
    ("Số lần restock (inbound) của từng sản phẩm trong năm 2024",                        "inventory"),
    ("Sản phẩm nào chưa được restock trong 90 ngày qua?",                                "inventory"),
    # ── HR — salary & compensation ────────────────────────────────────────────
    ("Nhân viên nào có lương cao nhất trong phòng Kinh doanh?",                          "hr"),
    ("Tổng overtime_hours theo từng phòng ban tháng 3/2024",                             "hr"),
    ("Phòng ban nào có tổng lương cao nhất so với budget tháng 7/2024?",                 "hr"),
    ("Danh sách nhân viên Senior bị nghỉ việc trong Q2 2024",                            "hr"),
    ("Nhân viên nào có bonus tháng 12 vượt quá 3 lần lương cơ bản?",                    "hr"),
    ("Phòng Tài chính có vượt budget trong tháng nào của 2024?",                         "hr"),
    ("Xu hướng overtime_hours phòng Kinh doanh từ tháng 1 đến 6/2024",                  "hr"),
    ("Nhân viên được tuyển dụng trong 12 tháng qua theo phòng ban",                      "hr"),
    ("Phòng nào có tỷ lệ headcount/budget hiệu quả nhất trong năm 2024?",                "hr"),
    ("Nhân viên có hire_date trong năm 2024 và level Junior",                            "hr"),
    ("Lương trung bình theo level trong từng phòng ban năm 2024",                        "hr"),
    ("Top 5 nhân viên có bonus/base_salary ratio cao nhất tháng 12/2024",                "hr"),
    # ── HR — headcount & turnover ─────────────────────────────────────────────
    ("Số nhân viên resigned trong từng quý năm 2024 theo phòng ban",                     "hr"),
    ("So sánh tổng payroll phòng IT với phòng Kinh doanh theo quý 2024",                 "hr"),
    ("Nhân viên nào có deduction > 10% net_salary trong năm 2024?",                      "hr"),
    ("Phòng ban nào có tỷ lệ nhân viên Senior cao nhất?",                                "hr"),
    ("Tổng headcount hiện tại theo phòng ban và level",                                  "hr"),
    ("Nhân viên có tenure > 5 năm và vẫn đang active",                                   "hr"),
    ("So sánh net_salary trung bình của Manager vs Senior trong từng phòng",             "hr"),
    ("Phòng nào chi nhiều nhất cho bonus trong Q4 2024?",                                "hr"),
    ("Số nhân viên mới tuyển dụng theo quý trong năm 2024",                              "hr"),
    ("Tổng deductions theo phòng ban trong năm 2024",                                    "hr"),
    ("Danh sách nhân viên chưa nhận bonus nào trong năm 2024",                           "hr"),
    ("Nhân viên level Senior phòng Kỹ thuật có lương cao hơn mức trung bình bao nhiêu?", "hr"),
    # ── HR — payroll & budget ─────────────────────────────────────────────────
    ("Phòng Tài chính có vượt budget trong tháng nào của năm 2024?",                     "hr"),
    ("So sánh budget_utilization các phòng ban trong Q4 2024",                           "hr"),
    ("Tổng chi phí nhân sự (base_salary + bonus - deduction) theo phòng năm 2024",       "hr"),
    ("Nhân viên nào có tổng deduction cao nhất trong năm 2024?",                         "hr"),
    ("Phòng IT có tổng overtime cost (overtime_hours × 200,000đ) cao nhất không?",       "hr"),
    ("Tổng net_salary theo phòng ban và level trong Q4 2024",                            "hr"),
    ("Phòng ban nào có chi phí bonus chiếm tỷ lệ cao nhất trên base_salary?",           "hr"),
    ("So sánh tổng payroll Q4 2024 vs Q4 2023 theo phòng ban",                           "hr"),
    # ── Cross-domain ─────────────────────────────────────────────────────────
    ("Sản phẩm bán chạy nhất Q4 2024 và tồn kho hiện tại là bao nhiêu?",                "cross"),
    ("Region nào có doanh thu cao nhất nhưng headcount thấp nhất?",                      "cross"),
    ("So sánh nhân viên active theo region với doanh thu region Q4 2024",                "cross"),
    ("Top 10 sản phẩm bán chạy Q4 2024 kèm tên, category và tồn kho hiện tại",          "cross"),
    ("Sản phẩm Electronics bán chạy nhất Q4 2024 có đủ tồn kho không?",                 "cross"),
    ("Category nào có doanh thu cao nhất Q4 2024 và tình trạng tồn kho như thế nào?",   "cross"),
    ("Danh sách top 5 sản phẩm bán nhiều nhất Q4 2024 kèm quantity in stock",           "cross"),
    ("Region nào có headcount thấp nhất nhưng revenue per employee cao nhất?",           "cross"),
    ("So sánh warehouse capacity utilization và sales volume theo region Q4 2024",       "cross"),
    ("Phòng ban nào có doanh thu bình quân nhân viên cao nhất trong Q4 2024?",           "cross"),
    ("Sản phẩm nào có write-off rate cao nhất mà doanh thu vẫn tốt trong Q4 2024?",     "cross"),
]

PYTHON_SEEDS: list[tuple[str, str]] = [
    # ── Aggregation & statistics ──────────────────────────────────────────────
    ("Tính YoY % thay đổi doanh thu theo region và đánh dấu outlier bằng IQR",
     "region, rev_2024, rev_2023"),
    ("Tính profit margin = (unit_price - cost) / unit_price theo category và sắp xếp giảm dần",
     "name, category, unit_price, cost"),
    ("Phát hiện nhân viên có bonus_ratio > 3σ so với cùng level",
     "name, level, base_salary, bonus"),
    ("Tính tốc độ xuất kho trung bình theo ngày cho mỗi product và dự báo ngày hết hàng",
     "product_id, quantity, avg_daily_outbound"),
    ("Group by quarter và tính cumulative sum của doanh thu năm 2024",
     "quarter, monthly_revenue"),
    ("Tính correlation giữa overtime_hours và doanh thu theo tháng",
     "month, total_revenue, avg_overtime"),
    ("Chuẩn hóa doanh thu (min-max scaling) theo region để so sánh",
     "region, total_revenue"),
    ("Tính số ngày tồn kho = quantity / avg_daily_outbound cho từng SKU",
     "sku, quantity, avg_daily_outbound"),
    ("Phân nhóm nhân viên theo salary band: Low/Mid/High dựa vào percentile",
     "name, level, salary"),
    ("Tính moving average 3 tháng của doanh thu Miền Bắc",
     "month, revenue_mien_bac"),
    ("Detect missing months trong chuỗi payroll của từng nhân viên",
     "employee_id, month, year, net_salary"),
    ("Tính weighted average unit_price theo quantity sold cho mỗi category",
     "category, unit_price, quantity"),
    ("So sánh tổng write_off Q4 2023 với các quý khác và tính % deviation",
     "quarter, year, total_writeoff"),
    ("Flatten nested dict trong cột stats và extract các metrics chính",
     "region, stats_dict"),
    ("Tính Gini coefficient của phân phối doanh thu theo region",
     "region, total_revenue"),
    # ── Aggregation & ranking ─────────────────────────────────────────────────
    ("Tính YoY revenue growth rate theo từng category và rank từ cao đến thấp",
     "category, rev_2024, rev_2023"),
    ("Tính profit per unit = unit_price - cost và tổng profit theo category",
     "name, category, unit_price, cost, quantity_sold"),
    ("Tính revenue contribution % của mỗi region so với tổng",
     "region, total_revenue"),
    ("Tính QoQ growth rate cho mỗi quý và đánh dấu quý tăng trưởng âm",
     "quarter, revenue"),
    ("Xếp hạng sản phẩm theo revenue descent và tính cumulative % (Pareto 80/20)",
     "name, revenue"),
    ("Tính z-score của salary trong mỗi phòng ban và đánh dấu outlier (|z|>2)",
     "name, department, salary"),
    ("Tính turnover rate = resigned / total_active theo phòng ban",
     "department, active_count, resigned_count"),
    ("Phân nhóm đơn hàng theo amount bucket và tính count + revenue per bucket",
     "order_id, region, amount"),
    ("Tính weighted average unit_price theo quantity_sold trong mỗi category",
     "category, unit_price, quantity_sold"),
    ("Tính CAGR (Compound Annual Growth Rate) từ 2022 đến 2024 theo region",
     "region, rev_2022, rev_2023, rev_2024"),
    # ── Time series & trend ───────────────────────────────────────────────────
    ("Xác định tháng có revenue cao nhất và thấp nhất trong năm, tính variance",
     "month, revenue"),
    ("Tính month-over-month % change và đánh dấu các tháng giảm liên tiếp",
     "month, year, revenue"),
    ("Tính cumulative revenue từ đầu năm 2024 theo từng tháng",
     "month, monthly_revenue"),
    ("Detect seasonal patterns bằng cách chia revenue theo quarter và tính mean/std",
     "quarter, year, revenue"),
    ("Tính rolling 3-month average của revenue và độ lệch so với actual",
     "month, revenue"),
    ("So sánh trend Q4 2024 với Q4 2023 bằng linear regression slope",
     "month, rev_2024, rev_2023"),
    ("Tính exponential moving average (EMA) 3 tháng của doanh thu",
     "month, revenue"),
    ("Phát hiện ngày có revenue drop > 30% so với ngày hôm trước",
     "date, daily_revenue"),
    ("Tính tốc độ tăng trưởng doanh thu trung bình theo tuần trong Q4 2024",
     "week, revenue"),
    ("Tính moving average 7 ngày và flag ngày có actual > 2× MA",
     "date, daily_revenue"),
    # ── Outlier detection & statistical tests ─────────────────────────────────
    ("Detect outlier salary trong mỗi level dùng IQR method và thêm cột is_outlier",
     "name, level, salary"),
    ("Tính Pearson correlation giữa overtime_hours và employee bonus",
     "name, overtime_hours, bonus"),
    ("Phát hiện sản phẩm có refund_rate > mean + 2×std trong mỗi category",
     "sku, category, order_count, refund_count"),
    ("Tính confidence interval 95% cho average order value theo region",
     "region, order_id, amount"),
    ("Detect inventory anomalies: kho có stock_change > 3×sigma so với average",
     "warehouse_id, date, stock_quantity"),
    ("Phát hiện employee có lương tăng bất thường (>2× peer average) trong 2024",
     "name, level, salary_2023, salary_2024"),
    ("Tính regression tuyến tính dự báo revenue tháng tiếp theo dựa trên 6 tháng",
     "month, revenue"),
    ("Tính Z-score và chuẩn hóa về [0,1] cho tất cả numeric features",
     "region, revenue, order_count, customer_count"),
    ("Detect outlier orders bằng IQR: amount > Q3 + 1.5×IQR",
     "order_id, region, amount"),
    ("Tính chi-square test kiểm tra phân phối orders theo region có đồng đều không",
     "region, order_count"),
    # ── Data transformation ───────────────────────────────────────────────────
    ("Pivot table: rows=region, cols=quarter, values=revenue, fill missing với 0",
     "region, quarter, revenue"),
    ("Melt wide format (region, q1_rev, q2_rev, q3_rev, q4_rev) thành long format",
     "region, q1_rev, q2_rev, q3_rev, q4_rev"),
    ("Tính pairwise correlation matrix giữa revenue, order_count, avg_amount",
     "month, revenue, order_count, avg_amount"),
    ("Reshape data: aggregate weekly revenue từ daily transactions, fill missing weeks",
     "date, daily_revenue"),
    ("Tính weighted rank của sản phẩm dựa trên revenue (40%) và quantity (60%)",
     "product_id, revenue, quantity_sold"),
    ("Compute inventory velocity = outbound_qty/days và dự báo stockout date",
     "product_id, warehouse_id, current_stock, daily_outbound"),
    ("Group customers thành 3 tiers dựa trên tổng revenue: top 20/30/50%",
     "customer_id, total_revenue"),
    ("Tính pct_of_total: % đóng góp của mỗi region vào tổng revenue toàn công ty",
     "region, monthly_revenue"),
    ("Detect và drop duplicate transactions: cùng customer+product+date+amount",
     "transaction_id, customer_id, product_id, date, amount"),
    ("Tính net revenue sau khi trừ refunds: revenue - refund_amount theo product",
     "product_id, revenue, refund_amount"),
    # ── Inventory analytics ───────────────────────────────────────────────────
    ("Tính tỷ lệ coverage của inventory: kho đáp ứng được bao nhiêu % orders?",
     "warehouse_id, orders_fulfilled, total_orders"),
    ("Tính ABC inventory classification dựa trên annual revenue contribution",
     "product_id, annual_revenue"),
    ("Phát hiện sản phẩm có safety_stock_days < 7 dựa trên demand variance",
     "product_id, avg_daily_demand, demand_std, current_stock"),
    ("Tính fill rate theo warehouse: units shipped / units demanded mỗi tháng",
     "warehouse_id, month, units_demanded, units_shipped"),
    ("So sánh inbound vs outbound movement ratio và tính net_stock_change theo kho",
     "warehouse_id, month, total_inbound, total_outbound"),
    ("Tính days-on-hand = current_stock / avg_daily_demand và flag SKU < 14 ngày",
     "product_id, current_stock, avg_daily_demand"),
    ("Tính inventory coverage in weeks: current_stock / avg_weekly_demand",
     "product_id, current_stock, weekly_demand"),
    ("Aggregate stock movements theo loại và tính net stock change theo ngày",
     "date, movement_type, quantity"),
    ("Phân tích Pareto của warehouse throughput: 20% kho đóng góp bao nhiêu %?",
     "warehouse_id, total_movements"),
    ("Tính tốc độ burn rate inventory: (stock_start - stock_end) / days",
     "product_id, stock_start, stock_end, days"),
    # ── HR analytics ──────────────────────────────────────────────────────────
    ("Tính attrition rate = resigned / start_headcount theo quý và phòng ban",
     "department, quarter, start_headcount, resigned_count"),
    ("Xác định salary compression ratio = highest/lowest trong mỗi level",
     "department, level, salary"),
    ("Tính tổng cost of overtime: overtime_hours × hourly_rate × 1.5",
     "employee_id, department, base_salary, overtime_hours"),
    ("Phân tích pay equity: tính salary gap giữa employees cùng level",
     "level, department, salary"),
    ("Tính total_compensation = base_salary + bonus - deduction mỗi employee Q4",
     "employee_id, base_salary, bonus, deduction"),
    ("Aggregate overtime theo quarter và tính average per employee per quarter",
     "employee_id, quarter, overtime_hours"),
    ("Tính tổng deduction và ratio deduction/base_salary cho mỗi nhân viên Q4",
     "employee_id, base_salary, deduction"),
    ("Phân tích distribution của bonus/base_salary ratio và tính percentiles",
     "employee_id, department, base_salary, bonus"),
    ("Tính số nhân viên mỗi phòng ban theo level và tính % Junior/Mid/Senior",
     "department, level, employee_count"),
    ("Detect salary anomaly: employees cùng level nhưng lương chênh lệch > 2× min",
     "employee_id, level, salary"),
    # ── Financial analysis ────────────────────────────────────────────────────
    ("Tính contribution margin ratio = (revenue - variable_cost) / revenue",
     "product_id, revenue, variable_cost"),
    ("So sánh budget variance (actual vs budget) theo quý và tính % over/under",
     "department, quarter, actual_spend, budget"),
    ("Detect budget overrun risk: months where YTD spend > 90% of annual budget",
     "department, month, ytd_spend, annual_budget"),
    ("Tính tổng profit margin trend theo quý và detect nếu trend giảm dần",
     "quarter, revenue, total_cost"),
    ("Tính actual profit = revenue - COGS - overhead với overhead = 15% COGS",
     "product_id, revenue, cogs"),
    ("Tính monthly_revenue CV = std/mean để đánh giá revenue stability",
     "month, monthly_revenue"),
    ("Phân tích gap analysis: actual revenue vs target revenue theo phòng ban",
     "department, actual_revenue, target_revenue"),
    ("Tính chi phí lưu kho: holding_cost = stock_value × 0.20 / 12 per month",
     "product_id, stock_quantity, unit_cost"),
    ("Tính YTD cumulative profit và YTD variance vs plan",
     "month, actual_profit, planned_profit"),
    ("So sánh variance explained bởi top 3 customers vs phần còn lại",
     "customer_id, region, revenue"),
    # ── Customer & cohort analytics ───────────────────────────────────────────
    ("Tính churn proxy: customers không mua trong 90 ngày = churned",
     "customer_id, last_order_date"),
    ("So sánh cohort retention: customers mua Q1 2024 có mua thêm Q2/Q3/Q4?",
     "customer_id, first_purchase_quarter, all_purchase_quarters"),
    ("Extract top N% customers đóng góp 80% revenue (Pareto principle)",
     "customer_id, total_revenue"),
    ("Tính average invoice gap (days between orders) cho mỗi customer",
     "customer_id, order_date"),
    ("Classify customers: new (1 order), repeat (2–5 orders), loyal (6+ orders)",
     "customer_id, order_count"),
    ("Tính customer lifetime value proxy: avg_order_value × order_frequency",
     "customer_id, avg_order_value, order_count, first_order_date, last_order_date"),
    ("Tính cross-sell rate: % customers mua > 1 category khác nhau cùng period",
     "customer_id, categories_purchased"),
    ("Phân tích revenue seasonality: tính seasonal index mỗi tháng",
     "month, revenue"),
    ("So sánh monthly revenue growth rate của top 3 sản phẩm best-sellers",
     "month, product_id, revenue"),
    ("Detect seasonal inventory shortage: tháng nào có nhiều SKU dưới threshold?",
     "month, product_id, stock, min_threshold"),
    # ── Advanced / miscellaneous ──────────────────────────────────────────────
    ("Tính Lorenz curve points cho phân phối doanh thu theo khách hàng",
     "customer_id, total_revenue"),
    ("So sánh phân phối salary bằng histogram bins và tính percentiles 25/50/75/90",
     "employee_id, salary"),
    ("Tính economic profit: revenue - (cost + overhead) với overhead = 15% revenue",
     "region, revenue, total_cost, order_count"),
    ("Tính inventory turnover ratio = COGS / avg_inventory và rank theo warehouse",
     "warehouse_id, cogs, avg_inventory"),
    ("Tính market share proxy của mỗi category trong tổng revenue",
     "category, revenue"),
    ("Detect unusual orders: amount là bội số chính xác của 1,000,000 (round numbers)",
     "order_id, amount"),
    ("Tính efficiency ratio: revenue per unit inventory held tại mỗi kho",
     "warehouse_id, revenue, avg_stock_held"),
    ("Detect inventory spike: kho nào có incoming > 3× average trong tháng?",
     "warehouse_id, month, incoming_quantity"),
    ("Phân tích power law distribution của customer spending",
     "customer_id, total_spend"),
    ("Tính số tuần liên tiếp revenue tăng liên tục (winning streak)",
     "week, revenue"),
    ("Detect revenue cliff: tháng nào có revenue giảm > 20% so với tháng trước?",
     "month, revenue"),
    ("Tính weighted contribution score của mỗi region: rev×0.5 + growth×0.3 + orders×0.2",
     "region, revenue, growth_rate, order_count"),
    ("Tính số ngày kể từ last_restock và flag SKU > 30 ngày chưa restock",
     "product_id, last_restock_date"),
    ("Tính net promoter score proxy: completed_rate - refunded_rate theo region",
     "region, completed, refunded, total"),
    ("Aggregate daily transactions thành weekly và tính 4-week moving average",
     "date, daily_transactions"),
]

SUPERVISOR_SEEDS: list[str] = [
    # ── Simple queries (route: sql → insight) ───────────────────────────────
    "Tổng doanh thu theo region Q4 2024 là bao nhiêu?",
    "Sản phẩm nào có tồn kho dưới min_threshold hiện tại?",
    "Phòng ban nào có tổng lương cao nhất trong năm 2024?",
    "Nhân viên nào có bonus cao nhất trong Q4 2024?",
    "Tháng nào có doanh thu cao nhất trong năm 2024?",
    "Kho nào có tổng write-off lớn nhất trong năm 2023?",
    "Top 5 sản phẩm bán chạy nhất theo quantity trong Q4 2024",
    "Tổng số đơn hàng cancelled theo region trong năm 2024",
    "Phòng Tài chính có vượt ngân sách không trong Q3 2024?",
    "Sản phẩm Food nào sắp hết hàng tại kho Miền Nam?",
    "Số nhân viên nghỉ việc trong từng quý năm 2024",
    "Tỷ lệ đơn hàng refunded theo region trong năm 2024",
    # ── Analytics queries (route: sql → python → insight) ────────────────────
    "So sánh YoY doanh thu Q4 2024 vs Q4 2023 và phát hiện region bất thường",
    "Phân tích xu hướng overtime của phòng Kinh doanh năm 2024",
    "Khách hàng Enterprise nào đã ngừng mua hàng từ tháng 7/2024?",
    "Region nào có doanh thu cao nhất nhưng workforce thấp nhất?",
    "Sản phẩm Electronics nào có tỷ lệ hoàn trả bất thường tháng 11/2023?",
    "Nhân viên nào nhận bonus bất thường so với peers cùng level?",
    "Tại sao doanh thu Miền Bắc tăng đột biến tháng 3/2024?",
    "Phân tích sản phẩm ELEC-007 có vấn đề pricing gì từ Q2 2024?",
    "Nhân viên level Junior có salary thấp hơn cùng nhóm bao nhiêu?",
    "Kho nào đang có dead stock nhiều nhất và giá trị tồn kho bị đọng?",
    "Phân tích cohort khách hàng mới Q1 2024 có retention rate như thế nào?",
    "Sản phẩm nào có gross margin âm trong Q4 2024?",
    "Category Food có trend doanh thu như thế nào trong 6 tháng cuối 2024?",
    "Nhân viên cấp Junior phòng IT có lương cạnh tranh không so với thị trường?",
    "Tại sao kho Miền Trung có write-off rate cao bất thường trong Q3 2024?",
    "Phòng Tài chính có nhân sự quá tải không so với workload trong Q4 2024?",
    "Doanh thu từ khách hàng loyal (≥6 orders) chiếm bao nhiêu % Q4 2024?",
    "Phân tích payment pattern của khách hàng Enterprise Miền Bắc năm 2024",
    "Sản phẩm nào có demand tăng đột biến trong tháng 11/2024?",
    "Revenue MoM growth của Electronics có dấu hiệu chững lại trong Q4 2024?",
    "Segment SME có tốc độ tăng trưởng như thế nào so với Enterprise năm 2024?",
    "Phân tích salary distribution trong phòng Kỹ thuật có bất thường không?",
    # ── Full pipeline (route: sql → python → viz → insight) ──────────────────
    "Sản phẩm nào đang có tồn kho nguy hiểm và cần nhập hàng gấp?",
    "Top 5 sản phẩm bán chạy nhất Q4 2024 có đủ hàng trong kho không?",
    "So sánh hiệu quả kho theo region: outbound, write-off, và transfer",
    "Kho Tây Nguyên có vấn đề gì nghiêm trọng trong Q4 2023?",
    "Dự báo risk nào cần chú ý Q1 2025 dựa vào trend Q3–Q4/2024?",
    "Tổng hợp các anomalies nổi bật nhất trong toàn bộ dữ liệu năm 2024",
    "Report tổng hợp hiệu quả kinh doanh Q4 2024: doanh thu, tồn kho, nhân sự",
    "So sánh hiệu suất làm việc (overtime per revenue) các phòng ban Q4 2024",
    "Phân tích churn risk: khách hàng nào chưa mua trong 60 ngày qua?",
    "Tổng chi phí nhân sự so với revenue theo phòng ban Q4 2024",
    "Kho nào đang có utilization rate vượt 90%?",
    "Sản phẩm bán chạy nhất và chậm nhất trong mỗi category trong Q4 2024",
    "Revenue trend của top 5 customers: có ai đang giảm mua không trong H2 2024?",
    "Phân tích dead stock value: tổng giá trị hàng tồn kho không di chuyển",
    "So sánh hiệu quả kinh doanh Miền Bắc và Miền Nam theo từng quý 2024",
    # ── Cross-domain analysis ─────────────────────────────────────────────────
    "Category nào có gross margin tốt nhất và tỷ lệ tồn kho đủ nhất?",
    "Danh sách kho nào nên được ưu tiên nhập hàng trong tuần tới?",
    "Xác định phòng ban nào nên được ưu tiên tuyển dụng trong Q1 2025?",
    "Phân tích top 20% customers (VIP): họ mua gì và mua bao nhiêu trong 2024?",
    "Tổng hợp rủi ro inventory Q1 2025: sản phẩm nào cần nhập hàng khẩn?",
    "Phân tích retention revenue: bao nhiêu % revenue Q4 2024 từ khách hàng cũ?",
    "Kho nào có tốc độ xử lý outbound chậm nhất trong Q4 2024?",
    "Tổng hợp Q4 2024 board report: top 3 achievements và top 3 risks",
    "Phân tích warehouse efficiency: outbound per square meter capacity theo kho",
    "Khách hàng nào đang tăng spending nhanh nhất trong 6 tháng cuối 2024?",
    "Kho nào có tổng inbound > tổng outbound trong năm 2024 (tích lũy hàng)?",
    "Phân tích inventory aging: sản phẩm nào tồn kho > 90 ngày chưa di chuyển?",
    "So sánh productivity của từng phòng ban: revenue_contribution / headcount",
    "Tổng hợp tháng 12/2024: doanh thu, nhân sự, tồn kho theo region",
    "Khách hàng SME nào có potential nâng cấp lên Enterprise tier trong Q1 2025?",
    "Doanh thu theo category và region có trend bất thường nào trong Q4 2024?",
    "Phòng ban nào có tỷ lệ bonus/base_salary cao nhất trong Q4 2024?",
    "Xác định pattern bất thường trong payroll tháng 12/2024",
]

INSIGHT_SEEDS: list[dict] = [
    {
        "query": "So sánh doanh thu Q4 2024 vs Q4 2023 theo region",
        "sql": "SELECT region, SUM(amount) rev_2024 FROM orders WHERE year=2024 AND quarter=4 GROUP BY region",
        "stats": "Miền Bắc: +88% YoY; Miền Nam: +19%; Miền Trung: +22%; Tây Nguyên: +15%; mean_yoy=+23%",
        "anomalies": "Miền Bắc is_anomaly=True (3.2 sigma above mean, positive_outlier)",
    },
    {
        "query": "Sản phẩm nào có tỷ lệ refund cao bất thường tháng 11/2023?",
        "sql": "SELECT p.sku, COUNT(*) refund_count FROM orders o JOIN products p ON o.product_id=p.id WHERE o.status='refunded' AND year=2023 AND EXTRACT(MONTH FROM order_date)=11 GROUP BY p.sku",
        "stats": "Electronics tháng 11/2023: 62 refunds vs baseline 18/month; refund_rate=24.8% vs 7.2%",
        "anomalies": "Electronics category tháng 11/2023: negative_outlier, 3.4x baseline",
    },
    {
        "query": "Phòng Tài chính vượt ngân sách không trong Q3 2024?",
        "sql": "SELECT p.month, SUM(p.net_salary) total FROM payroll p JOIN employees e ON p.employee_id=e.id JOIN departments d ON e.department_id=d.id WHERE d.name='Tài chính' AND p.year=2024 GROUP BY p.month",
        "stats": "Budget: 320M/month; T7: 380M; T8: 384M; T9: 376M; avg_overrun=+18.7%",
        "anomalies": "3 tháng liên tiếp vượt ngân sách (negative: budget_overrun)",
    },
    {
        "query": "Tình trạng tồn kho hiện tại theo region",
        "sql": "SELECT w.region, COUNT(*) below_threshold FROM stock s JOIN warehouses w ON s.warehouse_id=w.id WHERE s.quantity < s.min_threshold GROUP BY w.region",
        "stats": "Miền Bắc: 8 SKUs dưới threshold; Miền Nam: 3; Miền Trung: 1; Tây Nguyên: 0",
        "anomalies": "Miền Bắc có số SKUs nguy hiểm cao nhất: 8/80 sản phẩm (10%)",
    },
    {
        "query": "Nhân viên có bonus bất thường tháng 12 2024?",
        "sql": "SELECT e.name, e.level, p.bonus, p.base_salary, p.bonus/p.base_salary ratio FROM payroll p JOIN employees e ON p.employee_id=e.id WHERE p.month=12 AND p.year=2024 AND p.bonus/p.base_salary > 2",
        "stats": "4 nhân viên với bonus_ratio 3.6x–4.7x; mean_ratio cùng level: 0.15x",
        "anomalies": "2 có performance_score cao (justified); 2 không có lý do rõ ràng",
    },
    {
        "query": "Kho nào có write-off rate bất thường trong năm 2023?",
        "sql": "SELECT w.name, COUNT(*) write_offs, SUM(sm.quantity) total_qty FROM stock_movements sm JOIN warehouses w ON sm.from_warehouse_id=w.id WHERE sm.movement_type='write_off' AND EXTRACT(YEAR FROM sm.movement_date)=2023 GROUP BY w.name",
        "stats": "Kho Tây Nguyên: 145 write-offs (8,200 units); Kho Miền Bắc: 23 (1,100 units); Kho Miền Nam: 31 (1,850 units); mean=66 write-offs",
        "anomalies": "Kho Tây Nguyên vượt 2.7 sigma (positive_outlier) — write-off gấp 4.5x median",
    },
    {
        "query": "Phòng nào vượt ngân sách nhiều nhất trong năm 2024?",
        "sql": "SELECT d.name, SUM(p.net_salary) total_payroll, d.budget*12 annual_budget FROM payroll p JOIN employees e ON p.employee_id=e.id JOIN departments d ON e.department_id=d.id WHERE p.year=2024 GROUP BY d.name, d.budget",
        "stats": "Phòng Kinh doanh: thực tế 4.8B vs ngân sách 4.2B (+14.3%); Phòng IT: 3.2B vs 3.0B (+6.7%); Phòng Tài chính: 2.1B vs 2.3B (-8.7%)",
        "anomalies": "Phòng Kinh doanh có budget overrun lớn nhất (+14.3%); Phòng Tài chính tiết kiệm 8.7%",
    },
    {
        "query": "Sản phẩm Electronics nào có gross margin âm trong Q4 2024?",
        "sql": "SELECT p.sku, p.name, AVG(p.unit_price) avg_price, AVG(p.cost) avg_cost, (AVG(p.unit_price)-AVG(p.cost))/AVG(p.unit_price)*100 margin_pct FROM orders o JOIN products p ON o.product_id=p.id WHERE p.category='Electronics' AND o.year=2024 AND o.quarter=4 GROUP BY p.sku, p.name",
        "stats": "ELEC-007: margin = -3.2% (unit_price=285,000 vs cost=295,000); ELEC-012: margin = -1.8%; avg positive margin = +34%",
        "anomalies": "2 sản phẩm bán dưới giá vốn — ELEC-007 active từ tháng 4/2024 (price override error)",
    },
    {
        "query": "Tồn kho tại kho Miền Bắc có đủ cho nhu cầu Q1 2025?",
        "sql": "SELECT p.name, s.quantity current_stock, AVG(sm.quantity) avg_monthly_outbound FROM stock s JOIN products p ON s.product_id=p.id JOIN warehouses w ON s.warehouse_id=w.id LEFT JOIN stock_movements sm ON sm.product_id=p.id AND sm.from_warehouse_id=w.id WHERE w.region='Miền Bắc' GROUP BY p.name, s.quantity",
        "stats": "12 SKUs với coverage < 30 ngày; 8 SKUs với coverage < 14 ngày (critical); avg coverage = 45 ngày",
        "anomalies": "8 SKUs ở mức critical (<14 ngày); Electronics chiếm 6/8 critical SKUs",
    },
    {
        "query": "Category nào có tốc độ tăng trưởng doanh thu cao nhất trong năm 2024?",
        "sql": "SELECT p.category, SUM(CASE WHEN o.year=2024 THEN o.amount ELSE 0 END) rev_2024, SUM(CASE WHEN o.year=2023 THEN o.amount ELSE 0 END) rev_2023 FROM orders o JOIN products p ON o.product_id=p.id GROUP BY p.category",
        "stats": "Electronics: 2024=8.2B vs 2023=4.1B (+100%); Food: 3.1B vs 2.8B (+10.7%); Clothing: 2.4B vs 2.5B (-4%)",
        "anomalies": "Electronics tăng trưởng 100% YoY — outlier dương; Clothing giảm nhẹ — cần theo dõi",
    },
    {
        "query": "Region nào có tỷ lệ đơn hàng cancelled cao nhất trong Q3 2024?",
        "sql": "SELECT region, COUNT(*) total, COUNT(*) FILTER(WHERE status='cancelled') cancelled, COUNT(*) FILTER(WHERE status='cancelled')::float/COUNT(*) cancel_rate FROM orders WHERE year=2024 AND quarter=3 GROUP BY region",
        "stats": "Tây Nguyên: 18.2% cancel rate (68/374 orders); Miền Bắc: 7.1%; Miền Nam: 8.4%; avg=10.7%",
        "anomalies": "Tây Nguyên có cancel rate gấp 2× average — negative_outlier, cần điều tra nguyên nhân",
    },
    {
        "query": "Sản phẩm Food nào sắp hết hàng tại kho Miền Nam?",
        "sql": "SELECT p.name, s.quantity, s.min_threshold FROM stock s JOIN products p ON s.product_id=p.id JOIN warehouses w ON s.warehouse_id=w.id WHERE p.category='Food' AND w.region='Miền Nam' AND s.quantity <= s.min_threshold * 1.2",
        "stats": "7 sản phẩm Food dưới 120% min_threshold; 3 sản phẩm đã dưới threshold; avg safety_margin = -142 units",
        "anomalies": "3 SKUs đã breach threshold — immediate restock cần thiết trong 2–3 ngày",
    },
    {
        "query": "Khách hàng Enterprise nào có dấu hiệu churn trong Q4 2024?",
        "sql": "SELECT c.name, c.region, MAX(o.order_date) last_order, COUNT(*) order_count FROM customers c JOIN orders o ON c.id=o.customer_id WHERE c.segment='Enterprise' GROUP BY c.name, c.region HAVING MAX(o.order_date) < CURRENT_DATE - INTERVAL '60 days'",
        "stats": "8 Enterprise customers không mua trong 60+ ngày; top 3 có combined revenue Q3 2024 = 1.2B VND",
        "anomalies": "Top churn risk: Công ty A (180 ngày không mua), Công ty B (95 ngày), Công ty C (72 ngày)",
    },
    {
        "query": "Overtime tháng 3/2024 phòng Kinh doanh có bất thường không?",
        "sql": "SELECT e.name, p.month, SUM(p.overtime_hours) overtime_hours FROM payroll p JOIN employees e ON p.employee_id=e.id JOIN departments d ON e.department_id=d.id WHERE d.name='Kinh doanh' AND p.year=2024 AND p.month BETWEEN 1 AND 6 GROUP BY e.name, p.month",
        "stats": "Tháng 3: avg overtime 42h/employee (benchmark: 18h); tháng 1-2: 16–20h; tháng 4–6: 19–23h",
        "anomalies": "Tháng 3/2024 có overtime spike 2.3× benchmark — có thể do campaign launch hoặc system migration",
    },
    {
        "query": "Kho Miền Bắc tiếp nhận hàng trong Q4 2024 có đủ capacity không?",
        "sql": "SELECT w.name, w.capacity, SUM(s.quantity) current_stock FROM warehouses w JOIN stock s ON s.warehouse_id=w.id WHERE w.region='Miền Bắc' GROUP BY w.name, w.capacity",
        "stats": "Kho HN-01: 82% capacity (current=41,000/50,000); Kho HN-02: 91% capacity (27,300/30,000)",
        "anomalies": "Kho HN-02 ở mức critical utilization (91%) — cần redistribution trước Tết 2025",
    },
    {
        "query": "Sản phẩm nào có demand tăng đột biến trong tháng 11/2024?",
        "sql": "SELECT p.name, p.category, COUNT(*) FILTER(WHERE EXTRACT(MONTH FROM o.order_date)=11) orders_nov, COUNT(*) FILTER(WHERE EXTRACT(MONTH FROM o.order_date)=10) orders_oct FROM orders o JOIN products p ON o.product_id=p.id WHERE o.year=2024 AND EXTRACT(MONTH FROM o.order_date) IN (10,11) GROUP BY p.name, p.category",
        "stats": "ELEC-003: orders T11=342 vs T10=89 (+284%); CLOT-005: 156 vs 140 (+11%); avg spike=+45%",
        "anomalies": "ELEC-003 có demand spike 284% — viral marketing hoặc competitor stockout",
    },
    {
        "query": "Phòng IT có headcount phù hợp với workload không trong Q4 2024?",
        "sql": "SELECT d.name, d.headcount, AVG(p.overtime_hours) avg_overtime FROM departments d JOIN employees e ON e.department_id=d.id JOIN payroll p ON p.employee_id=e.id WHERE p.year=2024 AND p.month BETWEEN 10 AND 12 GROUP BY d.name, d.headcount",
        "stats": "Phòng IT: avg overtime 38h/tháng/người; Phòng Kinh doanh: 22h; Phòng Tài chính: 11h; benchmark: 20h",
        "anomalies": "Phòng IT có overtime gấp 1.9× benchmark — dấu hiệu understaffing hoặc project overload",
    },
    {
        "query": "Doanh thu theo region có trend seasonal rõ ràng không trong năm 2024?",
        "sql": "SELECT region, EXTRACT(MONTH FROM order_date) month, SUM(amount) revenue FROM orders WHERE year=2024 GROUP BY region, month ORDER BY region, month",
        "stats": "Tất cả regions: Q4 spike rõ ràng (+35–60% vs Q2); Miền Bắc có Q4 peak mạnh nhất (+58%); Tây Nguyên flat nhất (variance=±12%)",
        "anomalies": "Seasonal pattern rõ ràng với Q4 peak — cần inventory planning trước 8–10 tuần",
    },
    {
        "query": "AOV (average order value) thay đổi thế nào từ 2023 sang 2024 theo region?",
        "sql": "SELECT year, region, COUNT(*) order_count, SUM(amount) revenue, SUM(amount)/COUNT(*) avg_order_value FROM orders WHERE year IN (2023, 2024) GROUP BY year, region",
        "stats": "AOV 2024: 2,850,000 VND vs 2023: 2,340,000 VND (+21.8%); Miền Bắc AOV tăng mạnh nhất (+32%)",
        "anomalies": "AOV tăng 21.8% YoY trên tất cả regions — upselling hiệu quả hoặc shift sang higher-value products",
    },
    {
        "query": "Segment nào có revenue per customer cao nhất và growth rate tốt nhất?",
        "sql": "SELECT c.segment, COUNT(DISTINCT o.customer_id) customers, SUM(o.amount) total_revenue, SUM(o.amount)/COUNT(DISTINCT o.customer_id) rev_per_customer FROM customers c JOIN orders o ON c.id=o.customer_id WHERE o.year=2024 GROUP BY c.segment",
        "stats": "Enterprise: rev/customer=8.4M, 42 customers, total=353M (+18% YoY); SME: 2.1M, 185 customers, 389M (+31% YoY)",
        "anomalies": "SME segment có growth rate cao hơn Enterprise (+31% vs +18%) — tiềm năng upmarket conversion",
    },
    {
        "query": "Kho nào cần maintenance dựa trên error rate cao nhất trong 2024?",
        "sql": "SELECT w.name, COUNT(*) FILTER(WHERE sm.movement_type='write_off') write_offs, COUNT(*) total_movements FROM warehouses w JOIN stock_movements sm ON sm.from_warehouse_id=w.id WHERE EXTRACT(YEAR FROM sm.movement_date)=2024 GROUP BY w.name",
        "stats": "Kho TN-01: error_rate=4.2% (highest); Kho MB-01: 0.8%; avg=1.9%; Kho TN-01 có 127 write-offs trên 3,024 total",
        "anomalies": "Kho TN-01 có error rate gấp 2.2× average — storage conditions hoặc handling process issue",
    },
    {
        "query": "Top 5 sản phẩm bán chạy Q4 2024 có đủ tồn kho cho tháng 1/2025 không?",
        "sql": "SELECT p.name, SUM(o.quantity) q4_qty_sold, s.quantity current_stock FROM orders o JOIN products p ON o.product_id=p.id JOIN stock s ON s.product_id=p.id WHERE o.year=2024 AND o.quarter=4 GROUP BY p.name, s.quantity ORDER BY q4_qty_sold DESC LIMIT 5",
        "stats": "ELEC-003: 1,024 units sold Q4, 420 units in stock (0.41 months coverage); CLOT-002: 876 sold, 1,200 stock (1.37 months)",
        "anomalies": "ELEC-003 chỉ có 0.41 tháng coverage — cần nhập thêm 600+ units trước 15/1/2025",
    },
    {
        "query": "Revenue từ top 20% khách hàng chiếm bao nhiêu phần trăm tổng 2024?",
        "sql": "WITH ranked AS (SELECT customer_id, SUM(amount) revenue, NTILE(5) OVER (ORDER BY SUM(amount) DESC) tier FROM orders WHERE year=2024 GROUP BY customer_id) SELECT tier, COUNT(*) customer_count, SUM(revenue) tier_revenue FROM ranked GROUP BY tier ORDER BY tier",
        "stats": "Top 20% (tier 1): 295 customers, 4.2B revenue (68.5% of total); tier 2: 22.1%; tiers 3–5: 9.4%",
        "anomalies": "Concentration risk: top 20% = 68.5% revenue — healthy premium focus but risky nếu key accounts churn",
    },
    {
        "query": "Doanh thu SME Miền Trung có xu hướng gì trong H2 2024?",
        "sql": "SELECT EXTRACT(MONTH FROM o.order_date) month, SUM(o.amount) revenue FROM orders o JOIN customers c ON o.customer_id=c.id WHERE c.segment='SME' AND c.region='Miền Trung' AND o.year=2024 AND EXTRACT(MONTH FROM o.order_date) BETWEEN 7 AND 12 GROUP BY month ORDER BY month",
        "stats": "T7: 45M; T8: 42M; T9: 38M; T10: 35M; T11: 31M; T12: 28M — declining trend -38% từ T7 đến T12",
        "anomalies": "Consistent declining trend 6 tháng liên tiếp (-38%) — structural issue cần sales investigation",
    },
    {
        "query": "Kho Tây Nguyên có vấn đề gì với inventory movement trong năm 2024?",
        "sql": "SELECT movement_type, COUNT(*) count, SUM(quantity) total_qty FROM stock_movements WHERE from_warehouse_id IN (SELECT id FROM warehouses WHERE region='Tây Nguyên') AND EXTRACT(YEAR FROM movement_date)=2024 GROUP BY movement_type",
        "stats": "write_off: 89 movements (4,200 units); outbound: 312 (15,600 units); inbound: 178 (8,900 units); net=-10,900 units",
        "anomalies": "Write-off rate = 28.5% of outbound — 3× industry benchmark; net negative stock unsustainable",
    },
    {
        "query": "Electronics revenue MoM growth có dấu hiệu chững lại trong Q4 2024?",
        "sql": "SELECT EXTRACT(MONTH FROM o.order_date) month, SUM(o.amount) revenue FROM orders o JOIN products p ON o.product_id=p.id WHERE p.category='Electronics' AND o.year=2024 AND o.quarter=4 GROUP BY month ORDER BY month",
        "stats": "T10: 2.1B; T11: 2.3B (+9.5%); T12: 2.2B (-4.3%); avg Q4 MoM=+2.6%; Q3 avg MoM=+8.4%",
        "anomalies": "T12 giảm 4.3% MoM sau khi tăng 2 tháng — possible demand pull-forward, theo dõi T1/2025",
    },
    {
        "query": "Phòng Kinh doanh có đạt mục tiêu tăng trưởng Q4 2024 không?",
        "sql": "SELECT region, SUM(amount) q4_2024_revenue FROM orders WHERE year=2024 AND quarter=4 GROUP BY region",
        "stats": "Tổng Q4 2024: 6.1B vs Q4 2023: 4.8B (+27.1%); Miền Bắc +42%, Miền Nam +21%, Miền Trung +18%; target=+20%",
        "anomalies": "Tổng vượt target (+27.1%); Miền Bắc dẫn đầu +42% (outlier dương, 2.1 sigma above mean)",
    },
    {
        "query": "Tổng chi phí payroll Q4 2024 có trong ngân sách không?",
        "sql": "SELECT d.name, SUM(p.base_salary + p.bonus - p.deduction) actual_cost, d.budget budget FROM payroll p JOIN employees e ON p.employee_id=e.id JOIN departments d ON e.department_id=d.id WHERE p.year=2024 AND p.month BETWEEN 10 AND 12 GROUP BY d.name, d.budget",
        "stats": "Tổng payroll Q4: 3.2B vs budget 3.0B (+6.7%); Kinh doanh: +12%, IT: +3%, Tài chính: -5%",
        "anomalies": "Phòng Kinh doanh over-budget 12% trong Q4 — tương quan với commission cao do vượt target",
    },
    {
        "query": "SKU nào tại kho Miền Nam đang ở mức tồn kho nguy hiểm?",
        "sql": "SELECT p.name, p.category, s.quantity, s.min_threshold FROM stock s JOIN products p ON s.product_id=p.id JOIN warehouses w ON s.warehouse_id=w.id WHERE w.region='Miền Nam' AND s.quantity < s.min_threshold * 1.5",
        "stats": "15 SKUs dưới 1.5× threshold; 5 dưới 1× (critical breach); FOOD-023: coverage=0.3×, ELEC-008: 0.6×",
        "anomalies": "5 critical SKUs cần immediate restock; FOOD-023 với coverage 0.3× có thể stockout trong 2–3 ngày",
    },
    {
        "query": "Nhân viên nào có risk bỏ việc cao dựa trên salary compression và tenure?",
        "sql": "SELECT e.name, e.level, e.salary, AVG(e2.salary) OVER (PARTITION BY e.level) peer_avg FROM employees e JOIN employees e2 ON e.department_id=e2.department_id AND e.level=e2.level WHERE e.status='active'",
        "stats": "12 nhân viên với tenure > 2 năm và salary < 85% peer average; 5 có tenure > 4 năm; combined experience = 62 person-years",
        "anomalies": "5 long-tenured employees với salary compressed — highest flight risk profile",
    },
    {
        "query": "Hiệu quả vận chuyển (avg transfer size) của kho nào tốt nhất trong 2024?",
        "sql": "SELECT w_to.region, COUNT(*) transfer_count, SUM(sm.quantity) total_units, AVG(sm.quantity) avg_transfer_size FROM stock_movements sm JOIN warehouses w_to ON sm.to_warehouse_id=w_to.id WHERE sm.movement_type='transfer' AND EXTRACT(YEAR FROM sm.movement_date)=2024 GROUP BY w_to.region",
        "stats": "Miền Bắc: avg transfer size=420 units (most efficient); Tây Nguyên: 85 units (least efficient); avg=215 units",
        "anomalies": "Kho Tây Nguyên có avg transfer size nhỏ nhất (85 vs avg 215) — fragmented, inefficient deliveries",
    },
    {
        "query": "Revenue per SKU có phân phối đều hay tập trung vào vài sản phẩm?",
        "sql": "SELECT p.name, p.category, SUM(o.amount) revenue, RANK() OVER (ORDER BY SUM(o.amount) DESC) rnk FROM orders o JOIN products p ON o.product_id=p.id WHERE o.year=2024 GROUP BY p.name, p.category",
        "stats": "Top 10 SKUs đóng góp 62% tổng revenue; top 20 SKUs = 78%; total active SKUs = 147",
        "anomalies": "Moderate concentration (top 10 = 62%) — healthy diversity, không có extreme single-product dependency",
    },
    {
        "query": "Miền Trung có cải thiện inventory health trong năm 2024 không?",
        "sql": "SELECT EXTRACT(QUARTER FROM s.last_updated) quarter, COUNT(*) FILTER(WHERE s.quantity < s.min_threshold)::float/COUNT(*) critical_rate FROM stock s JOIN warehouses w ON s.warehouse_id=w.id WHERE w.region='Miền Trung' AND EXTRACT(YEAR FROM s.last_updated)=2024 GROUP BY quarter ORDER BY quarter",
        "stats": "Q1: critical_rate=18%; Q2: 22%; Q3: 15%; Q4: 11%; trend cải thiện trong H2 2024",
        "anomalies": "Q2 là điểm xấu nhất (22%) — supply chain disruption; Q4 cải thiện về 11% là tín hiệu tích cực",
    },
    {
        "query": "Phân tích tổng hợp Q4 2024: revenue, inventory risk, HR cost có balanced không?",
        "sql": "SELECT 'Revenue Q4 2024' metric, SUM(amount)::text value FROM orders WHERE year=2024 AND quarter=4 UNION ALL SELECT 'Critical Stock SKUs', COUNT(*)::text FROM stock WHERE quantity < min_threshold",
        "stats": "Revenue Q4: 6.1B VND; Critical Stock: 23 SKUs; HR Payroll Q4: 3.2B VND; HR/Revenue ratio = 52.5%",
        "anomalies": "HR/Revenue ratio 52.5% cao hơn Q3 (48%) nhưng trong ngưỡng chấp nhận do Q4 bonuses; 23 critical SKUs cần attention",
    },
    {
        "query": "Tỷ lệ refund của Clothing có bất thường trong Q4 2024 không?",
        "sql": "SELECT p.category, COUNT(*) total, COUNT(*) FILTER(WHERE o.status='refunded') refunds, COUNT(*) FILTER(WHERE o.status='refunded')::float/COUNT(*) refund_rate FROM orders o JOIN products p ON o.product_id=p.id WHERE o.year=2024 AND o.quarter=4 GROUP BY p.category",
        "stats": "Clothing Q4 2024: refund_rate=14.2% (89/627 orders); Electronics: 8.1%; Food: 3.2%; avg=8.5%",
        "anomalies": "Clothing refund rate 14.2% — 1.7× category average; 10 specific SKUs chiếm 72% refunds",
    },
    {
        "query": "Phòng nào có attrition rate cao nhất trong năm 2024?",
        "sql": "SELECT d.name, COUNT(*) FILTER(WHERE e.status='resigned') resigned, COUNT(*) total, COUNT(*) FILTER(WHERE e.status='resigned')::float/COUNT(*) attrition_rate FROM departments d JOIN employees e ON e.department_id=d.id WHERE EXTRACT(YEAR FROM e.hire_date) >= 2020 GROUP BY d.name",
        "stats": "Phòng Tài chính: 26.7% attrition (4/15 resigned); Phòng IT: 12.5% (2/16); avg=9.8%; benchmark=8%",
        "anomalies": "Phòng Tài chính có attrition 26.7% — gấp 3.3× benchmark; immediate HR investigation needed",
    },
    {
        "query": "Tăng trưởng revenue của SME vs Enterprise theo quý 2024 có xu hướng gì?",
        "sql": "SELECT c.segment, EXTRACT(QUARTER FROM o.order_date) quarter, SUM(o.amount) revenue FROM orders o JOIN customers c ON o.customer_id=c.id WHERE o.year=2024 GROUP BY c.segment, quarter ORDER BY c.segment, quarter",
        "stats": "SME: Q1=89M, Q2=102M, Q3=118M, Q4=145M (+63% YTD); Enterprise: Q1=320M, Q2=315M, Q3=310M, Q4=353M (+10% YTD)",
        "anomalies": "SME tăng trưởng 63% YTD vs Enterprise +10% — SME đang catch up; Q3 Enterprise dip đáng chú ý",
    },
    {
        "query": "Kho nào có throughput (outbound/capacity) cao nhất trong Q4 2024?",
        "sql": "SELECT w.name, w.region, w.capacity, SUM(sm.quantity) total_outbound, SUM(sm.quantity)::float/w.capacity throughput_ratio FROM warehouses w JOIN stock_movements sm ON sm.from_warehouse_id=w.id WHERE sm.movement_type='outbound' AND EXTRACT(QUARTER FROM sm.movement_date)=4 AND EXTRACT(YEAR FROM sm.movement_date)=2024 GROUP BY w.name, w.region, w.capacity",
        "stats": "Kho MB-01: throughput=0.85 (42,500/50,000); Kho MN-02: 0.91 (27,300/30,000); Kho TN-01: 0.32 (9,600/30,000)",
        "anomalies": "Kho TN-01 có throughput thấp nhất (0.32) — ngược với high utilization, suggests inefficient outbound processes",
    },
    {
        "query": "Sản phẩm nào có inventory turnover thấp nhất (dead stock risk)?",
        "sql": "SELECT p.name, p.category, s.quantity, AVG(sm.quantity) avg_monthly_outbound, s.quantity/NULLIF(AVG(sm.quantity),0) months_on_hand FROM stock s JOIN products p ON s.product_id=p.id LEFT JOIN stock_movements sm ON sm.product_id=p.id AND sm.movement_type='outbound' GROUP BY p.name, p.category, s.quantity",
        "stats": "CLOT-014: 18.4 months on hand (1,840 units, avg outbound 100/month); avg=3.2 months; 8 SKUs > 12 months",
        "anomalies": "8 SKUs có > 12 months on hand — dead stock risk, tổng giá trị tồn kho bị đọng ước tính 450M VND",
    },
    {
        "query": "Lương nhân viên phòng Kỹ thuật có tương xứng với tenure không?",
        "sql": "SELECT e.name, e.level, e.salary, DATE_PART('year', AGE(e.hire_date)) tenure_years FROM employees e JOIN departments d ON e.department_id=d.id WHERE d.name='Kỹ thuật' AND e.status='active' ORDER BY tenure_years DESC",
        "stats": "Senior với tenure > 5 năm: avg salary=35M; Senior với tenure < 2 năm: avg salary=38M; gap=-8.1%",
        "anomalies": "Salary inversion: newer Senior employees earn more than veterans — likely due to market rate adjustments, flight risk for tenured staff",
    },
    {
        "query": "Revenue concentration: 10 khách hàng lớn nhất chiếm bao nhiêu % tổng Q4 2024?",
        "sql": "SELECT c.name, SUM(o.amount) revenue, RANK() OVER (ORDER BY SUM(o.amount) DESC) rnk FROM orders o JOIN customers c ON o.customer_id=c.id WHERE o.year=2024 AND o.quarter=4 GROUP BY c.name ORDER BY revenue DESC LIMIT 10",
        "stats": "Top 10 customers: combined revenue 1.83B / total 6.1B = 30.0%; top 1 customer alone = 8.2% (502M)",
        "anomalies": "Top 1 customer = 8.2% revenue — single customer dependency risk; top 10 = 30% which is manageable",
    },
    {
        "query": "Số ngày hàng tồn kho (Days Inventory Outstanding) của Electronics là bao nhiêu?",
        "sql": "SELECT p.name, s.quantity current_stock, AVG(sm.quantity) avg_daily_outbound, s.quantity/NULLIF(AVG(sm.quantity)/30.0,0) dio FROM stock s JOIN products p ON s.product_id=p.id JOIN stock_movements sm ON sm.product_id=p.id WHERE p.category='Electronics' AND sm.movement_type='outbound' GROUP BY p.name, s.quantity",
        "stats": "Electronics avg DIO=42 days; ELEC-003: 12 days (critical low); ELEC-009: 187 days (excess); benchmark=30 days",
        "anomalies": "2 extreme outliers: ELEC-003 (stockout risk <2 weeks) và ELEC-009 (6 months excess). Cần rebalancing",
    },
    {
        "query": "Phòng Kinh doanh có overtime cost vượt ngân sách trong Q4 2024 không?",
        "sql": "SELECT e.name, SUM(p.overtime_hours) total_overtime, SUM(p.base_salary)/12 monthly_salary, SUM(p.overtime_hours) * SUM(p.base_salary)/12/160 * 1.5 overtime_cost FROM employees e JOIN payroll p ON p.employee_id=e.id JOIN departments d ON e.department_id=d.id WHERE d.name='Kinh doanh' AND p.year=2024 AND p.month BETWEEN 10 AND 12 GROUP BY e.name",
        "stats": "Tổng overtime cost Q4 phòng KD: 245M VND; budget cho overtime: 180M; overage = +36%; 3 nhân viên > 80h overtime",
        "anomalies": "Overtime cost vượt 36% budget — tương quan với Q4 target achievement; top 3 nhân viên cần workload review",
    },
    {
        "query": "Stock movement ratio (inbound/outbound) theo tháng có trend bất thường không?",
        "sql": "SELECT EXTRACT(MONTH FROM movement_date) month, EXTRACT(YEAR FROM movement_date) year, SUM(CASE WHEN movement_type='inbound' THEN quantity ELSE 0 END) inbound, SUM(CASE WHEN movement_type='outbound' THEN quantity ELSE 0 END) outbound FROM stock_movements WHERE EXTRACT(YEAR FROM movement_date) IN (2023,2024) GROUP BY year, month ORDER BY year, month",
        "stats": "Avg ratio 2024: 0.89 (inbound/outbound); T11 2024: 1.47 (spike inbound); T12 2023: 0.52 (depletion peak)",
        "anomalies": "T11 2024 inbound spike (ratio=1.47) — holiday restocking, expected; T12 2023 depletion (0.52) caused Q1 2024 stockouts",
    },
    {
        "query": "Top 5 sản phẩm có contribution margin cao nhất trong năm 2024?",
        "sql": "SELECT p.name, p.category, SUM(o.amount) revenue, SUM(o.quantity * p.cost) total_cost, SUM(o.amount) - SUM(o.quantity * p.cost) contribution_margin, (SUM(o.amount) - SUM(o.quantity * p.cost))/SUM(o.amount) cm_ratio FROM orders o JOIN products p ON o.product_id=p.id WHERE o.year=2024 GROUP BY p.name, p.category, p.cost ORDER BY contribution_margin DESC LIMIT 5",
        "stats": "Top 5 CM: ELEC-001 (CM=1.2B, ratio=48%), ELEC-003 (0.9B, 42%), CLOT-002 (0.7B, 38%); bottom 5 avg CM_ratio=12%",
        "anomalies": "Top 5 products đóng góp 3.8B/6.1B Q4 revenue (62%) và có CM ratio cao — priority products for inventory protection",
    },
    {
        "query": "Nhân viên mới tuyển trong Q3 2024 phòng IT có performance ổn không?",
        "sql": "SELECT e.name, e.level, e.hire_date, AVG(p.bonus/p.base_salary) avg_bonus_ratio, SUM(p.overtime_hours) total_overtime FROM employees e JOIN payroll p ON p.employee_id=e.id JOIN departments d ON e.department_id=d.id WHERE d.name IN ('IT','Kỹ thuật') AND EXTRACT(QUARTER FROM e.hire_date)=3 AND EXTRACT(YEAR FROM e.hire_date)=2024 AND p.year=2024 GROUP BY e.name, e.level, e.hire_date",
        "stats": "8 nhân viên IT mới Q3: avg bonus_ratio=0.18x; avg overtime=42h/person Q4; peer average (existing staff): ratio=0.24x, overtime=38h",
        "anomalies": "New hires có overtime cao hơn slightly (+11%) nhưng bonus thấp hơn (-25%) — onboarding ramp-up period, expected",
    },
    {
        "query": "Doanh thu channel trực tuyến (status=completed trước D+2) vs truyền thống trong 2024?",
        "sql": "SELECT region, EXTRACT(QUARTER FROM order_date) quarter, COUNT(*) total_orders, SUM(amount) total_revenue FROM orders WHERE year=2024 AND status='completed' GROUP BY region, quarter ORDER BY region, quarter",
        "stats": "Miền Bắc Q4: 1,240 completed orders, 2.1B revenue; Miền Nam: 890 orders, 1.6B; Tây Nguyên: 234 orders, 0.38B",
        "anomalies": "Miền Bắc có completion rate và revenue per order cao nhất — strong operational excellence or favorable customer mix",
    },
    {
        "query": "Số lượng SKU hết hàng (stockout) tại mỗi kho trong Q4 2024 là bao nhiêu?",
        "sql": "SELECT w.name, w.region, COUNT(*) FILTER(WHERE s.quantity = 0) stockout_count, COUNT(*) total_skus FROM stock s JOIN warehouses w ON s.warehouse_id=w.id WHERE s.last_updated >= '2024-10-01' GROUP BY w.name, w.region",
        "stats": "Kho MB-01: 3 stockouts/80 SKUs (3.75%); Kho MN-01: 7/65 (10.8%); Kho TN-01: 12/45 (26.7%); avg=12.4%",
        "anomalies": "Kho TN-01 có stockout rate 26.7% — 2.2× average; critical supply chain failure affecting 12 SKUs",
    },
]

REFLECTOR_SEEDS: list[dict] = [
    {
        "failed_agent": "sql",
        "error_type": "sql_syntax",
        "traceback": "ERROR: column \"month\" does not exist\nLINE 3: WHERE month = 3",
        "previous_attempt": "SELECT region, SUM(amount) FROM orders WHERE month = 3 AND year = 2024 GROUP BY region",
    },
    {
        "failed_agent": "sql",
        "error_type": "sql_logic",
        "traceback": "ERROR: column \"quarter\" of relation \"orders\" is generated\nDETAIL: Generated columns cannot be updated.",
        "previous_attempt": "INSERT INTO orders (customer_id, product_id, region, amount, order_date, quarter) VALUES (1, 1, 'Miền Bắc', 100000, '2024-01-15', 1)",
    },
    {
        "failed_agent": "python",
        "error_type": "python_runtime",
        "traceback": "KeyError: 'yoy_pct'\ndf['is_anomaly'] = df['yoy_pct'] > threshold",
        "previous_attempt": "threshold = df['yoy_pct'].mean() + 2*df['yoy_pct'].std()\ndf['is_anomaly'] = df['yoy_pct'] > threshold",
    },
    {
        "failed_agent": "python",
        "error_type": "data_quality",
        "traceback": "ValueError: Cannot convert NaN to integer\ndf['quantity'] = df['quantity'].astype(int)",
        "previous_attempt": "df['quantity'] = df['quantity'].astype(int)",
    },
    {
        "failed_agent": "sql",
        "error_type": "schema_mismatch",
        "traceback": "ERROR: column \"department\" does not exist\nHINT: Did you mean \"department_id\"?",
        "previous_attempt": "SELECT department, AVG(salary) FROM employees GROUP BY department",
    },
]

REFLECTOR_PROMPT = """\
## ROLE
You are an error diagnosis specialist for a multi-agent data pipeline.

A specialist agent failed. Diagnose the error and provide a corrected context.

Failed agent: {failed_agent}
Error type: {error_type}
Traceback: {traceback}
Previous attempt: {previous_attempt}

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
  "root_cause": "<what went wrong>",
  "error_category": "<sql_syntax|sql_logic|python_runtime|python_logic|data_quality|schema_mismatch>",
  "fix_strategy": "<how to fix it>",
  "corrected_context": "<the corrected SQL or Python code>"
}}
"""


# =============================================================================
# API CALLER
# =============================================================================

def call_api(
    client: OpenAI,
    system: str,
    user: str,
    skill_type: str,
) -> str | None:
    """Call GPT-4o-mini with retry logic. Returns assistant content or None."""
    temp = SKILL_TEMPERATURES.get(skill_type, 0.5)
    max_tokens = SKILL_MAX_TOKENS.get(skill_type, 1024)
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=temp,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"      rate limit — waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
            elif attempt < RETRY_ATTEMPTS - 1:
                print(f"      API error (attempt {attempt+1}): {err[:60]}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"      API error after {RETRY_ATTEMPTS} attempts: {err[:80]}")
                return None
    return None


# =============================================================================
# SAMPLE BUILDERS — one per skill type
# =============================================================================

def _strip_fence(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    import re
    return re.sub(r"```(?:json|sql|python)?\s*|\s*```", "", text).strip()


# Try to salvage SQL when model accidentally prepends prose.
def _extract_sql_candidate(text: str) -> str:
    cleaned = _strip_fence(text)
    upper = cleaned.upper()
    idx_with = upper.find("WITH ")
    idx_select = upper.find("SELECT ")
    candidates = [i for i in (idx_with, idx_select) if i >= 0]
    if not candidates:
        return cleaned
    start = min(candidates)
    sql = cleaned[start:].strip()
    # Keep one SQL statement only; trim trailing explanation if model appended it.
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
    return sql


# ── SQL validation helpers (Fix 1) ───────────────────────────────────────────

def _get_db_conn():
    """Return a psycopg2 connection. Returns None if DATABASE_URL not set."""
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    try:
        return psycopg2.connect(url)
    except Exception:
        return None


def _validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate SQL syntax and planner-level correctness using EXPLAIN.

    EXPLAIN does a full parse + planning pass without executing the query,
    so it catches:
      - syntax errors (missing column, wrong operator, etc.)
      - reference errors (table/column does not exist)
      - type mismatches caught at planning time

    It does NOT catch logic errors or empty result sets.

    Returns (is_valid: bool, reason: str).
    """
    conn = _get_db_conn()
    if conn is None:
        # No DB available — skip validation, accept the sample optimistically.
        # validate_dataset.py will do the real execution check later.
        return True, "db_unavailable"

    try:
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN {sql}")
        conn.rollback()   # never commit anything
        return True, "ok"
    except psycopg2.Error as e:
        conn.rollback()
        return False, str(e).splitlines()[0][:120]
    finally:
        conn.close()


def _heuristic_sql_checks(sql: str) -> tuple[bool, str]:
    """
    Fast heuristic checks that catch common generation mistakes BEFORE
    hitting the database. These run even when DB is unavailable.

    Returns (is_valid: bool, reason: str).
    """
    import re
    s = sql.upper()

    # Must be a SELECT (not INSERT/UPDATE/DELETE/DROP)
    first_kw = s.lstrip().split()[0] if sql.strip() else ""
    if first_kw not in ("SELECT", "WITH", "EXPLAIN"):
        return False, f"non-SELECT statement: {first_kw}"

    # Must not reference non-existent 'month' column on orders directly.
    # Pattern: "orders" ... "month" where month is used as a bare column
    # (not inside EXTRACT or a subquery alias).
    # Simple heuristic: flag if "month" appears as a standalone word near
    # orders without being wrapped in EXTRACT(
    if re.search(r'\bMONTH\s*=', s) and "EXTRACT" not in s:
        return False, "bare 'month' column on orders — use EXTRACT(MONTH FROM order_date)"

    # Must not INSERT into generated columns quarter/year
    if "INSERT" in s and re.search(r'\b(quarter|year)\b', s):
        return False, "INSERT into generated column quarter or year"

    return True, "ok"


def _rewrite_orders_month_filter(sql: str) -> str:
    """
    Best-effort rewrite:
    - When query targets orders and uses bare month comparison, rewrite
      `month = X` to `EXTRACT(MONTH FROM order_date) = X`.
    This keeps payroll queries untouched.
    """
    if "orders" not in sql.lower():
        return sql
    rewritten = re.sub(
        r"(?i)\bmonth\s*=\s*(\d{1,2})\b",
        r"EXTRACT(MONTH FROM order_date) = \1",
        sql,
    )
    return rewritten


def _has_absolute_comparison_item(evidence: list[str]) -> bool:
    """
    True when at least one evidence item contains a side-by-side comparison
    with 2 non-percentage numbers (e.g., "142,000,000 VND vs 75,400,000 VND"
    or "62 vs 18").
    """
    num_re = re.compile(r"\d+(?:[.,]\d+)?")
    pct_re = re.compile(r"^\d+(?:[.,]\d+)?%?$")
    for item in evidence:
        lower = item.lower()
        if not any(tok in lower for tok in (" vs ", " so với ", " compared to ", " so sanh ", " so sánh ")):
            continue
        tokens = [m.group(0) for m in num_re.finditer(item)]
        non_pct_tokens = []
        for tok in tokens:
            # Skip percentage-like token such as "24.8%" (allow "24.8" if not percent)
            around = item[item.find(tok): item.find(tok) + len(tok) + 1]
            next_char = item[item.find(tok) + len(tok): item.find(tok) + len(tok) + 1]
            if next_char == "%":   # chỉ filter khi % thực sự xuất hiện ngay sau số
                continue
            non_pct_tokens.append(tok)
        if len(non_pct_tokens) >= 2:
            return True
    return False


def _build_forced_comparison_from_stats(stats_text: str) -> str | None:
    """
    Build a minimal absolute comparison evidence line from stats text.
    Avoids years and percentage-like tokens. Returns None if not enough numbers.
    """
    num_re = re.compile(r"\d+(?:[.,]\d+)?")
    candidates: list[str] = []
    for m in num_re.finditer(stats_text):
        token = m.group(0)
        next_char = stats_text[m.end(): m.end() + 1]
        if next_char == "%":
            continue
        # Skip likely year tokens to avoid useless "2024 vs 2023"
        try:
            normalized = float(token.replace(",", ""))
            if len(token) == 4 and 1900 <= normalized <= 2100:
                continue
        except Exception:
            pass
        candidates.append(token)
        if len(candidates) >= 2:
            break

    if len(candidates) < 2:
        return None
    return f"So sánh số liệu tuyệt đối từ thống kê: {candidates[0]} vs {candidates[1]}."


def build_sql_sample(
    client: OpenAI,
    info_box: str,
    prompt_template: str,
    seed: str,
) -> dict | None:
    """
    Fix 1: Validate generated SQL with:
      1. Heuristic checks (fast, no DB needed)
      2. EXPLAIN on PostgreSQL (catches syntax + schema errors)
    Reject the sample if either check fails.
    """
    system = prompt_template.replace("{info_box}", info_box).replace("{task}", seed)
    user_prompt = seed
    last_reason = "unknown"
    sql = None

    for attempt in range(SQL_SELF_REPAIR_ATTEMPTS + 1):
        raw = call_api(client, system, user_prompt, "text-to-sql")
        if not raw:
            return None
        sql = _extract_sql_candidate(raw)
        sql = _rewrite_orders_month_filter(sql)

        # Step 1: heuristic checks
        ok, reason = _heuristic_sql_checks(sql)
        if not ok:
            last_reason = reason
            if attempt < SQL_SELF_REPAIR_ATTEMPTS:
                user_prompt = (
                    f"Task: {seed}\n\n"
                    f"Previous output failed validation: {reason}\n"
                    "Rewrite and return ONLY one valid PostgreSQL query.\n"
                    "First token MUST be SELECT or WITH. No explanation."
                )
                continue
            print(f"      sql heuristic failed: {reason}")
            return None

        # Step 2: EXPLAIN validation
        ok, reason = _validate_sql(sql)
        if not ok:
            last_reason = reason
            if attempt < SQL_SELF_REPAIR_ATTEMPTS:
                user_prompt = (
                    f"Task: {seed}\n\n"
                    f"Previous SQL failed EXPLAIN: {reason}\n"
                    "Fix the SQL and return ONLY one valid PostgreSQL query."
                )
                continue
            print(f"      sql EXPLAIN failed: {reason}")
            return None
        break

    if not sql:
        print(f"      sql generation failed after retries: {last_reason}")
        return None

    return {
        "skill_type": "text-to-sql",
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": seed},
            {"role": "assistant", "content": sql},
        ],
        "validation": reason,   # "ok" or "db_unavailable"
    }


def _validate_python(code: str) -> tuple[bool, str]:
    """
    Validate generated Python code structurally without executing it.

    Checks (in order):
      1. Parses as valid Python AST
      2. References 'df' (the expected input variable)
      3. Does NOT use forbidden imports (os, sys, open, requests, subprocess)
      4. Last statement is an expression (returns a value), not a print/None
      5. Uses df.copy() or creates a new variable (no silent in-place mutation)

    Returns (is_valid: bool, reason: str).
    """
    import ast
    import re

    # 1. AST parse
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # 2. Must reference df
    code_lower = code.lower()
    if "df" not in code_lower:
        return False, "code does not reference 'df' input variable"

    # 3. Forbidden imports / calls
    FORBIDDEN = ("import os", "import sys", "open(", "import requests",
                 "import subprocess", "__import__", "eval(", "exec(")
    for f in FORBIDDEN:
        if f in code:
            return False, f"forbidden usage: {f!r}"

    # 3b. If code uses pd/np/stats, require explicit imports to avoid runtime NameError
    imported_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_aliases.add(alias.asname or alias.name)

    used_names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    required_alias_imports = ("pd", "np", "stats")
    for alias in required_alias_imports:
        if alias in used_names and alias not in imported_aliases:
            return False, f"uses '{alias}' but missing explicit import alias"

    # 4. Last statement must be an expression (not print, assignment to None, etc.)
    if not tree.body:
        return False, "empty code body"
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        # Reject if the expression is a call to a void function
        # (print, display, show do not return a DataFrame)
        _VOID_CALLS = {"print", "display", "show", "pprint", "logging"}
        if isinstance(last.value, ast.Call):
            called = last.value.func
            fname = getattr(called, 'id', None) or getattr(called, 'attr', None)
            if fname in _VOID_CALLS:
                return False, (
                    f"last statement is a call to '{fname}()' which returns None. "
                    "End with the result DataFrame as a bare expression, e.g. 'result' or 'df'."
                )
    elif isinstance(last, ast.Assign):
        # Last line is an assignment — acceptable if it assigns to 'result' or 'df'
        targets = [getattr(t, 'id', '') for t in last.targets]
        if not any(t in ('result', 'df') for t in targets):
            return False, (
                "last statement is an assignment but not to 'result' or 'df'. "
                "The last line must evaluate to the result DataFrame."
            )
    else:
        return False, (
            f"last statement is {type(last).__name__}, not an expression. "
            "Code must end with the result DataFrame as the last expression."
        )

    # 5. Should not modify df in-place without copy
    # Heuristic: if code does df[...] = ... AND df.copy() not present, warn
    has_inplace = bool(re.search(r"df\[.+\]\s*=", code))
    has_copy    = "df.copy()" in code or "df = df" in code
    if has_inplace and not has_copy:
        return False, "modifies df in-place without df.copy() — add df = df.copy() first"

    return True, "ok"


def build_python_sample(
    client: OpenAI,
    prompt_template: str,
    task: str,
    columns: str,
) -> dict | None:
    """
    Fix 1: Validate generated Python code structurally before accepting.
    Checks: valid AST, references df, no forbidden imports,
    last statement returns a value, no silent in-place mutation.
    """
    system = (
        prompt_template
        .replace("{columns}", columns)
        .replace("{sample}", "// sample rows not available at generation time")
        .replace("{task}", task)
    )
    code = call_api(client, system, task, "data-analysis")
    if not code:
        return None
    code = _strip_fence(code)

    ok, reason = _validate_python(code)
    if not ok:
        print(f"      python validation failed: {reason}")
        return None

    return {
        "skill_type": "data-analysis",
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": task},
            {"role": "assistant", "content": code},
        ],
    }


def build_supervisor_sample(
    client: OpenAI,
    info_box: str,
    prompt_template: str,
    query: str,
) -> dict | None:
    from schemas.plan_schema import ExecutionPlan

    system = prompt_template.replace("{info_box}", info_box).replace("{query}", query)
    user_prompt = query
    raw = None
    plan_obj = None
    last_err = "unknown"

    for attempt in range(SUPERVISOR_SELF_REPAIR_ATTEMPTS + 1):
        raw = call_api(client, system, user_prompt, "supervisor-routing")
        if not raw:
            return None
        raw = _strip_fence(raw)

        # Validate with Pydantic first
        try:
            plan_obj = ExecutionPlan.model_validate_json(raw)
        except Exception as e:
            last_err = str(e).splitlines()[0][:180]
            if attempt < SUPERVISOR_SELF_REPAIR_ATTEMPTS:
                user_prompt = (
                    f"User query: {query}\n\n"
                    f"Previous plan failed schema validation: {last_err}\n"
                    "Regenerate a valid ExecutionPlan JSON.\n"
                    "Hard constraints: unique agents, valid dependencies, insight last."
                )
                continue
            print(f"      schema rejected: {last_err}")
            return None

        # Guardrail: sql task must be instruction text, not SQL literal.
        sql_keywords = (
            "SELECT ", " WITH ", " FROM ", " JOIN ", " WHERE ",
            " GROUP BY ", " ORDER BY ", " HAVING ", " LIMIT ",
        )
        leaked_sql = False
        for step in plan_obj.steps:
            if step.agent != "sql":
                continue
            task_upper = f" {step.task.strip().upper()} "
            hits = sum(kw in task_upper for kw in sql_keywords)
            if task_upper.startswith(("SELECT ", "WITH ")) or hits >= 2:
                leaked_sql = True
                break

        if leaked_sql:
            last_err = "sql task contains SQL literal"
            if attempt < SUPERVISOR_SELF_REPAIR_ATTEMPTS:
                user_prompt = (
                    f"User query: {query}\n\n"
                    "Previous plan invalid: sql step task contains SQL literal.\n"
                    "Rewrite with natural-language task instructions only."
                )
                continue
            print("      schema rejected: supervisor sql task contains SQL literal")
            return None

        # Passed all checks
        break

    if not raw or plan_obj is None:
        print(f"      schema rejected: {last_err}")
        return None

    return {
        "skill_type": "supervisor-routing",
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": query},
            {"role": "assistant", "content": raw},
        ],
    }


def build_insight_sample(
    client: OpenAI,
    prompt_template: str,
    seed: dict,
) -> dict | None:
    from schemas.insight_schema import InsightOutput

    system = (
        prompt_template
        .replace("{query}",    seed["query"])
        .replace("{sql}",      seed["sql"])
        .replace("{stats}",    seed["stats"])
        .replace("{anomalies}", seed["anomalies"])
    )
    user_prompt = (
        f"{seed['query']}\n\n"
        "Return ONLY valid JSON.\n"
        "Mandatory: evidence[0] must contain an absolute comparison in this form: "
        "'<metric>: <A> vs <B>'."
    )
    raw = None
    insight = None
    last_err = "unknown"

    for attempt in range(INSIGHT_SELF_REPAIR_ATTEMPTS + 1):
        raw = call_api(client, system, user_prompt, "insight-generation")
        if not raw:
            return None
        raw = _strip_fence(raw)

        try:
            insight = InsightOutput.model_validate_json(raw)
        except Exception as e:
            last_err = str(e).splitlines()[0][:180]
            if attempt < INSIGHT_SELF_REPAIR_ATTEMPTS:
                user_prompt = (
                    f"Original query: {seed['query']}\n\n"
                    f"Previous output failed schema validation: {last_err}\n"
                    "Regenerate valid JSON with exactly 2-3 evidence items."
                )
                continue
            print(f"      schema rejected: {last_err}")
            return None

        # Business-quality gate: require at least one side-by-side absolute comparison item.
        if not _has_absolute_comparison_item(insight.evidence):
            # Deterministic fallback: inject one absolute comparison line from stats.
            forced = _build_forced_comparison_from_stats(seed["stats"])
            if forced:
                patched = insight.model_dump()
                rest = [e for e in patched["evidence"] if e != forced]
                patched["evidence"] = [forced] + rest
                patched["evidence"] = patched["evidence"][:3]
                try:
                    insight = InsightOutput.model_validate(patched)
                    raw = json.dumps(patched, ensure_ascii=False)
                    break
                except Exception:
                    pass

            last_err = "evidence lacks absolute comparison pair (A vs B)"
            if attempt < INSIGHT_SELF_REPAIR_ATTEMPTS:
                user_prompt = (
                    f"Original query: {seed['query']}\n\n"
                    "Previous output failed quality gate: evidence must include at least one "
                    "side-by-side absolute comparison (e.g., '142,000,000 VND vs 75,400,000 VND' "
                    "or '62 vs 18').\n"
                    f"Use this context line if needed: {seed['stats']}\n"
                    "Rewrite JSON accordingly."
                )
                continue
            print("      schema rejected: evidence lacks absolute comparison pair (A vs B)")
            return None
        break

    if not raw or insight is None:
        print(f"      schema rejected: {last_err}")
        return None

    return {
        "skill_type": "insight-generation",
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": seed["query"]},
            {"role": "assistant", "content": raw},
        ],
    }


def build_reflector_sample(
    client: OpenAI,
    seed: dict,
) -> dict | None:
    system = REFLECTOR_PROMPT.format(**seed)
    raw = call_api(client, system, seed["traceback"], "error-reflection")
    if not raw:
        return None
    raw = _strip_fence(raw)

    # Light validation — must be parseable JSON with required keys
    try:
        obj = json.loads(raw)
        assert all(k in obj for k in ("root_cause", "error_category",
                                       "fix_strategy", "corrected_context"))
    except Exception as e:
        print(f"      reflector json invalid: {e}")
        return None

    return {
        "skill_type": "error-reflection",
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": seed["traceback"]},
            {"role": "assistant", "content": raw},
        ],
    }


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate(
    skill: str | None = None,
    count: int | None = None,
    resume: bool = True,
) -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts once
    sql_prompt        = load_prompt("text_to_sql")
    python_prompt     = load_prompt("data_analysis")
    supervisor_prompt = load_prompt("supervisor_routing")
    insight_prompt    = load_prompt("insight_generation")

    # Load info-boxes
    info_sales    = load_info_box("sales")
    info_inv      = load_info_box("inventory")
    info_hr       = load_info_box("hr")
    info_all      = load_all_info_box()
    # Slim info_box for Supervisor: drops sample_rows/indexes (~973 tokens vs ~6 270)
    info_slim     = load_slim_info_box()

    # Build targets for this run
    targets = {k: v for k, v in TARGETS.items()
               if skill is None or k == skill}
    if count:
        targets = {k: count for k in targets}

    # Resume: count existing records per skill_type
    existing: dict[str, int] = {k: 0 for k in targets}
    if resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    st = obj.get("skill_type", "")
                    # raw file stores skill_type values (e.g. "text-to-sql"),
                    # while targets/existing use internal keys (e.g. "sql").
                    key = SKILL_TYPE_TO_KEY.get(st, st)  # keep backward compatibility
                    if key in existing:
                        existing[key] += 1
                except Exception:
                    pass
        print("Resuming from existing file:")
        for k, v in existing.items():
            print(f"  {k}: {v}/{targets.get(k,0)} already done")

    total_generated = 0
    total_needed = sum(max(0, targets[k] - existing[k]) for k in targets)
    print(f"\nNeed to generate: {total_needed} samples\n")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:

        # ── SQL samples ───────────────────────────────────────────────────────
        if "sql" in targets:
            needed = targets["sql"] - existing["sql"]
            if needed > 0:
                _mr_sql = _max_reuse("sql")
                print(f"[SQL] Generating {needed} samples … (seeds={len(SQL_SEEDS)}, max_reuse={_mr_sql})")
                seeds_cycle = SQL_SEEDS * (needed // len(SQL_SEEDS) + 2)
                random.shuffle(seeds_cycle)
                generated = 0
                seed_use_count: dict[str, int] = {}
                INFO_BOX_MAP = {
                    "sales":     info_sales,
                    "inventory": info_inv,
                    "hr":        info_hr,
                    "cross":     info_all,
                }
                for seed, domain in seeds_cycle:
                    if generated >= needed:
                        break
                    if seed_use_count.get(seed, 0) >= _mr_sql:
                        continue
                    ib = INFO_BOX_MAP.get(domain, info_all)
                    sample = build_sql_sample(client, ib, sql_prompt, seed)
                    if sample:
                        # Issue 3: tag cross-domain records for val/test exclusion
                        if domain == "cross":
                            sample["is_cross_domain"] = True
                        seed_use_count[seed] = seed_use_count.get(seed, 0) + 1
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        out.flush()
                        generated += 1
                        total_generated += 1
                        if generated % 50 == 0:
                            print(f"  sql: {generated}/{needed}")
                    _rate_limit_sleep(total_generated)
                print(f"  ✓ sql done: {generated} samples")

        # ── Python samples ────────────────────────────────────────────────────
        if "python" in targets:
            needed = targets["python"] - existing["python"]
            if needed > 0:
                _mr_py = _max_reuse("python")
                print(f"[PYTHON] Generating {needed} samples … (seeds={len(PYTHON_SEEDS)}, max_reuse={_mr_py})")
                seeds_cycle = PYTHON_SEEDS * (needed // len(PYTHON_SEEDS) + 2)
                random.shuffle(seeds_cycle)
                generated = 0
                seed_use_count_py: dict[str, int] = {}
                for task, cols in seeds_cycle:
                    if generated >= needed:
                        break
                    if seed_use_count_py.get(task, 0) >= _mr_py:
                        continue
                    sample = build_python_sample(client, python_prompt, task, cols)
                    if sample:
                        seed_use_count_py[task] = seed_use_count_py.get(task, 0) + 1
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        out.flush()
                        generated += 1
                        total_generated += 1
                        if generated % 50 == 0:
                            print(f"  python: {generated}/{needed}")
                    _rate_limit_sleep(total_generated)
                print(f"  ✓ python done: {generated} samples")

        # ── Supervisor samples ────────────────────────────────────────────────
        if "supervisor" in targets:
            needed = targets["supervisor"] - existing["supervisor"]
            if needed > 0:
                _mr_sv = _max_reuse("supervisor")
                print(f"[SUPERVISOR] Generating {needed} samples … (seeds={len(SUPERVISOR_SEEDS)}, max_reuse={_mr_sv})")
                seeds_cycle = SUPERVISOR_SEEDS * (needed // len(SUPERVISOR_SEEDS) + 2)
                random.shuffle(seeds_cycle)
                generated = 0
                seed_use_count_sv: dict[str, int] = {}
                for query in seeds_cycle:
                    if generated >= needed:
                        break
                    if seed_use_count_sv.get(query, 0) >= _mr_sv:
                        continue
                    # Issue 2: use slim info_box (~973 tokens) instead of full (~6 270 tokens)
                    sample = build_supervisor_sample(client, info_slim, supervisor_prompt, query)
                    if sample:
                        seed_use_count_sv[query] = seed_use_count_sv.get(query, 0) + 1
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        out.flush()
                        generated += 1
                        total_generated += 1
                        if generated % 25 == 0:
                            print(f"  supervisor: {generated}/{needed}")
                    _rate_limit_sleep(total_generated)
                print(f"  ✓ supervisor done: {generated} samples")

        # ── Insight samples ───────────────────────────────────────────────────
        if "insight" in targets:
            needed = targets["insight"] - existing["insight"]
            if needed > 0:
                _mr_ins = _max_reuse("insight")
                print(f"[INSIGHT] Generating {needed} samples … (seeds={len(INSIGHT_SEEDS)}, max_reuse={_mr_ins})")
                seeds_cycle = INSIGHT_SEEDS * (needed // len(INSIGHT_SEEDS) + 2)
                random.shuffle(seeds_cycle)
                generated = 0
                seed_use_count_ins: dict[str, int] = {}
                for seed in seeds_cycle:
                    if generated >= needed:
                        break
                    seed_key = seed["query"]
                    if seed_use_count_ins.get(seed_key, 0) >= _mr_ins:
                        continue
                    sample = build_insight_sample(client, insight_prompt, seed)
                    if sample:
                        seed_use_count_ins[seed_key] = seed_use_count_ins.get(seed_key, 0) + 1
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        out.flush()
                        generated += 1
                        total_generated += 1
                        if generated % 25 == 0:
                            print(f"  insight: {generated}/{needed}")
                    _rate_limit_sleep(total_generated)
                print(f"  ✓ insight done: {generated} samples")

        # ── Reflector samples ─────────────────────────────────────────────────
        if "reflector" in targets:
            needed = targets["reflector"] - existing["reflector"]
            if needed > 0:
                print(f"[REFLECTOR] Generating {needed} samples …")
                seeds_cycle = REFLECTOR_SEEDS * (needed // len(REFLECTOR_SEEDS) + 2)
                random.shuffle(seeds_cycle)
                generated = 0
                for seed in seeds_cycle:
                    if generated >= needed:
                        break
                    sample = build_reflector_sample(client, seed)
                    if sample:
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        out.flush()
                        generated += 1
                        total_generated += 1
                    _rate_limit_sleep(total_generated)
                print(f"  ✓ reflector done: {generated} samples")

    # Final count
    print(f"\n{'='*45}")
    print(f"Total generated this run: {total_generated}")
    if OUTPUT_FILE.exists():
        total_lines = sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8"))
        print(f"Total in file:           {total_lines}")
        by_skill: dict[str, int] = {}
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    st = json.loads(line).get("skill_type", "unknown")
                    by_skill[st] = by_skill.get(st, 0) + 1
                except Exception:
                    pass
        for st, n in sorted(by_skill.items()):
            print(f"  {st:25s}: {n}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print("Next step: python training/validate_dataset.py")


def _rate_limit_sleep(total: int) -> None:
    """Sleep to stay under REQUESTS_PER_MINUTE."""
    # Simple: sleep 60/RPM seconds between each call
    time.sleep(60.0 / REQUESTS_PER_MINUTE)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ADBA training data")
    parser.add_argument("--skill",  type=str, default=None,
                        choices=list(TARGETS.keys()),
                        help="Generate only this skill type (default: all)")
    parser.add_argument("--count",  type=int, default=None,
                        help="Override count for selected skill")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh even if output file exists")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("✗ OPENAI_API_KEY not set. Export it before running.")
        sys.exit(1)

    generate(
        skill=args.skill,
        count=args.count,
        resume=not args.no_resume,
    )