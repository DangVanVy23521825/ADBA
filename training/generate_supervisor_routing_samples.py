"""
Generate deterministic supervisor-routing ShareGPT samples for M2 fine-tuning.

Output format follows .cursorrules:
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "supervisor_routing.txt"
OUT_PATH = ROOT / "data" / "supervisor_routing_samples.jsonl"

INFO_BOX = {
    "schemas": {
        "sales": {
            "tables": {
                "orders": ["id", "customer_id", "product_id", "region", "amount", "quantity", "order_date", "quarter", "year", "status"],
                "products": ["id", "name", "category", "unit_price", "cost"],
                "customers": ["id", "name", "email", "city", "segment", "region"],
            }
        },
        "inventory": {
            "tables": {
                "stock": ["id", "product_id", "warehouse_id", "quantity", "min_threshold", "last_updated"],
                "warehouses": ["id", "name", "city", "region", "capacity"],
                "stock_movements": ["id", "product_id", "from_warehouse_id", "to_warehouse_id", "quantity", "movement_date", "movement_type"],
            }
        },
        "hr": {
            "tables": {
                "employees": ["id", "name", "department_id", "role", "level", "salary", "hire_date", "status"],
                "departments": ["id", "name", "budget", "headcount", "manager_id"],
                "payroll": ["id", "employee_id", "month", "year", "base_salary", "bonus", "deduction", "net_salary"],
            }
        },
    }
}


def step(num: int, agent: str, task: str, depends_on: list[str], skill_type: str) -> dict:
    return {"step": num, "agent": agent, "task": task, "depends_on": depends_on, "skill_type": skill_type}


def simple_plan(topic: str, metric: str, segment: str, period: str) -> dict:
    return {
        "plan_summary": f"Tổng hợp {metric} theo {segment} trong {period} và tạo insight điều hành",
        "steps": [
            step(1, "sql", f"Lấy {metric} theo {segment} cho {period}, chỉ giữ dữ liệu hợp lệ và sắp xếp giảm dần", [], "text-to-sql"),
            step(2, "insight", f"Xác định {segment} nổi bật nhất về {metric}, nêu bằng chứng định lượng và đề xuất hành động", ["sql"], "insight-generation"),
        ],
    }


def analysis_plan(topic: str, metric: str, segment: str, period: str, comparison: str) -> dict:
    return {
        "plan_summary": f"So sánh {metric} {comparison} theo {segment}, phát hiện bất thường và tạo insight",
        "steps": [
            step(1, "sql", f"Lấy {metric} hiện tại và kỳ so sánh theo {segment} cho {period}, chuẩn hóa thành các cột current_value và previous_value", [], "text-to-sql"),
            step(2, "python", f"Tính phần trăm thay đổi {comparison}, thống kê trung bình/độ lệch chuẩn và thêm cột is_anomaly", ["sql"], "data-analysis"),
            step(3, "insight", f"Tóm tắt {segment} tăng/giảm mạnh nhất, lượng hóa mức lệch và đưa ra khuyến nghị cụ thể", ["sql", "python"], "insight-generation"),
        ],
    }


def viz_plan(topic: str, metric: str, segment: str, period: str, comparison: str) -> dict:
    return {
        "plan_summary": f"Phân tích và trực quan hóa {metric} {comparison} theo {segment}",
        "steps": [
            step(1, "sql", f"Lấy dữ liệu {metric} theo {segment} cho {period} và kỳ so sánh liên quan", [], "text-to-sql"),
            step(2, "python", f"Tính {comparison}, phát hiện outlier bằng sigma và chuẩn bị bảng kết quả cho biểu đồ", ["sql"], "data-analysis"),
            step(3, "viz", f"Vẽ biểu đồ so sánh {metric} theo {segment}, highlight các điểm is_anomaly=True", ["python"], "visualization"),
            step(4, "insight", f"Diễn giải biểu đồ, nêu anomaly quan trọng nhất và đề xuất hành động theo {topic}", ["sql", "python", "viz"], "insight-generation"),
        ],
    }


def build_samples() -> list[dict]:
    system = PROMPT_PATH.read_text(encoding="utf-8").replace("{info_box}", json.dumps(INFO_BOX, ensure_ascii=False))
    domains = [
        ("sales", "doanh thu", "region"),
        ("sales", "số lượng đơn hàng", "customer segment"),
        ("sales", "biên lợi nhuận", "product category"),
        ("inventory", "tồn kho", "warehouse"),
        ("inventory", "stock movement", "region"),
        ("hr", "net payroll", "department"),
        ("hr", "headcount", "role level"),
        ("hr", "bonus", "department"),
    ]
    periods = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024", "năm 2024"]
    comparisons = ["YoY", "QoQ"]
    samples: list[dict] = []

    for i in range(200):
        topic, metric, segment = domains[i % len(domains)]
        period = periods[(i // len(domains)) % len(periods)]
        comparison = comparisons[i % len(comparisons)]

        if i % 4 == 0:
            query = f"Tổng hợp {metric} theo {segment} trong {period} và cho biết điểm đáng chú ý nhất"
            plan = simple_plan(topic, metric, segment, period)
        elif i % 4 == 1:
            query = f"So sánh {metric} {comparison} theo {segment} trong {period}, tìm nhóm biến động bất thường"
            plan = analysis_plan(topic, metric, segment, period, comparison)
        else:
            query = f"Phân tích {metric} {comparison} theo {segment} trong {period}, vẽ biểu đồ và đề xuất hành động"
            plan = viz_plan(topic, metric, segment, period, comparison)

        samples.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
            ]
        })
    return samples


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for sample in build_samples():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote 200 supervisor-routing samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
