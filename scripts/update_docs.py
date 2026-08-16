#!/usr/bin/env python3
"""
ADBA — Bộ sinh tài liệu tự động
================================
Đọc trạng thái thật của repo (git log, mã nguồn, schema SQL, kết quả eval) và ghi
lại nội dung vào các **khối AUTO** trong `docs/`.

Khối AUTO có dạng:

    <!-- AUTO:begin id=db-tables -->
    ...nội dung do script sinh, đừng sửa tay...
    <!-- AUTO:end id=db-tables -->

Mọi thứ nằm ngoài cặp marker là văn bản do người viết và không bao giờ bị đụng tới.

Cách dùng
---------
    python scripts/update_docs.py            # ghi lại các khối AUTO
    python scripts/update_docs.py --check    # không ghi; exit 1 nếu tài liệu lỗi thời (dùng cho CI)
    python scripts/update_docs.py --list     # liệt kê id khối và nơi chúng xuất hiện

Chỉ dùng thư viện chuẩn — chạy được ở CI trước khi cài requirements.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"

# Commit do chính script này tạo ra. Bị loại khỏi changelog và khỏi "commit nguồn
# gần nhất" để việc chạy lại script là idempotent — nếu không, mỗi commit tài liệu
# lại làm tài liệu lỗi thời và hook sẽ tự kích hoạt vô hạn.
AUTO_COMMIT_PREFIX = "docs(auto):"

MARKER_RE = re.compile(
    r"(?P<begin><!-- AUTO:begin id=(?P<id>[a-z0-9\-]+) -->)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- AUTO:end id=(?P=id) -->)",
    re.DOTALL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tiện ích
# ─────────────────────────────────────────────────────────────────────────────

def git(*args: str) -> str:
    """Chạy lệnh git trong repo; trả về stdout đã strip, chuỗi rỗng nếu lỗi."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def tracked_files(*patterns: str) -> list[Path]:
    """Danh sách file được git theo dõi khớp pattern (bỏ .venv, worktree, …)."""
    raw = git("ls-files", "-z", *patterns)
    if not raw:
        return []
    return [ROOT / p for p in raw.split("\0") if p]


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def line_count(path: Path) -> int:
    text = read(path)
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def table(headers: list[str], rows: list[list[str]], empty: str = "_Không có dữ liệu._") -> str:
    if not rows:
        return empty
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def log_records(rev_range: str | None = None, limit: int | None = None) -> list[dict[str, str]]:
    """Commit (đã loại commit tài liệu tự động), mới nhất trước."""
    fmt = "%H%x1f%h%x1f%an%x1f%ad%x1f%s"
    # `--extended-regexp` là bắt buộc: mặc định git dùng regex cơ bản, ở đó `\(` nghĩa là
    # mở nhóm chứ không phải dấu ngoặc — bộ lọc `docs\(auto\):` sẽ không bao giờ khớp.
    args = ["log", f"--pretty=format:{fmt}", "--date=short", "--no-merges",
            "--extended-regexp", "--invert-grep", f"--grep=^{re.escape(AUTO_COMMIT_PREFIX)}"]
    if limit:
        args.append(f"-n{limit}")
    if rev_range:
        args.append(rev_range)
    raw = git(*args)
    records = []
    for line in raw.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 5:
            records.append(dict(zip(("sha", "short", "author", "date", "subject"), parts)))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Khối: stamp — dấu vết commit nguồn
# ─────────────────────────────────────────────────────────────────────────────

def gen_stamp() -> str:
    recs = log_records(limit=1)
    if not recs:
        return "_Không đọc được lịch sử git._"
    c = recs[0]
    # Đếm bằng log đã lọc, KHÔNG dùng `git rev-list --count HEAD`: con số đó tăng thêm
    # sau mỗi commit `docs(auto)` nên khối stamp lỗi thời ngay khi hook vừa chạy xong.
    total = len(log_records())
    rows = [
        ["Commit nguồn gần nhất", f"`{c['short']}` — {c['subject']}"],
        ["Tác giả", c["author"]],
        ["Ngày commit", c["date"]],
        ["Số commit nguồn", str(total)],
        ["Sinh bởi", "`scripts/update_docs.py` (hook `post-commit`)"],
    ]
    return table(["Trường", "Giá trị"], rows)


# ─────────────────────────────────────────────────────────────────────────────
# Khối: changelog + commit-history
# ─────────────────────────────────────────────────────────────────────────────

COMMIT_TYPES: list[tuple[tuple[str, ...], str]] = [
    (("feat",), "Tính năng mới"),
    (("fix",), "Sửa lỗi"),
    (("perf",), "Hiệu năng"),
    (("refactor",), "Tái cấu trúc"),
    (("docs",), "Tài liệu"),
    (("test",), "Kiểm thử"),
    (("build", "ci", "chore"), "Hạ tầng & công cụ"),
]


def _classify(subject: str) -> str:
    head = subject.split(":", 1)[0].split("(", 1)[0].strip().lower()
    for prefixes, label in COMMIT_TYPES:
        if head in prefixes:
            return label
    return "Khác"


def _render_commits(records: list[dict[str, str]]) -> str:
    if not records:
        return "_Không có thay đổi._"
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for c in records:
        grouped[_classify(c["subject"])].append(c)
    order = [label for _, label in COMMIT_TYPES] + ["Khác"]
    out = []
    for label in order:
        items = grouped.get(label)
        if not items:
            continue
        out.append(f"**{label}**")
        out.append("")
        for c in items:
            subject = c["subject"]
            if ":" in subject.split(" ", 1)[0]:
                subject = subject.split(":", 1)[1].strip()
            out.append(f"- {subject} — `{c['short']}` ({c['date']}, {c['author']})")
        out.append("")
    return "\n".join(out).strip()


def gen_changelog() -> str:
    tags = [t for t in git("tag", "--sort=-creatordate").splitlines() if t]
    sections: list[str] = []

    if tags:
        unreleased = log_records(f"{tags[0]}..HEAD")
        if unreleased:
            sections.append(f"### Chưa phát hành (`{tags[0]}..HEAD`)\n\n{_render_commits(unreleased)}")
        for i, tag in enumerate(tags):
            prev = tags[i + 1] if i + 1 < len(tags) else None
            rng = f"{prev}..{tag}" if prev else tag
            date = git("log", "-1", "--format=%ad", "--date=short", tag)
            sections.append(f"### {tag} — {date}\n\n{_render_commits(log_records(rng))}")
    else:
        recs = log_records()
        sections.append(
            "### Chưa phát hành — chưa có git tag\n\n"
            "> Chưa có tag phiên bản nào. Sau khi gắn tag (`git tag v1.0.0`), khối này tự "
            "tách theo từng phiên bản.\n\n"
            + _render_commits(recs)
        )

    return "\n\n".join(sections)


def gen_commit_history() -> str:
    recs = log_records(limit=15)
    rows = [[f"`{c['short']}`", c["date"], c["author"], c["subject"]] for c in recs]
    return table(["Commit", "Ngày", "Tác giả", "Nội dung"], rows)


# ─────────────────────────────────────────────────────────────────────────────
# Khối: repo-map
# ─────────────────────────────────────────────────────────────────────────────

MODULE_ROLES: dict[str, str] = {
    "app.py": "Streamlit UI — điểm vào duy nhất cho người dùng cuối",
    "graph": "LangGraph: state, các node agent, và tool thực thi",
    "model": "ModelClient (Ollama local-first, fallback OpenAI) + tham số theo agent",
    "schemas": "Pydantic contract: ExecutionPlan (Supervisor) và InsightOutput (Insight)",
    "perception": "Perception layer — introspect PostgreSQL sinh `info_box` JSON",
    "prompts": "System prompt của từng skill, dạng file text tách khỏi code",
    "data": "DDL 3 domain, seed, và dataset huấn luyện/đánh giá (JSONL)",
    "eval": "Runner đo baseline / PEFT và so sánh hai lần chạy",
    "training": "Sinh dữ liệu, LoRA/QLoRA notebook, checkpoint và kết quả",
    "tests": "pytest — unit theo từng agent, integration theo độ phức tạp câu hỏi",
    "scripts": "Tiện ích vận hành: áp schema, kiểm tra kết nối, sinh tài liệu",
    "docs": "Bộ tài liệu dự án (chính file này)",
    "adapters": "Điểm cắm cho backend tool thay thế",
    "sandbox": "Chỗ dành cho sandbox container hoá của Python Agent",
    ".github": "CI/CD — unit test, build & push image lên GHCR",
}

TEXT_SUFFIXES = {".py", ".md", ".txt", ".sql", ".yml", ".yaml", ".json", ".jsonl", ".sh", ".toml", ".cfg"}


def gen_repo_map() -> str:
    files = tracked_files()
    buckets: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        rel = f.relative_to(ROOT)
        key = rel.parts[0] if len(rel.parts) > 1 else rel.name
        buckets[key].append(f)

    rows = []
    for key in sorted(buckets, key=lambda k: (k.startswith("."), k)):
        group = buckets[key]
        py = sum(1 for f in group if f.suffix == ".py")
        if key == "docs":
            # Không đếm dòng của docs/: script này ghi vào chính docs/, nên số dòng
            # đổi mỗi lần sinh → khối AUTO không bao giờ hội tụ và hook lặp vô hạn.
            loc_label = "—"
        else:
            loc = sum(line_count(f) for f in group if f.suffix in TEXT_SUFFIXES and f.exists())
            loc_label = f"{loc:,}".replace(",", ".")
        rows.append([
            f"`{key}`",
            str(len(group)),
            str(py),
            loc_label,
            MODULE_ROLES.get(key, "—"),
        ])
    return table(["Đường dẫn", "File", "File .py", "Dòng", "Vai trò"], rows)


# ─────────────────────────────────────────────────────────────────────────────
# Khối: tech-stack
# ─────────────────────────────────────────────────────────────────────────────

STACK_RATIONALE: dict[str, tuple[str, str]] = {
    "langgraph": ("Điều phối multi-agent dạng đồ thị có trạng thái",
                  "Cần vòng lặp agent → reflector → agent với state chia sẻ; LangGraph cho conditional edge và state reducer sẵn, còn chain tuyến tính (LCEL) hay CrewAI không diễn tả được vòng self-repair này"),
    "langchain-core": ("Kiểu dữ liệu message/runnable nền", "Đi kèm LangGraph; không dùng abstraction cao hơn để giữ quyền kiểm soát prompt"),
    "langchain-ollama": ("Kết nối LangChain ↔ Ollama", "Cho phép đổi sang LCEL sau này mà không viết lại adapter"),
    "ollama": ("Chạy LLM cục bộ", "Yêu cầu on-prem: dữ liệu khách không rời máy; Ollama quản lý model + quantization và có HTTP API ổn định, không cần GPU cluster như vLLM"),
    "psycopg2-binary": ("Driver PostgreSQL", "Cần `SET LOCAL statement_timeout`, `EXPLAIN`, và `psycopg2.sql.Identifier` để quote định danh an toàn — thứ ORM che mất"),
    "sqlalchemy": ("Tiện ích kết nối/pool", "Dùng ở mức thấp; agent sinh SQL thô nên ORM không phải trung tâm"),
    "python-dotenv": ("Nạp cấu hình từ `.env`", "Chuẩn 12-factor, tránh hardcode credential"),
    "pandas": ("Vật mang dữ liệu giữa các agent", "SQL → DataFrame → Python → Viz dùng chung một kiểu, serialize được qua `df_to_state()`"),
    "numpy": ("Tính toán số", "Phụ thuộc nền của pandas/scipy, cũng nằm trong namespace sandbox"),
    "scipy": ("Thống kê phát hiện bất thường (z-score, IQR)", "Có sẵn hàm kiểm định, không cần tự cài đặt"),
    "matplotlib": ("Sinh biểu đồ PNG", "Backend `Agg` chạy không cần màn hình — hợp với server; xuất base64 nhúng thẳng vào state"),
    "seaborn": ("Style biểu đồ", "Chỉ dùng theme `seaborn-v0_8-whitegrid` cho đồng nhất"),
    "pydantic": ("Contract cho output LLM", "Ranh giới an toàn: plan có chu trình / insight sai định dạng bị chặn tại validate thay vì nổ giữa pipeline"),
    "openai": ("Fallback khi Ollama lỗi", "Giữ hệ thống trả lời được khi model cục bộ chết; tắt được bằng `ENABLE_OPENAI_FALLBACK=0` cho triển khai kín"),
    "streamlit": ("UI chat + bảng + biểu đồ", "Một file Python ra được UI có state; React/FastAPI tốn công gấp nhiều lần cho cùng phạm vi demo"),
    "ragas": ("Đánh giá chất lượng sinh", "Bộ metric có sẵn cho đánh giá đầu ra LLM"),
    "faker": ("Sinh dữ liệu seed tiếng Việt", "Có locale vi_VN — tên/địa chỉ thật hợp cảnh dữ liệu doanh nghiệp Việt Nam"),
}


def gen_tech_stack() -> str:
    req = read(ROOT / "requirements.txt")
    rows = []
    for line in req.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", line, 1)[0].strip()
        constraint = line[len(name):].strip() or "—"
        role, why = STACK_RATIONALE.get(name, ("—", "—"))
        rows.append([f"`{name}`", f"`{constraint}`", role, why])
    return table(["Package", "Ràng buộc", "Vai trò", "Lý do chọn"], rows)


# ─────────────────────────────────────────────────────────────────────────────
# Khối: env-vars
# ─────────────────────────────────────────────────────────────────────────────

GETENV_RE = re.compile(
    r"""os\.getenv\(\s*["'](?P<name>[A-Z0-9_]+)["']\s*(?:,\s*(?P<default>[^)]*?))?\s*\)""",
    re.DOTALL,
)


def gen_env_vars() -> str:
    found: dict[str, dict[str, object]] = {}
    for path in tracked_files("*.py"):
        rel = str(path.relative_to(ROOT))
        for m in GETENV_RE.finditer(read(path)):
            name = m.group("name")
            default = (m.group("default") or "").strip().replace("\n", " ")
            entry = found.setdefault(name, {"default": default, "files": set()})
            if not entry["default"] and default:
                entry["default"] = default
            entry["files"].add(rel)  # type: ignore[union-attr]

    example_keys = set()
    for line in read(ROOT / "env.example").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            example_keys.add(line.split("=", 1)[0].strip())

    rows = []
    for name in sorted(set(found) | example_keys):
        entry = found.get(name, {"default": "", "files": set()})
        files = sorted(entry["files"])  # type: ignore[arg-type]
        where = ", ".join(f"`{f}`" for f in files[:3]) or "_chỉ có trong `env.example`_"
        if len(files) > 3:
            where += f" (+{len(files) - 3})"
        default = entry["default"] or "—"  # type: ignore[assignment]
        rows.append([
            f"`{name}`",
            f"`{default}`" if default != "—" else "—",
            "✅" if name in example_keys else "—",
            where,
        ])
    return table(["Biến", "Mặc định trong code", "Có trong `env.example`", "Nơi đọc"], rows)


# ─────────────────────────────────────────────────────────────────────────────
# Khối: db-tables, db-erd
# ─────────────────────────────────────────────────────────────────────────────

CREATE_TABLE_RE = re.compile(
    r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(?P<name>\w+)\s*\((?P<body>.*?)\n\);",
    re.DOTALL | re.IGNORECASE,
)
COLUMN_RE = re.compile(r"^\s{4}(?P<name>[a-z_][a-z0-9_]*)\s+(?P<type>[A-Z][A-Z0-9_]*(?:\s*\([^)]*\))?)")
ENUM_RE = re.compile(r"IN\s*\((?P<vals>[^)]*)\)", re.IGNORECASE)
CONSTRAINT_START = ("CONSTRAINT", "PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "--")


def _column_chunks(body: str) -> list[str]:
    """Gom thân CREATE TABLE thành từng định nghĩa cột.

    Một cột bắt đầu ở dòng thụt 4 khoảng trắng; các dòng thụt sâu hơn là phần tiếp
    theo của cột đó (CHECK nhiều dòng). Không tách theo dấu phẩy — làm vậy sẽ kéo
    ràng buộc của cột sau vào cột trước.
    """
    chunks: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        if COLUMN_RE.match(line) and not line.strip().upper().startswith(CONSTRAINT_START):
            if current:
                chunks.append("\n".join(current))
            current = [line]
        elif current and line.startswith((" " * 5, "\t")):
            current.append(line)
        elif current:
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))
    return chunks


def _parse_schemas() -> dict[str, list[dict]]:
    """domain → danh sách bảng đã parse từ data/schemas/schema_*.sql."""
    domains: dict[str, list[dict]] = {}
    for path in sorted(tracked_files("data/schemas/schema_*.sql")):
        domain = path.stem.replace("schema_", "")
        sql = read(path)
        tables = []
        for m in CREATE_TABLE_RE.finditer(sql):
            name = m.group("name")
            body = m.group("body")
            columns = []
            for raw in _column_chunks(body):
                cm = COLUMN_RE.match(raw)
                if not cm:
                    continue
                col, ctype = cm.group("name"), " ".join(cm.group("type").split())
                flags = []
                upper = raw.upper()
                if "PRIMARY KEY" in upper:
                    flags.append("PK")
                ref = re.search(r"REFERENCES\s+(\w+)\s*\((\w+)\)", raw, re.IGNORECASE)
                if ref:
                    flags.append(f"FK → `{ref.group(1)}.{ref.group(2)}`")
                if "GENERATED ALWAYS" in upper:
                    flags.append("GENERATED (không ghi trực tiếp)")
                if "NOT NULL" in upper:
                    flags.append("NOT NULL")
                if "UNIQUE" in upper:
                    flags.append("UNIQUE")
                em = ENUM_RE.search(raw)
                if em and "CHECK" in upper:
                    vals = ", ".join(v.strip().strip("'") for v in em.group("vals").split(","))
                    flags.append(f"CHECK ∈ {{{vals}}}")
                elif "CHECK" in upper:
                    chk = re.search(r"CHECK\s*\((?P<c>.+?)\)\s*(?:,|$)", " ".join(raw.split()), re.IGNORECASE)
                    if chk:
                        flags.append(f"CHECK `{chk.group('c').strip()}`")
                columns.append({"name": col, "type": ctype, "flags": flags, "ref": ref.group(1) if ref else None})
            idx = re.findall(rf"CREATE (?:UNIQUE )?INDEX\s+(\w+)\s+ON\s+{name}\s*\(", sql, re.IGNORECASE)
            tables.append({"name": name, "columns": columns, "indexes": idx})
        domains[domain] = tables
    return domains


def gen_db_tables() -> str:
    domains = _parse_schemas()
    if not domains:
        return "_Không tìm thấy `data/schemas/schema_*.sql`._"
    out = []
    for domain, tables in domains.items():
        out.append(f"#### Domain `{domain}` — `data/schemas/schema_{domain}.sql`")
        out.append("")
        for t in tables:
            out.append(f"**`{t['name']}`**" + (f" · {len(t['indexes'])} index" if t["indexes"] else ""))
            out.append("")
            rows = [[f"`{c['name']}`", f"`{c['type']}`", ", ".join(c["flags"]) or "—"] for c in t["columns"]]
            out.append(table(["Cột", "Kiểu", "Ràng buộc"], rows))
            out.append("")
    return "\n".join(out).strip()


def gen_db_rowcounts() -> str:
    """Số dòng thực tế theo lần chạy `perception/extract_info_box.py` gần nhất."""
    path = ROOT / "perception" / "info_box_all.json"
    if not path.exists():
        return "_Chưa có `perception/info_box_all.json` — chạy `python perception/extract_info_box.py`._"
    try:
        data = json.loads(read(path))
    except json.JSONDecodeError:
        return "_`info_box_all.json` không parse được._"

    # info_box_all gộp cả ba domain và mất nhãn domain của từng bảng — dựng lại nhãn
    # từ các file info_box_<domain>.json.
    table_domain: dict[str, str] = {}
    for per_domain in sorted(ROOT.glob("perception/info_box_*.json")):
        if per_domain.name == "info_box_all.json":
            continue
        try:
            payload = json.loads(read(per_domain))
        except json.JSONDecodeError:
            continue
        for t in payload.get("tables", []):
            table_domain[t.get("table_name", "")] = payload.get("domain", per_domain.stem)

    domains = data if isinstance(data, dict) and "tables" not in data else {"all": data}
    rows = []
    total = 0
    for domain, payload in domains.items():
        for t in payload.get("tables", []):
            n = t.get("row_count", 0)
            total += n if isinstance(n, int) else 0
            name = t.get("table_name", "")
            rows.append([
                f"`{name}`",
                table_domain.get(name, payload.get("domain", domain)),
                f"{n:,}".replace(",", "."),
                str(len(t.get("columns", []))),
                str(len(t.get("foreign_keys", []))),
                str(len(t.get("indexes", []))),
            ])
    rows.append(["**Tổng**", "—", f"**{total:,}**".replace(",", "."), "—", "—", "—"])
    return table(["Bảng", "Domain", "Số dòng", "Cột", "FK", "Index"], rows)


DATASET_ROLES: dict[str, str] = {
    "raw_dataset.jsonl": "Toàn bộ mẫu sinh ra trước khi lọc",
    "validated_dataset.jsonl": "Mẫu qua được `training/validate_dataset.py`",
    "rejected_dataset.jsonl": "Mẫu bị loại kèm lý do",
    "long_context_excluded.jsonl": "Mẫu vượt 4096 token — loại để không cắt cụt lúc train",
    "train.jsonl": "Tập huấn luyện LoRA",
    "valid.jsonl": "Tập validation (theo dõi val loss)",
    "test.jsonl": "Tập kiểm thử — dùng bởi `eval/eval_runner.py`",
    "supervisor_routing_samples.jsonl": "Mẫu routing bổ sung cho Supervisor v2",
}

SKILL_MARKERS = [
    ("PostgreSQL specialist", "text-to-sql"),
    ("Supervisor Agent", "supervisor-routing"),
    ("data analysis specialist", "data-analysis"),
    ("visualization specialist", "visualization"),
    ("business intelligence analyst", "insight-generation"),
    ("error diagnosis", "error-reflection"),
]


def _skill_of(system_prompt: str) -> str:
    for marker, skill in SKILL_MARKERS:
        if marker in system_prompt:
            return skill
    return "unknown"


def gen_datasets() -> str:
    rows = []
    for path in sorted(tracked_files("data/*.jsonl")):
        n = 0
        skills: dict[str, int] = defaultdict(int)
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    n += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msgs = obj.get("messages") or []
                    system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
                    skills[_skill_of(system)] += 1
        except OSError:
            continue
        top = ", ".join(
            f"{k} ({v})" for k, v in sorted(skills.items(), key=lambda kv: -kv[1]) if k != "unknown"
        ) or "—"
        size = path.stat().st_size
        size_label = f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.0f} KB"
        rows.append([
            f"`data/{path.name}`",
            f"{n:,}".replace(",", "."),
            size_label,
            DATASET_ROLES.get(path.name, "—"),
            top,
        ])
    return table(["File", "Số mẫu", "Kích thước", "Vai trò", "Phân bố skill"], rows)


ANOMALY_BLOCK_RE = re.compile(
    r'"(?P<id>[SIH]\d)":\s*\{(?P<body>.*?)\n    \},',
    re.DOTALL,
)


def gen_anomalies() -> str:
    """Danh mục bất thường được cấy vào dữ liệu seed (data/seed/seed_data.py)."""
    src = read(ROOT / "data" / "seed" / "seed_data.py")
    start = src.find("ANOMALY_CATALOGUE = {")
    if start == -1:
        return "_Không tìm thấy `ANOMALY_CATALOGUE`._"
    rows = []
    for m in ANOMALY_BLOCK_RE.finditer(src[start:]):
        body = m.group("body")

        def field(key: str) -> str:
            fm = re.search(rf'"{key}":\s*(?:\(\s*)?((?:"[^"]*"\s*)+)', body)
            if not fm:
                return "—"
            return " ".join(re.findall(r'"([^"]*)"', fm.group(1))).strip()

        rows.append([f"`{m.group('id')}`", field("domain"), field("name"), field("type"), field("description")])
    return table(["ID", "Domain", "Tên", "Loại", "Mô tả"], rows)


def gen_db_erd() -> str:
    domains = _parse_schemas()
    lines = ["```mermaid", "erDiagram"]
    rels = []
    for tables in domains.values():
        for t in tables:
            for c in t["columns"]:
                if c["ref"]:
                    rels.append(f'  {c["ref"]} ||--o{{ {t["name"]} : "{c["name"]}"')
    if not rels:
        return "_Chưa parse được quan hệ khoá ngoại._"
    lines.extend(sorted(set(rels)))
    lines.append("```")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Khối: agents, prompts, state-fields
# ─────────────────────────────────────────────────────────────────────────────

def _model_config_dicts() -> dict[str, dict]:
    """Đọc AGENT_* dict trong model/model_config.py bằng ast (không import)."""
    out: dict[str, dict] = {}
    try:
        tree = ast.parse(read(ROOT / "model" / "model_config.py"))
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("AGENT_"):
                try:
                    out[name] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass
    return out


def gen_agents() -> str:
    cfg = _model_config_dicts()
    temps = cfg.get("AGENT_TEMPERATURES", {})
    tokens = cfg.get("AGENT_MAX_TOKENS", {})
    timeouts = cfg.get("AGENT_TIMEOUT_S", {})

    rows = []
    for path in sorted(tracked_files("graph/agents/*.py")):
        if path.name == "__init__.py":
            continue
        src = read(path)
        key = path.stem.replace("_agent", "")
        node = next((m.group(1) for m in re.finditer(r"^def (\w+_node)\(", src, re.M)), "—")
        prompt = re.search(r'"prompts"\s*/\s*"([\w.]+)"', src)
        retries = re.search(r"^MAX_RETRIES\s*=\s*(\d+)", src, re.M)
        rows.append([
            f"`{key}`",
            f"`{node}()`",
            f"`prompts/{prompt.group(1)}`" if prompt else "inline (trong file agent)",
            retries.group(1) if retries else "—",
            str(temps.get(key, "—")),
            str(tokens.get(key, "—")),
            str(timeouts.get(key, "—")),
        ])
    return table(
        ["Agent", "Node LangGraph", "Prompt", "Retry nội bộ", "Temperature", "Max tokens", "Timeout (s)"],
        rows,
    )


def gen_prompts() -> str:
    rows = []
    for path in sorted(tracked_files("prompts/*.txt")):
        text = read(path)
        placeholders = sorted(set(re.findall(r"\{(\w+)\}", text)))
        rows.append([
            f"`prompts/{path.name}`",
            str(text.count("\n") + 1),
            f"{len(text.encode('utf-8')) / 1024:.1f} KB",
            ", ".join(f"`{{{p}}}`" for p in placeholders) or "—",
        ])
    return table(["File", "Dòng", "Kích thước", "Placeholder được thay lúc chạy"], rows)


def gen_state_fields() -> str:
    src = read(ROOT / "graph" / "state.py")
    group = "—"
    rows = []
    inside = False
    for line in src.splitlines():
        if line.startswith("class MultiAgentState"):
            inside = True
            continue
        if inside and line and not line.startswith((" ", "\t")):
            break
        if not inside:
            continue
        gm = re.match(r"\s*#\s*─+\s*(.+?)\s*─+", line)
        if gm:
            group = gm.group(1)
            continue
        fm = re.match(r"\s{4}(?P<name>\w+):\s*(?P<type>.+?)\s*$", line)
        if fm:
            ftype = fm.group("type")
            required = "tuỳ chọn" if ftype.startswith("NotRequired") else "bắt buộc"
            rows.append([f"`{fm.group('name')}`", f"`{ftype}`", required, group])
    return table(["Trường", "Kiểu", "Bắt buộc", "Nhóm"], rows)


# ─────────────────────────────────────────────────────────────────────────────
# Khối: metrics, tests
# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = [
    ("sql_execution_accuracy", "SQL Execution Accuracy", "%"),
    ("sql_heuristic_accuracy", "SQL Heuristic Accuracy", "%"),
    ("python_syntax_rate", "Python Syntax Rate", "%"),
    ("supervisor_json_rate", "Supervisor JSON Valid", "%"),
    ("insight_json_rate", "Insight JSON Valid", "%"),
    ("reflector_json_rate", "Reflector JSON Valid", "%"),
    ("overall_json_valid_rate", "Overall JSON Valid", "%"),
    ("avg_latency_s", "Latency trung bình / mẫu", "s"),
]


def _load_summary(rel: str) -> dict:
    path = ROOT / rel
    if not path.exists():
        return {}
    try:
        data = json.loads(read(path))
    except json.JSONDecodeError:
        return {}
    return data.get("summary", data)


def gen_metrics() -> str:
    base = _load_summary("eval/baseline_results.json")
    tuned = _load_summary("training/finetuned_checkpoint50_results.json")
    if not base and not tuned:
        return "_Chưa có file kết quả eval._"
    targets = (tuned or base).get("targets", {})

    rows = []
    for key, label, unit in METRIC_LABELS:
        b, t = base.get(key), tuned.get(key)
        delta = "—"
        if isinstance(b, (int, float)) and isinstance(t, (int, float)):
            diff = t - b
            arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "=")
            delta = f"{arrow} {abs(diff):.1f}{unit}"
        rows.append([
            label,
            f"{b}{unit}" if b is not None else "—",
            f"{t}{unit}" if t is not None else "—",
            delta,
            targets.get(key, "—"),
        ])

    meta = []
    for name, d in (("Baseline", base), ("Fine-tuned", tuned)):
        if d:
            meta.append(
                f"- **{name}** — `{d.get('model', '?')}`, {d.get('total_samples', '?')} mẫu, "
                f"chạy {d.get('timestamp', '?')}"
            )
    return (
        table(["Metric", "Baseline", "Fine-tuned (LoRA ckpt-50)", "Chênh lệch", "Mục tiêu"], rows)
        + "\n\n"
        + "\n".join(meta)
    )


def gen_tests() -> str:
    rows = []
    total = 0
    for path in sorted(tracked_files("tests/**/*.py")):
        if path.name == "__init__.py":
            continue
        n = len(re.findall(r"^\s*def (test_\w+)", read(path), re.M))
        total += n
        rows.append([f"`{path.relative_to(ROOT)}`", str(n)])
    rows.append(["**Tổng**", f"**{total}**"])
    return table(["File", "Số test"], rows)


# ─────────────────────────────────────────────────────────────────────────────
# Registry & áp dụng
# ─────────────────────────────────────────────────────────────────────────────

GENERATORS = {
    "stamp": gen_stamp,
    "changelog": gen_changelog,
    "commit-history": gen_commit_history,
    "repo-map": gen_repo_map,
    "tech-stack": gen_tech_stack,
    "env-vars": gen_env_vars,
    "db-tables": gen_db_tables,
    "db-erd": gen_db_erd,
    "db-rowcounts": gen_db_rowcounts,
    "anomalies": gen_anomalies,
    "datasets": gen_datasets,
    "agents": gen_agents,
    "prompts": gen_prompts,
    "state-fields": gen_state_fields,
    "metrics": gen_metrics,
    "tests": gen_tests,
}


# Fence có thể thụt lề khi nằm trong danh sách đánh số — vẫn phải coi là code block.
FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>```+|~~~+).*?^[ \t]*(?P=fence)[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def _fenced_spans(doc: str) -> list[tuple[int, int]]:
    """Khoảng ký tự nằm trong code fence — marker ở đây là ví dụ, không phải khối thật."""
    return [(m.start(), m.end()) for m in FENCE_RE.finditer(doc)]


def render(doc: str) -> tuple[str, set[str], set[str]]:
    """Thay nội dung mọi khối AUTO trong `doc`. Trả về (văn bản mới, id đã dùng, id lạ)."""
    used: set[str] = set()
    unknown: set[str] = set()
    cache: dict[str, str] = {}
    fences = _fenced_spans(doc)

    def repl(m: re.Match[str]) -> str:
        if any(start <= m.start() < end for start, end in fences):
            return m.group(0)  # ví dụ minh hoạ trong code fence — bỏ qua
        block_id = m.group("id")
        if block_id not in GENERATORS:
            unknown.add(block_id)
            return m.group(0)
        used.add(block_id)
        if block_id not in cache:
            cache[block_id] = GENERATORS[block_id]().strip()
        return f"{m.group('begin')}\n\n{cache[block_id]}\n\n{m.group('end')}"

    return MARKER_RE.sub(repl, doc), used, unknown


def main() -> int:
    ap = argparse.ArgumentParser(description="Cập nhật các khối AUTO trong docs/")
    ap.add_argument("--check", action="store_true", help="Không ghi file; exit 1 nếu tài liệu lỗi thời")
    ap.add_argument("--list", action="store_true", help="Liệt kê id khối AUTO và nơi xuất hiện")
    args = ap.parse_args()

    md_files = sorted(DOCS_DIR.rglob("*.md")) if DOCS_DIR.exists() else []
    if not md_files:
        print("Không tìm thấy file .md nào trong docs/", file=sys.stderr)
        return 1

    if args.list:
        for path in md_files:
            doc = read(path)
            fences = _fenced_spans(doc)
            ids = sorted({
                m.group("id")
                for m in MARKER_RE.finditer(doc)
                if not any(start <= m.start() < end for start, end in fences)
            })
            if ids:
                print(f"{path.relative_to(ROOT)}: {', '.join(ids)}")
        print(f"\nGenerator khả dụng: {', '.join(sorted(GENERATORS))}")
        return 0

    stale, written, unknown_all = [], [], set()
    for path in md_files:
        original = read(path)
        if "AUTO:begin" not in original:
            continue
        updated, _used, unknown = render(original)
        unknown_all |= unknown
        if updated != original:
            stale.append(path)
            if not args.check:
                path.write_text(updated, encoding="utf-8")
                written.append(path)

    for block_id in sorted(unknown_all):
        print(f"⚠️  Khối AUTO không có generator: id={block_id}", file=sys.stderr)

    if args.check:
        if stale:
            print("Tài liệu đã lỗi thời:", file=sys.stderr)
            for p in stale:
                print(f"  - {p.relative_to(ROOT)}", file=sys.stderr)
            print("\nChạy: python scripts/update_docs.py", file=sys.stderr)
            return 1
        print("✅ Tài liệu đồng bộ với mã nguồn.")
        return 0

    if written:
        print(f"✅ Đã cập nhật {len(written)} file:")
        for p in written:
            print(f"  - {p.relative_to(ROOT)}")
    else:
        print("✅ Không có gì thay đổi — tài liệu đã đồng bộ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
