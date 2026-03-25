"""
ADBA — Validate generated training dataset.

Input:  data/raw_dataset.jsonl
Output: data/validated_dataset.jsonl
Reject: data/rejected_dataset.jsonl

Validation coverage by skill_type:
- text-to-sql         : JSON shape + SQL safety checks + EXPLAIN on Postgres
- data-analysis       : JSON shape + Python syntax check (ast.parse)
- supervisor-routing  : JSON shape + ExecutionPlan Pydantic validation
- insight-generation  : JSON shape + InsightOutput Pydantic validation
- error-reflection    : JSON shape + required fields + category whitelist
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from schemas.insight_schema import InsightOutput
from schemas.plan_schema import ExecutionPlan

DEFAULT_INPUT = ROOT / "data" / "raw_dataset.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "validated_dataset.jsonl"
DEFAULT_REJECT = ROOT / "data" / "rejected_dataset.jsonl"

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://adba_user:adba@localhost:5432/adba_db"
)

ALLOWED_ROLES = ["system", "user", "assistant"]
ALLOWED_REFLECT_CATEGORIES = {
    "sql_syntax",
    "sql_logic",
    "python_runtime",
    "python_logic",
    "data_quality",
    "schema_mismatch",
}
SKILL_TYPES = {
    "text-to-sql",
    "data-analysis",
    "supervisor-routing",
    "insight-generation",
    "error-reflection",
}
FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|create)\b",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""


def strip_code_fence(text: str) -> str:
    return re.sub(r"```(?:json|python|sql)?\s*|\s*```", "", text).strip()


def parse_json_text(text: str) -> Any:
    clean = strip_code_fence(text)
    return json.loads(clean)


def validate_message_shape(obj: dict[str, Any]) -> ValidationResult:
    if "skill_type" not in obj:
        return ValidationResult(False, "missing skill_type")
    if obj["skill_type"] not in SKILL_TYPES:
        return ValidationResult(False, f"unknown skill_type: {obj['skill_type']}")

    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        return ValidationResult(False, "messages must be a list with exactly 3 items")

    for idx, expected_role in enumerate(ALLOWED_ROLES):
        msg = msgs[idx]
        if not isinstance(msg, dict):
            return ValidationResult(False, f"messages[{idx}] must be object")
        if msg.get("role") != expected_role:
            return ValidationResult(
                False, f"messages[{idx}].role must be '{expected_role}'"
            )
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            return ValidationResult(False, f"messages[{idx}].content is empty")

    return ValidationResult(True)


def validate_sql_content(sql: str, conn: psycopg2.extensions.connection, timeout_ms: int) -> ValidationResult:
    sql_clean = strip_code_fence(sql).rstrip(";").strip()
    if not sql_clean:
        return ValidationResult(False, "empty sql content")
    if not re.match(r"^(with|select)\b", sql_clean, re.IGNORECASE):
        return ValidationResult(False, "sql must start with SELECT or WITH")
    if FORBIDDEN_SQL.search(sql_clean):
        return ValidationResult(False, "sql contains forbidden mutation statement")

    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
            cur.execute("EXPLAIN " + sql_clean)
            cur.execute("ROLLBACK")
        return ValidationResult(True)
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return ValidationResult(False, f"sql execution failed: {e}")


def validate_python_content(code: str) -> ValidationResult:
    code_clean = strip_code_fence(code)
    if not code_clean:
        return ValidationResult(False, "empty python content")
    try:
        tree = ast.parse(code_clean)
    except SyntaxError as e:
        return ValidationResult(False, f"python syntax error: {e}")

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
    for alias in ("pd", "np", "stats"):
        if alias in used_names and alias not in imported_aliases:
            return ValidationResult(
                False, f"python may fail at runtime: uses '{alias}' without import alias"
            )
    return ValidationResult(True)


def validate_supervisor_content(content: str) -> ValidationResult:
    try:
        payload = parse_json_text(content)
        plan = ExecutionPlan.model_validate(payload)
    except Exception as e:
        return ValidationResult(False, f"invalid ExecutionPlan: {e}")

    # Guardrail: SQL step task must be instruction text, not SQL literal.
    sql_keywords = (
        "SELECT ", " WITH ", " FROM ", " JOIN ", " WHERE ",
        " GROUP BY ", " ORDER BY ", " HAVING ", " LIMIT ",
    )
    for step in plan.steps:
        if step.agent != "sql":
            continue
        task_upper = f" {step.task.strip().upper()} "
        hits = sum(kw in task_upper for kw in sql_keywords)
        if task_upper.startswith(("SELECT ", "WITH ")) or hits >= 2:
            return ValidationResult(
                False,
                "supervisor sql task contains SQL literal; must be NL instruction",
            )

    return ValidationResult(True)


def validate_insight_content(content: str) -> ValidationResult:
    try:
        payload = parse_json_text(content)
        insight = InsightOutput.model_validate(payload)
    except Exception as e:
        return ValidationResult(False, f"invalid InsightOutput: {e}")

    # Quality gate (aligned with generator):
    # At least one evidence item must contain a side-by-side comparison
    # with >=2 non-percentage numbers (e.g., "142,000,000 vs 75,400,000"
    # or "62 vs 18").
    num_re = re.compile(r"\d+(?:[.,]\d+)?")
    has_comparison_pair = False
    for item in insight.evidence:
        lower = item.lower()
        if not any(tok in lower for tok in (" vs ", " so với ", " compared to ", " so sanh ", " so sánh ")):
            continue
        count_non_pct = 0
        for m in num_re.finditer(item):
            next_char = item[m.end(): m.end() + 1]
            if next_char == "%":
                continue
            count_non_pct += 1
        if count_non_pct >= 2:
            has_comparison_pair = True
            break

    if not has_comparison_pair:
        return ValidationResult(
            False, "insight evidence lacks absolute comparison pair (A vs B)"
        )

    return ValidationResult(True)


def validate_reflector_content(content: str) -> ValidationResult:
    try:
        payload = parse_json_text(content)
    except Exception as e:
        return ValidationResult(False, f"invalid reflector json: {e}")

    required = ["root_cause", "error_category", "fix_strategy", "corrected_context"]
    for key in required:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return ValidationResult(False, f"missing/empty reflector field: {key}")

    if payload["error_category"] not in ALLOWED_REFLECT_CATEGORIES:
        return ValidationResult(
            False, f"invalid error_category: {payload['error_category']}"
        )

    return ValidationResult(True)


def validate_record(
    record: dict[str, Any],
    conn: psycopg2.extensions.connection | None,
    timeout_ms: int,
) -> ValidationResult:
    shape = validate_message_shape(record)
    if not shape.ok:
        return shape

    skill = record["skill_type"]
    assistant = record["messages"][2]["content"]

    if skill == "text-to-sql":
        if conn is None:
            return ValidationResult(False, "sql validation requires database connection")
        return validate_sql_content(assistant, conn, timeout_ms)
    if skill == "data-analysis":
        return validate_python_content(assistant)
    if skill == "supervisor-routing":
        return validate_supervisor_content(assistant)
    if skill == "insight-generation":
        return validate_insight_content(assistant)
    if skill == "error-reflection":
        return validate_reflector_content(assistant)

    return ValidationResult(False, f"unsupported skill_type: {skill}")


def run_validation(
    input_path: Path,
    output_path: Path,
    reject_path: Path,
    timeout_ms: int,
    skip_sql_exec: bool,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)

    conn = None
    if not skip_sql_exec:
        conn = psycopg2.connect(DATABASE_URL)

    total = 0
    valid = 0
    invalid = 0
    by_skill_total: dict[str, int] = {}
    by_skill_valid: dict[str, int] = {}

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
        reject_path.open("w", encoding="utf-8") as frej,
    ):
        for lineno, line in enumerate(fin, start=1):
            raw = line.strip()
            if not raw:
                continue

            total += 1
            try:
                obj = json.loads(raw)
            except Exception as e:
                invalid += 1
                frej.write(
                    json.dumps(
                        {
                            "line": lineno,
                            "reason": f"invalid json line: {e}",
                            "raw": raw[:5000],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            skill = obj.get("skill_type", "unknown")
            by_skill_total[skill] = by_skill_total.get(skill, 0) + 1

            result = validate_record(obj, conn, timeout_ms)
            if result.ok:
                valid += 1
                by_skill_valid[skill] = by_skill_valid.get(skill, 0) + 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                invalid += 1
                frej.write(
                    json.dumps(
                        {
                            "line": lineno,
                            "skill_type": skill,
                            "reason": result.reason,
                            "record": obj,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if conn is not None:
        conn.close()

    print("=" * 56)
    print("Validation finished")
    print(f"Input file:     {input_path}")
    print(f"Validated file: {output_path}")
    print(f"Rejected file:  {reject_path}")
    print("-" * 56)
    print(f"Total records:  {total}")
    print(f"Valid records:  {valid}")
    print(f"Invalid records:{invalid}")
    print("\nBy skill:")
    for skill in sorted(by_skill_total):
        t = by_skill_total.get(skill, 0)
        v = by_skill_valid.get(skill, 0)
        print(f"  {skill:20s} {v:4d}/{t:<4d} valid")
    print("=" * 56)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ADBA generated dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Validated JSONL output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--reject",
        type=Path,
        default=DEFAULT_REJECT,
        help=f"Rejected JSONL output (default: {DEFAULT_REJECT})",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=3000,
        help="statement_timeout for SQL EXPLAIN validation",
    )
    parser.add_argument(
        "--skip-sql-exec",
        action="store_true",
        help="Skip SQL execution checks (syntax/shape only for SQL samples)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation(
        input_path=args.input,
        output_path=args.output,
        reject_path=args.reject,
        timeout_ms=args.timeout_ms,
        skip_sql_exec=args.skip_sql_exec,
    )
