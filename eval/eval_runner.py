"""
eval/eval_runner.py
===================
Baseline Evaluation Runner — ADBA Project
------------------------------------------
Runs the 120 test samples in data/test.jsonl through the base Ollama model
(before any fine-tuning), measures key quality metrics, and saves a detailed
report to eval/baseline_results.json.

Metrics collected
-----------------
- sql_execution_accuracy  : % of SQL samples where PostgreSQL EXPLAIN succeeds
- sql_heuristic_accuracy  : % of SQL samples passing structural heuristic checks
- python_syntax_rate      : % of Python samples passing AST parse + df reference + import contract
- supervisor_json_rate    : % of supervisor samples passing Pydantic ExecutionPlan validation
- insight_json_rate       : % of insight samples passing Pydantic InsightOutput validation
- reflector_json_rate     : % of reflector samples containing all 4 required keys
- overall_json_valid_rate : average across all JSON-output tasks
- avg_latency_s           : average inference time per sample (seconds)

Usage
-----
  # Full eval — calls Ollama and PostgreSQL
  python eval/eval_runner.py

  # Dry run — skip Ollama calls, evaluate gold-standard answers instead
  python eval/eval_runner.py --dry-run

  # Limit to N samples for quick sanity check
  python eval/eval_runner.py --limit 10

  # Save output to custom path
  python eval/eval_runner.py --output eval/my_results.json

Dependencies
------------
  pip install ollama psycopg2-binary pydantic python-dotenv
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import psycopg2
from pydantic import ValidationError

from schemas.insight_schema import InsightOutput
from schemas.plan_schema import ExecutionPlan

# ── Configuration ────────────────────────────────────────────────────────────

TEST_FILE     = ROOT / "data" / "test.jsonl"
OUTPUT_FILE   = ROOT / "eval"  / "baseline_results.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EVAL_MODEL      = os.getenv("EVAL_MODEL", "qwen2.5-coder:7b-instruct-q5_K_M")
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

def _effective_postgres_url() -> str:
    """Use POSTGRES_URL if set; else build from POSTGRES_* (Docker defaults).

    Matches docker-compose.yml: adba_db / adba_user / POSTGRES_PASSWORD.
    """
    direct = os.getenv("POSTGRES_URL", "").strip()
    if direct:
        return direct
    user = os.getenv("POSTGRES_USER", "adba_user")
    password = os.getenv("POSTGRES_PASSWORD", "adba_password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "adba_db")
    user_q = quote_plus(user)
    pw_q = quote_plus(password)
    return f"postgresql://{user_q}:{pw_q}@{host}:{port}/{db}"


POSTGRES_URL = _effective_postgres_url()

# Per-skill inference temperature (same as production agents)
SKILL_TEMPERATURES: dict[str, float] = {
    "text-to-sql":        0.0,
    "data-analysis":      0.1,
    "supervisor-routing": 0.1,
    "insight-generation": 0.2,
    "error-reflection":   0.1,
}

SKILL_MAX_TOKENS: dict[str, int] = {
    "text-to-sql":        1800,
    "data-analysis":      1200,
    "supervisor-routing": 700,
    "insight-generation": 800,
    "error-reflection":   700,
}

REFLECTOR_REQUIRED_KEYS = {"root_cause", "error_category", "fix_strategy", "corrected_context"}

# ── Skill Detection ───────────────────────────────────────────────────────────

def detect_skill(system_prompt: str) -> str:
    """Infer skill type from system prompt markers (test.jsonl has no skill_type field)."""
    if "PostgreSQL specialist" in system_prompt:
        return "text-to-sql"
    if (
        "data analysis specialist" in system_prompt
        or "pandas" in system_prompt
        and "DataFrame" in system_prompt
        and "Supervisor" not in system_prompt
    ):
        return "data-analysis"
    if "Supervisor Agent" in system_prompt or "ExecutionPlan" in system_prompt:
        return "supervisor-routing"
    if (
        "business intelligence analyst" in system_prompt
        or '"finding"' in system_prompt
        or '"action"' in system_prompt
    ):
        return "insight-generation"
    if (
        "error diagnosis" in system_prompt
        or "root_cause" in system_prompt
        or "error_category" in system_prompt
    ):
        return "error-reflection"
    return "unknown"


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_fences(text: str) -> str:
    """Remove markdown code fences (```json / ```sql / ```python / ```)."""
    return re.sub(r"```(?:json|python|sql)?\s*|\s*```", "", text).strip()


def safe_json(text: str) -> dict | list | None:
    clean = strip_fences(text)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract the first JSON object/array
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


# ── SQL Checks ────────────────────────────────────────────────────────────────

_SQL_MUTATION_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

def sql_heuristic(sql: str) -> tuple[bool, str]:
    """Lightweight structural checks before hitting the DB."""
    stripped = sql.strip()
    if not stripped:
        return False, "empty output"
    first_token = stripped.split()[0].upper() if stripped.split() else ""
    if first_token not in ("SELECT", "WITH", "EXPLAIN"):
        return False, f"non-SELECT first token: {first_token!r}"
    if _SQL_MUTATION_RE.match(stripped):
        return False, "mutation statement detected"
    return True, "ok"


def sql_explain(sql: str, conn) -> tuple[bool, str]:
    """Run EXPLAIN on the SQL; returns (passed, message)."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN {sql}")
            conn.commit()
        return True, "ok"
    except Exception as e:
        conn.rollback()
        return False, str(e)[:200]


# ── Python Checks ─────────────────────────────────────────────────────────────

_ALIAS_RE = {
    "pd":    re.compile(r"\bimport\s+pandas\s+as\s+pd\b"),
    "np":    re.compile(r"\bimport\s+numpy\s+as\s+np\b"),
    "stats": re.compile(r"\bfrom\s+scipy\s+import\s+stats\b"),
}

def python_check(code: str) -> tuple[bool, str]:
    """AST parse + df reference + import contract."""
    clean = strip_fences(code)
    if not clean:
        return False, "empty output"
    try:
        tree = ast.parse(clean)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    source = clean
    # df must be referenced
    if "df" not in source:
        return False, "no df reference"

    # Import contract: if alias used, import must exist
    for alias, import_re in _ALIAS_RE.items():
        uses_alias = bool(re.search(rf"\b{alias}\.", source))
        has_import = bool(import_re.search(source))
        if uses_alias and not has_import:
            return False, f"uses '{alias}' without import"

    return True, "ok"


# ── JSON-Schema Checks ────────────────────────────────────────────────────────

def supervisor_check(text: str) -> tuple[bool, str]:
    data = safe_json(text)
    if data is None:
        return False, "invalid JSON"
    try:
        ExecutionPlan(**data)
        return True, "ok"
    except (ValidationError, TypeError) as e:
        return False, str(e)[:200]


def insight_check(text: str) -> tuple[bool, str]:
    data = safe_json(text)
    if data is None:
        return False, "invalid JSON"
    try:
        InsightOutput(**data)
        return True, "ok"
    except (ValidationError, TypeError) as e:
        return False, str(e)[:200]


def reflector_check(text: str) -> tuple[bool, str]:
    data = safe_json(text)
    if data is None:
        return False, "invalid JSON"
    if not isinstance(data, dict):
        return False, "expected JSON object"
    missing = REFLECTOR_REQUIRED_KEYS - data.keys()
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    return True, "ok"


# ── Ollama Inference ──────────────────────────────────────────────────────────

def call_ollama(messages: list[dict], skill: str) -> tuple[str, float]:
    """
    Call Ollama with the given messages.
    Returns (generated_text, elapsed_seconds).
    """
    import ollama

    t0 = time.time()
    response = ollama.chat(
        model=EVAL_MODEL,
        messages=messages,
        options={
            "temperature":   SKILL_TEMPERATURES.get(skill, 0.1),
            "num_predict":   SKILL_MAX_TOKENS.get(skill, 1024),
            "num_ctx":       OLLAMA_NUM_CTX,
            "stop":          ["<|im_end|>", "<|endoftext|>"],
        },
    )
    elapsed = time.time() - t0
    return response["message"]["content"], elapsed


# ── Per-Sample Evaluation ─────────────────────────────────────────────────────

def evaluate_sample(
    sample: dict,
    pg_conn,
    dry_run: bool = False,
    idx: int = 0,
) -> dict[str, Any]:
    """
    Evaluate a single test sample.
    Returns a result dict with pass/fail, skill, latency, and details.
    """
    messages  = sample["messages"]
    system    = messages[0]["content"] if messages[0]["role"] == "system" else ""
    user_msg  = next((m for m in messages if m["role"] == "user"), None)
    reference = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    skill = detect_skill(system)

    if dry_run:
        # Evaluate the gold-standard reference answer (sanity-checks data quality)
        generated = reference
        latency   = 0.0
    else:
        input_msgs = [m for m in messages if m["role"] != "assistant"]
        try:
            generated, latency = call_ollama(input_msgs, skill)
        except Exception as e:
            return {
                "idx":       idx,
                "skill":     skill,
                "passed":    False,
                "error":     f"ollama_error: {e}",
                "latency_s": 0.0,
                "generated": "",
            }

    # ── Skill-specific grading ─────────────────────────────────────────────
    result: dict[str, Any] = {
        "idx":       idx,
        "skill":     skill,
        "latency_s": round(latency, 3),
        "generated": generated[:500],   # truncated for readability in JSON
    }

    if skill == "text-to-sql":
        sql = strip_fences(generated)
        heuristic_ok, h_msg = sql_heuristic(sql)
        if heuristic_ok and pg_conn is not None:
            exec_ok, exec_msg = sql_explain(sql, pg_conn)
        else:
            exec_ok  = False
            exec_msg = "heuristic_failed" if not heuristic_ok else "no_db_conn"
        result.update({
            "passed":          exec_ok and heuristic_ok,
            "heuristic_pass":  heuristic_ok,
            "exec_pass":       exec_ok,
            "heuristic_msg":   h_msg,
            "exec_msg":        exec_msg,
        })

    elif skill == "data-analysis":
        ok, msg = python_check(generated)
        result.update({"passed": ok, "detail": msg})

    elif skill == "supervisor-routing":
        ok, msg = supervisor_check(generated)
        result.update({"passed": ok, "detail": msg})

    elif skill == "insight-generation":
        ok, msg = insight_check(generated)
        result.update({"passed": ok, "detail": msg})

    elif skill == "error-reflection":
        ok, msg = reflector_check(generated)
        result.update({"passed": ok, "detail": msg})

    else:
        result.update({"passed": False, "detail": f"unknown skill: {skill}"})

    return result


# ── Metrics Aggregation ───────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict[str, Any]:
    """Compute per-skill and overall summary metrics."""
    from collections import defaultdict

    by_skill: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_skill[r["skill"]].append(r)

    def pct(items: list[dict], field: str = "passed") -> float:
        if not items:
            return 0.0
        return round(sum(1 for x in items if x.get(field)) / len(items) * 100, 1)

    sql   = by_skill.get("text-to-sql",        [])
    py    = by_skill.get("data-analysis",       [])
    sup   = by_skill.get("supervisor-routing",  [])
    ins   = by_skill.get("insight-generation",  [])
    ref   = by_skill.get("error-reflection",    [])

    # Overall JSON valid rate — avg across all JSON-output skills
    json_skills = sup + ins + ref
    overall_json = pct(json_skills) if json_skills else 0.0

    latencies = [r["latency_s"] for r in results if r["latency_s"] > 0]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    return {
        "total_samples":              len(results),
        "sql_execution_accuracy":     pct(sql, "exec_pass"),
        "sql_heuristic_accuracy":     pct(sql, "heuristic_pass"),
        "python_syntax_rate":         pct(py),
        "supervisor_json_rate":       pct(sup),
        "insight_json_rate":          pct(ins),
        "reflector_json_rate":        pct(ref),
        "overall_json_valid_rate":    overall_json,
        "avg_latency_s":              avg_latency,
        "per_skill_counts": {
            skill: len(items) for skill, items in by_skill.items()
        },
        "targets": {
            "sql_execution_accuracy":  "≥82%",
            "python_syntax_rate":      "≥90%",
            "supervisor_json_rate":    "≥88%",
            "insight_json_rate":       "≥88%",
            "overall_json_valid_rate": "≥88%",
        },
    }


# ── Progress Printer ─────────────────────────────────────────────────────────

def _status(idx: int, total: int, r: dict) -> str:
    icon  = "✓" if r.get("passed") else "✗"
    skill = r["skill"][:12].ljust(12)
    extra = r.get("detail") or r.get("exec_msg") or ""
    extra = extra[:60] if extra != "ok" else ""
    lat   = f"{r['latency_s']:.1f}s" if r["latency_s"] else ""
    return f"  [{idx:>3}/{total}] {icon} {skill}  {lat}  {extra}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ADBA Baseline Evaluation Runner")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Evaluate gold answers instead of calling Ollama")
    parser.add_argument("--limit",    type=int, default=0,
                        help="Evaluate only first N samples (0 = all)")
    parser.add_argument("--output",   default=str(OUTPUT_FILE),
                        help="Path to save results JSON (use --output to avoid overwriting baseline)")
    parser.add_argument("--test-file", default=str(TEST_FILE),
                        help="Path to test.jsonl")
    args = parser.parse_args()

    test_path  = Path(args.test_file)
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load samples ──────────────────────────────────────────────────────
    if not test_path.exists():
        print(f"[ERROR] test file not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    samples = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
    if args.limit > 0:
        samples = samples[: args.limit]
    total = len(samples)

    print(f"ADBA Baseline Eval — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Model   : {EVAL_MODEL}")
    print(f"Samples : {total}")
    print(f"Output  : {out_path}")
    print("─" * 60)

    # ── DB connection ─────────────────────────────────────────────────────
    pg_conn = None
    if not args.dry_run:
        try:
            pg_conn = psycopg2.connect(POSTGRES_URL)
            pg_conn.autocommit = False
            print(f"[DB] Connected to {POSTGRES_URL}")
        except Exception as e:
            print(f"[WARN] Cannot connect to PostgreSQL ({e}). SQL exec checks disabled.")

    # ── Skill distribution preview ────────────────────────────────────────
    from collections import Counter
    skill_preview = Counter(
        detect_skill(s["messages"][0]["content"] if s["messages"][0]["role"] == "system" else "")
        for s in samples
    )
    print("[Skill distribution]", dict(skill_preview))
    print("─" * 60)

    # ── Run evaluation ────────────────────────────────────────────────────
    results: list[dict] = []
    t_total_start = time.time()

    for i, sample in enumerate(samples, start=1):
        r = evaluate_sample(sample, pg_conn, dry_run=args.dry_run, idx=i)
        results.append(r)
        print(_status(i, total, r))

    total_elapsed = round(time.time() - t_total_start, 1)

    # ── Aggregate ─────────────────────────────────────────────────────────
    summary = aggregate(results)
    summary["total_elapsed_s"] = total_elapsed
    summary["model"]           = EVAL_MODEL
    summary["dry_run"]         = args.dry_run
    summary["test_file"]       = str(test_path)
    summary["timestamp"]       = time.strftime("%Y-%m-%dT%H:%M:%S")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("BASELINE RESULTS")
    print("═" * 60)

    metrics = [
        ("SQL Execution Accuracy",    "sql_execution_accuracy",  "≥82%"),
        ("SQL Heuristic Accuracy",    "sql_heuristic_accuracy",  "≥85%"),
        ("Python Syntax Rate",        "python_syntax_rate",      "≥90%"),
        ("Supervisor JSON Rate",      "supervisor_json_rate",    "≥88%"),
        ("Insight JSON Rate",         "insight_json_rate",       "≥88%"),
        ("Reflector JSON Rate",       "reflector_json_rate",     "≥88%"),
        ("Overall JSON Valid Rate",   "overall_json_valid_rate", "≥88%"),
    ]
    for label, key, target in metrics:
        val = summary.get(key, 0.0)
        # Color coding via ASCII: show target
        status = "✓" if _metric_passes(val, target) else "✗"
        print(f"  {status} {label:<30} {val:>6.1f}%   (target {target})")

    print(f"\n  Samples evaluated : {summary['total_samples']}")
    print(f"  Avg latency       : {summary['avg_latency_s']}s")
    print(f"  Total elapsed     : {total_elapsed}s")
    print(f"  Per-skill counts  : {summary['per_skill_counts']}")
    print("═" * 60)

    # ── Save output ───────────────────────────────────────────────────────
    output_data = {"summary": summary, "results": results}
    out_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
    print(f"\n[Saved] {out_path}")

    if pg_conn:
        pg_conn.close()


def _metric_passes(value: float, target: str) -> bool:
    """Check if metric value meets target string like '≥82%'."""
    m = re.match(r"[≥>=<]+\s*(\d+(?:\.\d+)?)%", target)
    if not m:
        return False
    threshold = float(m.group(1))
    if "≥" in target or ">=" in target:
        return value >= threshold
    if "<" in target:
        return value < threshold
    return False


if __name__ == "__main__":
    main()
