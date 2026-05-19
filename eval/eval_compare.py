"""
Compare baseline vs finetuned eval JSON summaries.

  python eval/eval_compare.py
  python eval/eval_compare.py --baseline eval/baseline_results.json --finetuned eval/finetuned_checkpoint50_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

METRICS = [
    ("sql_execution_accuracy", "SQL Execution"),
    ("sql_heuristic_accuracy", "SQL Heuristic"),
    ("python_syntax_rate", "Python Syntax"),
    ("supervisor_json_rate", "Supervisor JSON"),
    ("insight_json_rate", "Insight JSON"),
    ("reflector_json_rate", "Reflector JSON"),
    ("overall_json_valid_rate", "Overall JSON"),
    ("avg_latency_s", "Avg latency (s)"),
]


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(ROOT / "eval" / "baseline_results.json"))
    parser.add_argument("--finetuned", default=str(ROOT / "eval" / "finetuned_checkpoint50_results.json"))
    parser.add_argument("--output", default=str(ROOT / "eval" / "comparison_checkpoint50.json"))
    args = parser.parse_args()

    base_path = Path(args.baseline)
    ft_path = Path(args.finetuned)
    if not base_path.exists():
        raise SystemExit(f"Missing baseline: {base_path}")
    if not ft_path.exists():
        raise SystemExit(f"Missing finetuned: {ft_path}. Run eval/eval_peft_runner.py first.")

    b = load_summary(base_path)
    f = load_summary(ft_path)

    rows = []
    print("\n" + "=" * 72)
    print("BASELINE vs FINETUNED (checkpoint-50)")
    print("=" * 72)
    print(f"{'Metric':<22} {'Baseline':>12} {'Finetuned':>12} {'Delta':>12}")
    print("-" * 72)

    for key, label in METRICS:
        bv = b.get(key, 0)
        fv = f.get(key, 0)
        if key == "avg_latency_s":
            delta = round(fv - bv, 2)
            delta_str = f"{delta:+.2f}s"
        else:
            delta = round(fv - bv, 1)
            delta_str = f"{delta:+.1f}%"
        print(f"{label:<22} {bv:>11} {fv:>12} {delta_str:>12}")
        rows.append({"metric": key, "label": label, "baseline": bv, "finetuned": fv, "delta": delta})

    print("-" * 72)
    print(f"Baseline model : {b.get('model')}")
    print(f"Finetuned      : {f.get('model')}")
    print(f"Samples        : baseline={b.get('total_samples')} finetuned={f.get('total_samples')}")

    out = {
        "baseline_file": str(base_path),
        "finetuned_file": str(ft_path),
        "comparison": rows,
        "baseline_summary": b,
        "finetuned_summary": f,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Saved] {args.output}")


if __name__ == "__main__":
    main()
