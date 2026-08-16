"""Eval tầng 1: retriever có chọn ĐỦ tập bảng đúng không?

Không LLM, không GPU, không database. Tập bảng đúng parse từ SQL mẫu.
Chạy hết vài giây cho hàng nghìn câu — đây là vòng lặp để chỉnh retriever.

"Đủ" chứ không phải "có": thiếu một bảng JOIN là SQL sai chắc chắn, nên
không tính điểm từng phần. Xem spec mục 6.1 và 6.5.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from eval.datasets import EvalRecord, load_adba_golden, load_normalized
from perception.connection_profile import ALL_TABLES, build_profile
from perception.retrieval import FullRetriever, LexicalRetriever
from perception.schema_context import resolve_schema_context
from perception.sql_tables import tables_in_sql


@dataclass
class RecallReport:
    total: int
    full_hits: int
    recall: float
    avg_context_tables: float
    misses: list[tuple[str, frozenset[str]]] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"câu chấm được   : {self.total}",
            f"phủ trọn vẹn    : {self.full_hits}",
            f"recall          : {self.recall:.3f}",
            f"bảng/context TB : {self.avg_context_tables:.1f}",
        ]
        if self.misses:
            lines.append(f"\n{len(self.misses)} câu trượt (10 câu đầu):")
            for q, missing in self.misses[:10]:
                lines.append(f"  - {q[:70]}  ← thiếu {sorted(missing)}")
        return "\n".join(lines)


def measure_recall(
    records: Sequence[EvalRecord],
    resolve: Callable[[EvalRecord], frozenset[str]],
) -> RecallReport:
    """Đo tỉ lệ câu mà context chứa TOÀN BỘ tập bảng đúng.

    Bản ghi có SQL mẫu không parse ra bảng nào sẽ bị bỏ qua chứ không tính
    là trượt — đó là lỗi dữ liệu, không phải lỗi retriever.
    """
    total = 0
    hits = 0
    ctx_sizes: list[int] = []
    misses: list[tuple[str, frozenset[str]]] = []

    for rec in records:
        gold = tables_in_sql(rec.gold_sql)
        if not gold:
            continue
        total += 1
        got = resolve(rec)
        ctx_sizes.append(len(got))
        missing = gold - got
        if missing:
            misses.append((rec.question, missing))
        else:
            hits += 1

    return RecallReport(
        total=total,
        full_hits=hits,
        recall=(hits / total) if total else 0.0,
        avg_context_tables=(sum(ctx_sizes) / len(ctx_sizes)) if ctx_sizes else 0.0,
        misses=misses,
    )


def _resolver(strategy: str, k: int) -> Callable[[EvalRecord], frozenset[str]]:
    def resolve(rec: EvalRecord) -> frozenset[str]:
        permitted = frozenset(t.name for t in rec.tables)
        threshold = 10**9 if strategy == "full" else 1
        profile = build_profile(
            dsn="", tables=rec.tables,
            grants={"eval": frozenset({ALL_TABLES})},
            threshold_tokens=threshold,
        )
        retriever = (FullRetriever(rec.tables) if strategy == "full"
                     else LexicalRetriever(rec.tables))
        ctx = resolve_schema_context(profile, rec.question, permitted,
                                     retriever=retriever, k=k)
        return frozenset(ctx.retrieved_tables)

    return resolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo recall chọn bảng (eval tầng 1)")
    ap.add_argument("--golden", type=Path, required=False, help="JSONL {question, sql}")
    ap.add_argument("--info-box", type=Path, default=Path("perception/info_box_all.json"))
    ap.add_argument("--normalized", type=Path,
                     help="thư mục chứa questions.jsonl + schemas.json")
    ap.add_argument("--strategy", choices=["full", "lexical"], default="lexical")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--json", type=Path, help="ghi báo cáo dạng JSON ra đây")
    args = ap.parse_args()

    if bool(args.golden) == bool(args.normalized):
        ap.error("phải cung cấp đúng một trong hai cờ: --golden hoặc --normalized")

    if args.normalized:
        records = load_normalized(args.normalized / "questions.jsonl",
                                   args.normalized / "schemas.json")
    else:
        records = load_adba_golden(args.golden, args.info_box)
    report = measure_recall(records, _resolver(args.strategy, args.k))
    print(f"[{args.strategy}, k={args.k}]")
    print(report.as_text())

    if args.json:
        args.json.write_text(json.dumps({
            "strategy": args.strategy, "k": args.k,
            "total": report.total, "full_hits": report.full_hits,
            "recall": report.recall,
            "avg_context_tables": report.avg_context_tables,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
