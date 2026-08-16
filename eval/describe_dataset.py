"""Đo cấu hình thực tế của một bộ dữ liệu eval.

Tồn tại để bảng cấu hình trong eval/README_multischema.md được SINH RA chứ
không viết tay. Spec mục 6.2 nói rõ: các con số này không được chốt từ trí
nhớ.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.datasets import EvalRecord, load_normalized


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    n_databases: int
    n_questions: int
    avg_tables_per_db: float
    max_tables_per_db: int

    def as_markdown_row(self) -> str:
        return (f"| {self.name} | {self.n_databases} | {self.n_questions} | "
                f"{self.avg_tables_per_db:.1f} | {self.max_tables_per_db} | |")


def describe(name: str, records: Sequence[EvalRecord]) -> DatasetProfile:
    sizes = {r.db_id: len(r.tables) for r in records}
    return DatasetProfile(
        name=name,
        n_databases=len(sizes),
        n_questions=len(records),
        avg_tables_per_db=(sum(sizes.values()) / len(sizes)) if sizes else 0.0,
        max_tables_per_db=max(sizes.values()) if sizes else 0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo cấu hình một bộ dữ liệu eval")
    ap.add_argument("--name", required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--schemas", type=Path, required=True)
    args = ap.parse_args()

    profile = describe(args.name, load_normalized(args.questions, args.schemas))
    print(profile.as_markdown_row())


if __name__ == "__main__":
    main()
