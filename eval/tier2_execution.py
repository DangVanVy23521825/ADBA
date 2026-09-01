"""Eval tầng 2: SQL sinh ra có trả về ĐÚNG KẾT QUẢ không?

Tầng 1 (`eval/tier1_recall.py`) hỏi "context có đủ bảng đúng không". Tầng
này hỏi câu tiếp theo và đắt hơn nhiều: chạy SQL sinh ra cạnh SQL vàng
trên cùng một database, so hai tập kết quả.

Vì sao cần: hai thước đo đang có đều không đo được điều khách quan tâm.
Tầng 1 không thấy chất lượng SQL — `few_shots` chỉ vào prompt sinh SQL,
không chạm đường retrieval, nên tầng 1 mù với nó theo đúng thiết kế. Còn
`eval_runner.sql_execution_accuracy` chỉ kiểm `EXPLAIN` chạy trót lọt, tức
là SQL đúng cú pháp và gọi đúng tên bảng có thật — `SELECT 1 FROM orders`
qua `EXPLAIN` ngon lành.

KHÔNG so chuỗi SQL. Hai câu khác hẳn nhau về hình thức vẫn có thể cùng
đúng, và so chuỗi phạt oan đúng những lời giải hay.

Hàm đo là hàm THUẦN, nhận `generate` và `execute` làm callable — giống
`measure_recall`. Nhờ vậy toàn bộ ngữ nghĩa so sánh test được mà không cần
database lẫn model.

Thiết kế đầy đủ và lý do từng đánh đổi:
docs/superpowers/plans/2026-09-01-eval-tang-2-execution-accuracy.md
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path

import sqlparse
from sqlparse.sql import Parenthesis
from sqlparse.tokens import Keyword

from eval.datasets import EvalRecord

# Làm tròn số thực trước khi so. Postgres trả `Decimal` cho `numeric` và
# `float` cho `double precision`, nên cùng một phép SUM có thể ra
# Decimal('1234.00') bên này và 1234.0 bên kia. Sáu chữ số là thừa an toàn
# cho cột tiền, mà vẫn nuốt được sai khác do thứ tự cộng dồn của floating
# point khi hai câu SQL gộp nhóm theo đường khác nhau.
_FLOAT_PLACES = 6


class Verdict(str, Enum):
    """Kết luận của phép SO SÁNH, không phải của cả lượt chấm.

    Lỗi chạy và quá hạn giờ không nằm ở đây: chúng phát sinh trước khi có
    gì để so, và `measure_execution` phân loại chúng riêng.
    """

    MATCH = "match"
    MISMATCH = "mismatch"
    BOTH_EMPTY = "both_empty"


class QueryError(Exception):
    """SQL không chạy được: sai cú pháp, sai tên bảng/cột, thiếu quyền."""


class QueryTimeout(QueryError):
    """SQL vượt `statement_timeout`.

    Là lớp con của `QueryError` để nơi gọi chỉ cần bắt `QueryError` nếu
    không quan tâm phân biệt, nhưng `measure_execution` thì có phân biệt:
    một câu quá hạn giờ là tín hiệu khác hẳn một câu sai cú pháp — nó
    thường nghĩa là thiếu điều kiện nối, và đó là lỗi sửa được.
    """


@dataclass(frozen=True)
class QueryResult:
    rows: tuple[tuple, ...]
    truncated: bool = False


def _normalize_value(v):
    """Đưa một ô về dạng so sánh được giữa hai đường SQL khác nhau."""
    if isinstance(v, bool):
        # Gắn nhãn kiểu, không trả `v` trần. Xét trước `int` là cần nhưng
        # KHÔNG đủ: trong Python `bool` là lớp con của `int` và `True == 1`
        # cho ra True, nên trả `v` trần thì một cột boolean vẫn khớp nhầm
        # một cột số chỉ chứa 0 và 1 — phép so tuple dùng `==`. Nhãn làm
        # hai kiểu không bao giờ bằng nhau, mà vẫn phân biệt True với False.
        #
        # Chọn chặt là có chủ đích: `CASE WHEN ... THEN 1 ELSE 0` trả về
        # thứ mà người đọc báo cáo nhìn thấy là 1/0 chứ không phải
        # true/false, nên đó là kết quả khác thật, không phải khác hình
        # thức. Và chặt thì sai về phía không thổi phồng điểm.
        return ("bool", v)
    if isinstance(v, Decimal):
        return round(float(v), _FLOAT_PLACES)
    if isinstance(v, float):
        return round(v, _FLOAT_PLACES)
    if isinstance(v, int):
        # int giữ nguyên (không hoá float): số nguyên lớn hơn 2^53 mất chính
        # xác khi qua float, mà khoá chính bigint thì hay nằm ở vùng đó.
        return v
    if isinstance(v, memoryview):
        return bytes(v)
    return v


def _normalize_rows(rows: Sequence[Sequence]) -> tuple[tuple, ...]:
    return tuple(tuple(_normalize_value(v) for v in row) for row in rows)


def _sort_key(row: tuple):
    """Khoá sắp xếp tổng cho tuple chứa kiểu hỗn hợp.

    So trực tiếp `None < 1` hay `1 < "a"` ném TypeError trong Python 3, mà
    một tập kết quả bất kỳ đều có thể trộn NULL với số với chuỗi. Sắp theo
    (tên kiểu, repr) cho một thứ tự toàn phần ổn định — nội dung của thứ tự
    đó không quan trọng, chỉ cần hai bên dùng chung một thứ tự.
    """
    return tuple((type(v).__name__, repr(v)) for v in row)


def has_top_level_order_by(sql: str) -> bool:
    """SQL này có `ORDER BY` ở mức NGOÀI CÙNG không?

    Quyết định phép so có xét thứ tự dòng hay không. Không có `ORDER BY`
    thì Postgres không cam kết thứ tự và thứ tự có thể đổi giữa hai lượt
    chạy, nên so có thứ tự sẽ đẻ ra trượt giả. Có `ORDER BY` thì câu hỏi
    thật sự hỏi về xếp hạng ("5 khách mua nhiều nhất") và thứ tự là một
    phần của đáp án.

    Bắt buộc duyệt token, KHÔNG khớp chuỗi: `ORDER BY` nằm trong subquery
    hoặc trong `OVER (ORDER BY ...)` của window function không làm cho kết
    quả ngoài cùng có thứ tự. Khớp chuỗi `"order by" in sql.lower()` sẽ
    nhận nhầm cả hai, và nhận nhầm theo hướng nguy hiểm: nó bật chế độ so
    chặt hơn cho những câu vốn không cam kết thứ tự, biến nhiễu thành
    trượt.

    Đi cùng lối của `perception/sql_tables.py`: duyệt token phẳng, không
    chui vào `Parenthesis`.
    """
    for statement in sqlparse.parse(sql):
        for token in statement.tokens:
            if isinstance(token, Parenthesis):
                continue
            if token.ttype is Keyword and token.normalized.upper().startswith("ORDER"):
                return True
    return False


def compare_results(
    gold: Sequence[Sequence],
    pred: Sequence[Sequence],
    *,
    ordered: bool,
) -> Verdict:
    """So hai tập kết quả. Xem mục 3 của tài liệu thiết kế cho từng đánh đổi.

    `ordered=False` so như MULTISET, không phải như set: nếu SQL vàng trả
    ba dòng giống nhau mà SQL sinh ra trả một dòng thì KHÔNG tính đạt. Đây
    là chỗ cố ý khác bộ chấm chính thức của BIRD — họ dùng `set`, và điều
    đó tha nhầm những câu thiếu `GROUP BY`. Nhớ ghi chú khi đem số đi so
    với con số đã công bố của BIRD.

    Cả hai bên rỗng KHÔNG trả về MATCH mà trả BOTH_EMPTY, để nơi gọi đếm
    riêng. Về kỹ thuật thì rỗng khớp rỗng, nhưng một câu SQL sai cũng rất
    hay trả về rỗng, nên gộp vào tỉ lệ đạt là tự thổi phồng — nhất là trên
    database nhỏ nơi nhiều câu hỏi vốn không có dữ liệu trả về.
    """
    g = _normalize_rows(gold)
    p = _normalize_rows(pred)

    if not g and not p:
        return Verdict.BOTH_EMPTY

    if ordered:
        return Verdict.MATCH if g == p else Verdict.MISMATCH

    if len(g) != len(p):
        return Verdict.MISMATCH
    same = sorted(g, key=_sort_key) == sorted(p, key=_sort_key)
    return Verdict.MATCH if same else Verdict.MISMATCH


def compare_ignoring_column_order(
    gold: Sequence[Sequence],
    pred: Sequence[Sequence],
    *,
    ordered: bool,
) -> Verdict:
    """Số CHẨN ĐOÁN: bao nhiêu phần của khoảng cách chỉ là hình thức?

    `SELECT a, b` với `SELECT b, a` chứa cùng thông tin nhưng khác thứ tự
    tuple, nên `compare_results` (so chặt theo vị trí, giống BIRD) tính là
    trượt. Con số này cho biết chuyện đó xảy ra nhiều đến đâu.

    ⚠️ Hàm này THA QUÁ TAY, cố ý. Nó sắp giá trị trong TỪNG dòng một cách
    độc lập, nên nó chấp nhận cả những trường hợp mỗi dòng hoán vị một
    kiểu — thứ không tương ứng với bất kỳ phép hoán vị cột nào. Làm đúng
    đắn là tìm một hoán vị cột DÙNG CHUNG cho mọi dòng, nhưng đó là bài
    toán đắt và không đáng cho một con số chẩn đoán.

    Vì vậy nó là CẬN TRÊN của "độ chính xác nếu bỏ qua thứ tự cột", không
    phải con số để công bố. Con số công bố là `exec_accuracy`.
    """
    g = tuple(tuple(sorted(r, key=lambda v: (type(v).__name__, repr(v))))
              for r in _normalize_rows(gold))
    p = tuple(tuple(sorted(r, key=lambda v: (type(v).__name__, repr(v))))
              for r in _normalize_rows(pred))
    return compare_results(g, p, ordered=ordered)


@dataclass
class ExecReport:
    """Các bucket, KHÔNG gộp thành một con số.

    Gộp lại thì mất đúng thông tin cần để hành động: "SQL sai cú pháp" và
    "SQL chạy được nhưng ra kết quả khác" đòi hai cách sửa khác hẳn nhau.
    """

    total: int = 0
    match: int = 0
    mismatch: int = 0
    both_empty: int = 0
    pred_error: int = 0
    pred_timeout: int = 0
    truncated: int = 0
    gold_error: int = 0
    loose_match: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def scored(self) -> int:
        """Mẫu số: bỏ những câu mà chính SQL VÀNG hỏng.

        Cùng lối với `measure_recall`, nơi bản ghi có SQL vàng không parse
        ra bảng nào bị bỏ qua chứ không tính là trượt: lỗi dữ liệu không
        phải lỗi hệ thống.
        """
        return self.total - self.gold_error

    @property
    def exec_accuracy(self) -> float:
        return (self.match / self.scored) if self.scored else 0.0

    def as_text(self) -> str:
        lines = [
            f"câu chấm được   : {self.scored}  (trên {self.total} câu đọc vào)",
            f"exec accuracy   : {self.exec_accuracy:.3f}",
            "",
            f"  khớp          : {self.match}",
            f"  khác kết quả  : {self.mismatch}",
            f"  cả hai rỗng   : {self.both_empty}   (KHÔNG tính là khớp)",
            f"  SQL sinh lỗi  : {self.pred_error}",
            f"  quá hạn giờ   : {self.pred_timeout}",
            f"  cụt kết quả   : {self.truncated}",
        ]
        if self.gold_error:
            lines += [
                "",
                f"⚠️  {self.gold_error} câu có SQL VÀNG chạy lỗi — đã trừ khỏi mẫu số.",
                "   Con số này cao nghĩa là golden set hỏng, và mọi số ở trên đáng ngờ.",
            ]
        if self.loose_match > self.match:
            lines += [
                "",
                f"Bỏ qua thứ tự cột thì lên tới {self.loose_match} câu "
                f"(+{self.loose_match - self.match}). Đây là CẬN TRÊN, không",
                "phải số công bố — xem docstring compare_ignoring_column_order.",
            ]
        if self.failures:
            lines.append(f"\n{len(self.failures)} câu không đạt (10 câu đầu):")
            for q, why in self.failures[:10]:
                lines.append(f"  - [{why}] {q[:66]}")
        return "\n".join(lines)


def measure_execution(
    records: Sequence[EvalRecord],
    generate: Callable[[EvalRecord], str],
    execute: Callable[[str], QueryResult],
) -> ExecReport:
    """Chấm execution accuracy trên `records`.

    `generate` sinh SQL từ một bản ghi; `execute` chạy SQL và trả
    `QueryResult`, ném `QueryError`/`QueryTimeout` khi hỏng. Cả hai là
    callable để hàm này test được không cần database lẫn model.

    Thứ tự CỐ Ý: chạy SQL vàng TRƯỚC khi gọi model. Một bản ghi vàng hỏng
    thì không tốn lượt gọi model nào — mà lượt gọi model là toàn bộ chi phí
    của tầng này (1.534 câu của BIRD mất vài giờ mỗi nhánh).
    """
    report = ExecReport()

    for rec in records:
        report.total += 1

        try:
            gold_res = execute(rec.gold_sql)
        except QueryError as e:
            report.gold_error += 1
            report.failures.append((rec.question, f"gold_error: {type(e).__name__}"))
            continue

        try:
            pred_sql = generate(rec)
            pred_res = execute(pred_sql)
        except QueryTimeout:
            report.pred_timeout += 1
            report.failures.append((rec.question, "pred_timeout"))
            continue
        except QueryError as e:
            report.pred_error += 1
            report.failures.append((rec.question, f"pred_error: {e}"[:80]))
            continue
        except Exception as e:  # noqa: BLE001 — model/mạng hỏng, không giết cả lượt chạy
            # Kèm THÔNG ĐIỆP, không chỉ tên kiểu: một lượt chạy hàng giờ mà
            # báo "8 câu ValueError" thì phải chạy lại mới chẩn được, và
            # chạy lại là thứ đắt nhất ở tầng này.
            report.pred_error += 1
            report.failures.append((rec.question, f"generate: {type(e).__name__}: {e}"[:120]))
            continue

        if gold_res.truncated or pred_res.truncated:
            # Không tính đạt cũng không tính trượt: khi một bên bị cắt cụt
            # thì phép so không còn nói lên điều gì về tính đúng.
            report.truncated += 1
            report.failures.append((rec.question, "truncated"))
            continue

        ordered = has_top_level_order_by(rec.gold_sql)
        verdict = compare_results(gold_res.rows, pred_res.rows, ordered=ordered)

        if compare_ignoring_column_order(
            gold_res.rows, pred_res.rows, ordered=ordered
        ) is Verdict.MATCH:
            report.loose_match += 1

        if verdict is Verdict.MATCH:
            report.match += 1
        elif verdict is Verdict.BOTH_EMPTY:
            report.both_empty += 1
        else:
            report.mismatch += 1
            report.failures.append((rec.question, "mismatch"))

    return report


def make_executor(conn, *, timeout_s: int = 30, max_rows: int = 10_000):
    """Bọc một kết nối psycopg2 thành `execute` mà `measure_execution` cần.

    Bốn lớp an toàn, đều bắt buộc vì ta đang chạy SQL do model sinh ra:

    1. CHỈ ĐỌC — `SET TRANSACTION READ ONLY`, áp cho cả SQL vàng. Một câu
       SQL sinh ra mang `DELETE` sẽ bị chính Postgres chặn, không phải dựa
       vào việc ta rà chuỗi.
    2. HẠN GIỜ — `statement_timeout`. Một phép nối chéo trên 29.830 dòng
       `orders` treo vô hạn, và một lượt chạy 1.534 câu thì không ai ngồi
       canh.
    3. TRẦN DÒNG — `fetchmany`. Không chèn được `LIMIT` vào SQL tuỳ ý một
       cách an toàn, nên chặn ở phía client và BÁO CÁO việc bị cắt thay vì
       lặng lẽ so hai tập cụt.
    4. LUÔN ROLLBACK — kể cả khi truy vấn thành công.

    Lấy `max_rows + 1` dòng để phân biệt "vừa đủ trần" với "vượt trần";
    lấy đúng `max_rows` thì hai trường hợp đó không phân biệt được.
    """
    def execute(sql: str) -> QueryResult:
        import psycopg2

        try:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(f"SET LOCAL statement_timeout = {int(timeout_s * 1000)}")
                cur.execute(sql)
                if cur.description is None:
                    # Câu lệnh không trả bảng nào (ví dụ model sinh ra DDL).
                    # Coi là rỗng chứ không phải lỗi — nó chạy được, chỉ là
                    # không trả lời câu hỏi, và `mismatch` mô tả đúng hơn.
                    return QueryResult(rows=(), truncated=False)
                fetched = cur.fetchmany(max_rows + 1)
                return QueryResult(
                    rows=tuple(tuple(r) for r in fetched[:max_rows]),
                    truncated=len(fetched) > max_rows,
                )
        except psycopg2.errors.QueryCanceled as e:
            raise QueryTimeout(str(e)) from e
        except psycopg2.Error as e:
            raise QueryError(str(e)) from e
        finally:
            conn.rollback()

    return execute


def _load_records(
    golden: Path, *, profile_dir: Path | None, info_box: Path | None
) -> tuple[list[EvalRecord], object]:
    """Đọc golden set cùng schema của nó. Trả `(records, profile)`.

    Hai nguồn schema, vì hai bộ dữ liệu có hình dạng khác nhau:

    - `--profile-dir`: thư mục do `onboard.py` sinh ra. Đây là đường CHÍNH,
      vì nó đúng thứ khách hàng thật sự có, và `read_profile` trả về bảng
      ĐÃ GẮN CHÚ GIẢI — tức là đo đúng cái sản phẩm bán. BIRD chỉ có đường
      này: nó không có info_box.
    - `--info-box`: đường cũ của golden set ADBA.
    """
    import json

    from perception.connection_profile import ALL_TABLES, build_profile
    from perception.profile_store import read_profile

    if profile_dir:
        profile = read_profile(profile_dir)
    else:
        from perception.schema_model import tables_from_info_box

        tables = tables_from_info_box(json.loads(Path(info_box).read_text(encoding="utf-8")))
        profile = build_profile(
            dsn="", tables=tables, grants={"eval": frozenset({ALL_TABLES})}
        )

    records: list[EvalRecord] = []
    for line in Path(golden).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        records.append(
            EvalRecord(
                question=row["question"],
                gold_sql=row["sql"],
                db_id=row.get("db_id", "default"),
                tables=profile.tables,
            )
        )
    return records, profile


def _translate_gold(records: Sequence[EvalRecord], tables) -> list[EvalRecord]:
    """Dịch SQL vàng SQLite -> Postgres. Xem eval/sqlite_dialect.py.

    Bắt buộc cho BIRD/Spider: SQL vàng của chúng viết cho SQLite, và để
    nguyên thì chỉ 34,6% chạy được trên Postgres — 65% còn lại sẽ vào
    bucket `gold_error` và lượt đo thành vô nghĩa.
    """
    import dataclasses

    from eval.sqlite_dialect import SchemaNames, translate

    names = SchemaNames(tables)
    return [dataclasses.replace(r, gold_sql=translate(r.gold_sql, names)) for r in records]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Đo execution accuracy (eval tầng 2). Cần database sống."
    )
    ap.add_argument("--golden", type=Path, required=True, help="JSONL {question, sql}")
    ap.add_argument("--profile-dir", type=Path,
                    help="thư mục profile do onboard.py sinh (chú giải đã gắn sẵn)")
    ap.add_argument("--info-box", type=Path,
                    help="đường cũ cho golden set ADBA")
    ap.add_argument("--translate-gold", action="store_true",
                    help="dịch SQL vàng SQLite -> Postgres (bắt buộc cho BIRD/Spider)")
    ap.add_argument("--dsn", help="DSN Postgres; mặc định lấy POSTGRES_URL/DATABASE_URL")
    ap.add_argument("--limit", type=int,
                    help="chỉ chấm N câu đầu (lượt chạy đầy đủ tốn vài giờ)")
    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=10_000)
    ap.add_argument("--json", type=Path, help="ghi báo cáo dạng JSON ra đây")
    args = ap.parse_args()

    if bool(args.profile_dir) == bool(args.info_box):
        ap.error("phải cung cấp đúng một trong hai cờ: --profile-dir hoặc --info-box")

    import os

    import psycopg2

    dsn = args.dsn or os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if not dsn:
        ap.error("cần --dsn, hoặc đặt DATABASE_URL / POSTGRES_URL")

    records, profile = _load_records(
        args.golden, profile_dir=args.profile_dir, info_box=args.info_box
    )
    if args.translate_gold:
        records = _translate_gold(records, profile.tables)
    if args.limit:
        records = records[: args.limit]

    print(f"{len(records)} câu, {len(profile.tables)} bảng, "
          f"schema_mode={profile.schema_mode}", flush=True)

    conn = psycopg2.connect(dsn)
    try:
        execute = make_executor(conn, timeout_s=args.timeout_s, max_rows=args.max_rows)
        report = measure_execution(records, _generator(profile), execute)
    finally:
        conn.close()

    print(report.as_text())

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "total": report.total,
                    "scored": report.scored,
                    "exec_accuracy": report.exec_accuracy,
                    "match": report.match,
                    "mismatch": report.mismatch,
                    "both_empty": report.both_empty,
                    "pred_error": report.pred_error,
                    "pred_timeout": report.pred_timeout,
                    "truncated": report.truncated,
                    "gold_error": report.gold_error,
                    "loose_match": report.loose_match,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _generator(profile) -> Callable[[EvalRecord], str]:
    """Sinh SQL qua đúng đường mà production dùng.

    Nhập bên trong hàm chứ không ở đầu file: `measure_execution` và toàn bộ
    phần so sánh phải nhập được mà không kéo theo tầng model, để test chạy
    không cần Ollama.
    """
    from graph.agents.sql_agent import _extract_sql, build_system_prompt
    from model.model_client import ModelClient
    from perception.retrieval import LexicalRetriever
    from perception.schema_context import resolve_schema_context
    from perception.sql_identifiers import requote_for_tables

    # `enable_openai_fallback=False` cố ý, giống `onboard.py`: một lượt chấm
    # BIRD gửi hàng nghìn schema qua model, và nếu Ollama chết giữa chừng
    # thì lặng lẽ đẩy sang OpenAI là vừa phá lời hứa on-prem vừa làm số đo
    # trộn hai model khác nhau — không còn nghĩa gì.
    client = ModelClient(agent_type="sql", enable_openai_fallback=False)
    permitted = profile.table_names()

    # Dựng ĐÚNG như app.py: `resolve_schema_context` cố ý ném khi
    # schema_mode='retrieval' mà không có retriever, không tự lùi về chế độ
    # full — im lặng lùi sẽ đo một hệ thống khác hệ thống đang bán. Ở chế
    # độ full thì `FullRetriever` là đường production tương ứng.
    retriever = LexicalRetriever(profile.tables)

    def generate(rec: EvalRecord) -> str:
        ctx = resolve_schema_context(profile, rec.question, permitted, retriever=retriever)
        raw = client.invoke(build_system_prompt(ctx), f"Task: {rec.question}")
        # Đi qua đúng CHUỖI hậu xử lý mà `sql_agent_node` dùng, cả hai
        # bước. Bỏ sót một bước là đo một hệ thống không tồn tại: lần đầu
        # tôi chỉ gọi `_extract_sql`, và lượt đo báo bản sửa bọc nháy
        # "không ăn thua" trong khi nó chưa hề được chạy.
        return requote_for_tables(_extract_sql(raw), ctx.tables)

    return generate


if __name__ == "__main__":
    main()
