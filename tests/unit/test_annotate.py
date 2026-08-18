import json

from perception.annotate import annotate_schema, build_annotation_prompt
from perception.schema_model import Column, Table

TABLES = (
    Table(
        name="orders",
        columns=(Column("id", "integer"), Column("flg_tt", "boolean")),
        primary_key=("id",),
    ),
)
SAMPLES = {"orders": [{"id": 1, "flg_tt": True}]}


def _reply(payload: dict):
    return lambda system, user: json.dumps(payload, ensure_ascii=False)  # noqa: ARG005


def test_produces_a_table_description():
    ann, failures = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {"text": "Đơn hàng bán", "confidence": "high"},
                "columns": {}}),
    )
    assert ann.tables["orders"].text == "Đơn hàng bán"
    assert failures == 0


def test_produces_column_descriptions():
    ann, failures = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {"text": "Đơn hàng", "confidence": "high"},
                "columns": {"flg_tt": {"text": "Cờ đã thanh toán", "confidence": "low"}}}),
    )
    assert ann.columns["orders"]["flg_tt"].text == "Cờ đã thanh toán"
    assert ann.columns["orders"]["flg_tt"].confidence == "low"
    assert failures == 0


def test_everything_it_generates_is_marked_as_llm_not_human():
    ann, failures = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {"text": "x", "confidence": "high"},
                "columns": {"flg_tt": {"text": "y", "confidence": "high"}}}),
    )
    assert ann.tables["orders"].reviewed_by == "llm"
    assert ann.columns["orders"]["flg_tt"].reviewed_by == "llm"
    assert failures == 0


def test_an_unparseable_reply_becomes_low_confidence_not_a_crash():
    """Onboarding chạy trên 150 bảng; một lần model trả rác không được làm hỏng cả lượt."""
    ann, failures = annotate_schema(TABLES, SAMPLES, lambda s, u: "không phải json")  # noqa: ARG005
    assert ann.tables["orders"].confidence == "low"
    assert ann.tables["orders"].text == ""
    assert failures == 1


def test_a_confidence_value_outside_the_allowed_set_is_treated_as_low():
    ann, failures = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {"text": "x", "confidence": "rất chắc chắn"}, "columns": {}}),
    )
    assert ann.tables["orders"].confidence == "low"
    assert failures == 0


def test_the_prompt_carries_column_names_and_sample_values():
    prompt = build_annotation_prompt(TABLES[0], SAMPLES["orders"])
    assert "flg_tt" in prompt
    assert "orders" in prompt


def test_the_prompt_asks_for_vietnamese():
    assert "tiếng Việt" in build_annotation_prompt(TABLES[0], SAMPLES["orders"])


# --- Amendment 1: invoke raising must not kill the run ---

def test_invoke_raising_becomes_low_confidence_not_a_crash():
    """Ollama down/timeout/OOM must not discard the tables already annotated."""

    def _raise(system, user):
        raise ConnectionError("connection refused")

    ann, failures = annotate_schema(TABLES, SAMPLES, _raise)
    assert ann.tables["orders"].text == ""
    assert ann.tables["orders"].confidence == "low"
    assert ann.tables["orders"].reviewed_by == "llm"
    assert failures == 1


def test_one_table_raising_does_not_stop_the_others():
    tables = TABLES + (
        Table(
            name="customers",
            columns=(Column("id", "integer"),),
            primary_key=("id",),
        ),
    )
    samples = dict(SAMPLES, customers=[{"id": 1}])

    def _invoke(system, user):
        if "orders" in user:
            raise TimeoutError("timed out")
        return json.dumps({"table": {"text": "Khách hàng", "confidence": "high"}, "columns": {}})

    ann, failures = annotate_schema(tables, samples, _invoke)
    assert ann.tables["orders"].confidence == "low"
    assert ann.tables["customers"].text == "Khách hàng"
    assert failures == 1


# --- Amendment 2: _parse must tolerate prose on both sides of the JSON ---

def test_prose_on_both_sides_of_the_json_is_still_parsed():
    def _invoke(system, user):
        payload = json.dumps(
            {"table": {"text": "Đơn hàng bán", "confidence": "high"}, "columns": {}},
            ensure_ascii=False,
        )
        return f"Đây là kết quả: {payload} Hy vọng giúp ích."

    ann, failures = annotate_schema(TABLES, SAMPLES, _invoke)
    assert ann.tables["orders"].text == "Đơn hàng bán"
    assert failures == 0


# --- Amendment 3: row_count=None must not render as the English "None" ---

def test_unknown_row_count_renders_as_vietnamese_placeholder():
    table = Table(
        name="orders",
        columns=(Column("id", "integer"),),
        primary_key=("id",),
        row_count=None,
    )
    prompt = build_annotation_prompt(table, [])
    assert "(chưa biết)" in prompt
    assert "None" not in prompt


def test_known_row_count_still_renders():
    table = Table(
        name="orders",
        columns=(Column("id", "integer"),),
        primary_key=("id",),
        row_count=42,
    )
    prompt = build_annotation_prompt(table, [])
    assert "42" in prompt


# --- Amendment 4: sample values must be truncated ---

def test_long_sample_values_are_truncated():
    long_value = "x" * 500
    table = TABLES[0]
    prompt = build_annotation_prompt(table, [{"id": 1, "flg_tt": long_value}])
    assert long_value not in prompt
    assert ("x" * 200 + "…") in prompt


def test_short_sample_values_are_not_truncated():
    prompt = build_annotation_prompt(TABLES[0], [{"id": 1, "flg_tt": "short"}])
    assert '"short"' in prompt


# --- Amendment 5: annotate_schema returns (SchemaAnnotations, failure_count) ---

def test_failure_count_is_zero_on_a_clean_run():
    ann, failures = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {"text": "Đơn hàng bán", "confidence": "high"}, "columns": {}}),
    )
    assert failures == 0


def test_failure_count_equals_table_count_when_invoke_always_raises():
    tables = TABLES + (
        Table(
            name="customers",
            columns=(Column("id", "integer"),),
            primary_key=("id",),
        ),
    )
    samples = dict(SAMPLES, customers=[{"id": 1}])

    def _raise(system, user):
        raise RuntimeError("boom")

    ann, failures = annotate_schema(tables, samples, _raise)
    assert failures == len(tables) == 2


# --- Fix round 1 ---

# Fix 1: _parse must not lock onto the first "{" when it belongs to prose,
# not the JSON payload — it must keep scanning candidate braces until one
# of them decodes.

def test_a_brace_inside_leading_prose_does_not_defeat_parsing():
    def _invoke(system, user):
        return (
            'Kết quả (dạng {key:val}): '
            '{"table": {"text": "Đơn hàng", "confidence": "high"}, "columns": {}}'
        )

    ann, failures = annotate_schema(TABLES, SAMPLES, _invoke)
    assert ann.tables["orders"].text == "Đơn hàng"
    assert failures == 0


def test_many_prose_braces_do_not_consume_the_candidate_budget():
    """Dấu ngoặc trong văn xuôi không được tính là ứng viên JSON.

    `_OBJECT_START` chỉ khớp `{` mở ra object thật (`{"` hoặc `{}`), nên
    hàng nghìn dấu ngoặc kiểu `{a{b{c` bị bỏ qua mà không tốn một lần
    `raw_decode` hỏng nào — và JSON thật nằm sau chúng vẫn được tìm thấy.
    """
    noise = "{a{b{c " * 500

    def _invoke(system, user):
        return (
            f"{noise}"
            '{"table": {"text": "Đơn hàng", "confidence": "high"}, "columns": {}}'
        )

    ann, failures = annotate_schema(TABLES, SAMPLES, _invoke)
    assert ann.tables["orders"].text == "Đơn hàng"
    assert failures == 0


def test_a_reply_that_is_only_open_braces_fails_instead_of_hanging():
    """Model local lặp vô hạn một ký tự là chuyện có thật.

    Mỗi `json.JSONDecodeError` tính lineno/colno bằng cách quét lại cả văn
    bản, nên thử mọi dấu ngoặc là O(N × độ dài). Trần ứng viên giữ cho một
    câu trả lời rác chỉ làm hỏng chú giải của MỘT bảng, không treo cả lượt
    chạy 150 bảng.
    """

    def _invoke(system, user):
        return '{"' * 50_000

    ann, failures = annotate_schema(TABLES, SAMPLES, _invoke)
    assert ann.tables["orders"].text == ""
    assert failures == 1


# Fix 2: a well-formed but empty reply must count as a failure — the
# failure count now tracks "did this table end up with no text", not
# "did parsing raise".

def test_a_well_formed_but_empty_reply_counts_as_a_failure():
    ann, failures = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {}, "columns": {}}),
    )
    assert ann.tables["orders"].text == ""
    assert ann.tables["orders"].confidence == "low"
    assert ann.tables["orders"].reviewed_by == "llm"
    assert failures == 1


def test_a_well_formed_but_empty_reply_for_every_table_counts_all_of_them():
    tables = TABLES + (
        Table(
            name="customers",
            columns=(Column("id", "integer"),),
            primary_key=("id",),
        ),
    )
    samples = dict(SAMPLES, customers=[{"id": 1}])

    ann, failures = annotate_schema(
        tables, samples,
        _reply({"table": {}, "columns": {}}),
    )
    assert failures == len(tables) == 2


# Fix 3: pin the 200-character truncation boundary exactly.

def test_a_200_character_value_is_not_truncated():
    value = "x" * 200
    prompt = build_annotation_prompt(TABLES[0], [{"id": 1, "flg_tt": value}])
    assert f'"{value}"' in prompt
    assert "…" not in prompt


def test_a_201_character_value_is_truncated():
    value = "x" * 201
    prompt = build_annotation_prompt(TABLES[0], [{"id": 1, "flg_tt": value}])
    assert f'"{value}"' not in prompt
    assert ("x" * 200 + "…") in prompt
