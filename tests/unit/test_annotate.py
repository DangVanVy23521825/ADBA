import json

import model.model_config as model_config
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


def test_a_batch_prompt_only_shows_sample_values_for_its_own_columns():
    """Regression: bảng `Match` thật (bird_all, 115 cột) trượt đúng lỗi này.

    Lô đầu chỉ xin 7 cột trong "Cột:", nhưng `sample_rows` trả về DÒNG ĐẦY
    ĐỦ (mọi cột của bảng) -- nếu dòng mẫu không được lọc theo đúng cột của
    lô, model nhìn thấy các khoá khác trong JSON mẫu và tự ý mô tả thêm
    cột ngoài lô, đẩy output vượt ngân sách token và bị cắt cụt giữa
    chừng -- mất luôn cả mô tả bảng lẫn cột hợp lệ đã sinh ra trước đó.
    """
    table = Table(
        name="wide",
        columns=(Column("a", "int"), Column("b", "int"), Column("secret", "text")),
    )
    samples = [{"a": 1, "b": 2, "secret": "KHÔNG ĐƯỢC LỘ RA"}]

    prompt = build_annotation_prompt(table, samples, (table.columns[0], table.columns[1]))

    assert "KHÔNG ĐƯỢC LỘ RA" not in prompt
    assert '"secret"' not in prompt


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


# -- Chu Han lot vao tieng Viet -> ha xuong low ----------------------------
#
# Qwen huan luyen chu yeu tren tieng Trung va thinh thoang ro token Han vao
# giua mot tu tieng Viet. Do duoc tren BIRD formula_1: 10/89 muc dinh (11%).
# Chuoi duoi day la mau THAT lay tu lan chay do.
#
# Mo ta kieu nay qua duoc moi kiem tra khac -- co text, parse duoc, model tu
# nhan "high" -- nen neu khong bat thi no vao thang prompt sinh SQL va analyst
# khong bao gio thay no (pending_review chi lay muc "low").

_HONG = "Bảng chứa thông tin về các tr阿森 đua xe F1."


def test_a_description_with_han_characters_is_downgraded_to_low():
    reply = f'{{"table": {{"text": "{_HONG}", "confidence": "high"}}, "columns": {{}}}}'
    ann, failures = annotate_schema(TABLES, SAMPLES, lambda s, u: reply)  # noqa: ARG005

    entry = ann.tables["orders"]
    assert entry.confidence == "low", "model tu nhan high, nhung chuoi nay hong"
    assert entry.text == _HONG, "giu nguyen text -- nguoi sua mot tu nhanh hon viet lai"
    assert failures == 0, "van co chu giai, chi la can nguoi xem -- khong phai that bai"


def test_a_contaminated_entry_reaches_the_review_queue():
    """Day moi la diem cua ban sua: no phai HIEN RA cho analyst."""
    from perception.annotations import pending_review

    reply = f'{{"table": {{"text": "{_HONG}", "confidence": "high"}}, "columns": {{}}}}'
    ann, _ = annotate_schema(TABLES, SAMPLES, lambda s, u: reply)  # noqa: ARG005

    assert ("orders", None) in pending_review(ann)


def test_a_contaminated_column_description_is_downgraded_too():
    reply = (
        '{"table": {"text": "Đơn hàng", "confidence": "high"}, '
        f'"columns": {{"flg_tt": {{"text": "{_HONG}", "confidence": "high"}}}}}}'
    )
    ann, _ = annotate_schema(TABLES, SAMPLES, lambda s, u: reply)  # noqa: ARG005

    assert ann.columns["orders"]["flg_tt"].confidence == "low"


def test_clean_vietnamese_keeps_the_confidence_the_model_reported():
    """Khong duoc ha oan moi thu -- day la ly do bo loc phai hep."""
    reply = (
        '{"table": {"text": "Bảng đơn hàng bán lẻ, mỗi dòng một đơn.", '
        '"confidence": "high"}, "columns": {}}'
    )
    ann, _ = annotate_schema(TABLES, SAMPLES, lambda s, u: reply)  # noqa: ARG005

    assert ann.tables["orders"].confidence == "high"


def test_vietnamese_diacritics_are_not_mistaken_for_cjk():
    """Dau tieng Viet nam trong Latin Extended, khong dinh dai CJK."""
    reply = (
        '{"table": {"text": "Đơn hàng — ướm thử, ễnh ương, quỹ, ngoằn ngoèo.", '
        '"confidence": "high"}, "columns": {}}'
    )
    ann, _ = annotate_schema(TABLES, SAMPLES, lambda s, u: reply)  # noqa: ARG005

    assert ann.tables["orders"].confidence == "high"


# --- Wide-table chunking ----------------------------------------------------
#
# `_local_invoke`'s output budget (`AGENT_MAX_TOKENS["annotate"]`) is a fixed
# number of tokens; a wide table's full column list does not fit, so the
# reply gets cut off mid-JSON and the table is silently counted as a
# failure. The fix batches a table's columns across several `invoke` calls
# and merges the results. These tests force a tiny budget via monkeypatch so
# a small fixture table (not a 44/115-column fixture) already needs more
# than one batch -- see `perception.annotate._column_batches` for the exact
# arithmetic this mirrors.

def _wide_table(name: str, n: int) -> Table:
    return Table(
        name=name,
        columns=tuple(Column(f"col{i}", "text") for i in range(n)),
        primary_key=("col0",),
    )


def test_a_narrow_table_still_makes_exactly_one_call():
    """Most real tables are narrow; chunking must not touch that path."""
    calls = []

    def _invoke(system, user):
        calls.append(user)
        return json.dumps({"table": {"text": "Bảng hẹp", "confidence": "high"}, "columns": {}})

    table = _wide_table("narrow", 3)
    ann, failures = annotate_schema((table,), {"narrow": []}, _invoke)

    assert len(calls) == 1
    assert ann.tables["narrow"].text == "Bảng hẹp"
    assert failures == 0


def test_a_wide_table_is_split_and_every_column_appears_once(monkeypatch):
    # budget=200 -> entries_per_call = floor(200*0.75/45) = 3
    # -> first batch 2 columns, later batches 3 columns each.
    monkeypatch.setitem(model_config.AGENT_MAX_TOKENS, "annotate", 200)

    table = _wide_table("wide", 7)
    calls = []

    def _invoke(system, user):  # noqa: ARG001
        calls.append(user)
        present = [c.name for c in table.columns if c.name in user]
        cols = {name: {"text": f"mô tả {name}", "confidence": "high"} for name in present}
        payload = {"table": {"text": "Bảng rộng", "confidence": "high"}, "columns": cols}
        return json.dumps(payload, ensure_ascii=False)

    ann, failures = annotate_schema((table,), {"wide": []}, _invoke)

    assert len(calls) > 1, "bảng 7 cột với ngân sách nhỏ phải cần nhiều hơn một lượt gọi"
    got = set(ann.columns.get("wide", {}))
    assert got == {c.name for c in table.columns}
    assert failures == 0
    assert ann.tables["wide"].text == "Bảng rộng"


def test_one_batch_failing_does_not_discard_the_other_batches_columns(monkeypatch):
    monkeypatch.setitem(model_config.AGENT_MAX_TOKENS, "annotate", 200)

    table = _wide_table("wide", 7)

    def _invoke(system, user):  # noqa: ARG001
        # The batch carrying col5/col6 (the last batch) fails outright --
        # simulates Ollama down/timeout for just that batch.
        if "col5" in user or "col6" in user:
            raise ConnectionError("connection refused")
        present = [c.name for c in table.columns if c.name in user]
        cols = {name: {"text": f"mô tả {name}", "confidence": "high"} for name in present}
        payload = {"table": {"text": "Bảng rộng", "confidence": "high"}, "columns": cols}
        return json.dumps(payload, ensure_ascii=False)

    ann, failures = annotate_schema((table,), {"wide": []}, _invoke)

    got = set(ann.columns.get("wide", {}))
    assert got == {"col0", "col1", "col2", "col3", "col4"}
    assert "col5" not in got and "col6" not in got
    # The table description rode along with the first batch, which did not
    # fail -- the table itself is not counted as a failure just because one
    # of its later batches was.
    assert ann.tables["wide"].text == "Bảng rộng"
    assert failures == 0


def test_the_table_description_appears_once_not_duplicated_per_batch(monkeypatch):
    monkeypatch.setitem(model_config.AGENT_MAX_TOKENS, "annotate", 200)

    table = _wide_table("wide", 7)
    call_count = {"n": 0}

    def _invoke(system, user):  # noqa: ARG001
        call_count["n"] += 1
        present = [c.name for c in table.columns if c.name in user]
        cols = {name: {"text": f"mô tả {name}", "confidence": "high"} for name in present}
        # A misbehaving model that emits a "table" key on every batch, each
        # with different text -- only the FIRST one must be kept.
        payload = {
            "table": {"text": f"lượt {call_count['n']}", "confidence": "high"},
            "columns": cols,
        }
        return json.dumps(payload, ensure_ascii=False)

    ann, _ = annotate_schema((table,), {"wide": []}, _invoke)

    assert call_count["n"] > 1
    assert ann.tables["wide"].text == "lượt 1"


# -- Ten cot khong mang thong tin -> luon low -------------------------------
#
# Do tren BIRD financial.district, bang co cot A2..A16: model bia ra
# "A11 = Ma xa" (that ra la LUONG TRUNG BINH), "A12 = Ty le dan so" (that ra
# la TY LE THAT NGHIEP 1995), va tu cham "high" cho 7 trong 10 cot. Dung o
# cho de (A9 = So dan) va sai o cho kho -- nguoc dung chieu voi thu can.
#
# Khong phai model kem: voi A11 thi thong tin KHONG TON TAI trong dau vao.
# Khong o ten, khong o kieu, va gia tri mau la so nen khong phan biet duoc
# luong voi ma xa. Buoc phai doan, nen confidence cua chinh no vo dung o
# dung nhung cot can nhat.


def test_a_code_like_column_name_is_always_low_confidence():
    """`A11` -- ca da do duoc, model tu cham high va bia ra y nghia."""
    reply = (
        '{"table": {"text": "Quận huyện. ~ district", "confidence": "high"}, '
        '"columns": {"A11": {"text": "Mã xã. ~ ward code", "confidence": "high"}}}'
    )
    tables = (
        Table(
            name="district",
            columns=(Column("A11", "integer"), Column("name", "text")),
            primary_key=("A11",),
        ),
    )
    ann, _ = annotate_schema(tables, {"district": []}, lambda s, u: reply)  # noqa: ARG005

    assert ann.columns["district"]["A11"].confidence == "low", (
        "model tu cham high, nhung ten A11 khong the suy ra duoc gi"
    )


def test_a_vowelless_abbreviation_is_low_confidence():
    """`CPK`, `CRP`, `B365D` -- ma xet nghiem va keo ca cuoc trong BIRD."""
    from perception.annotate import _name_is_opaque

    for ten in ("CPK", "CRP", "B365D", "BSH", "GBD"):
        assert _name_is_opaque(ten), f"{ten} phai bi coi la mu mo"


def test_a_meaningful_name_keeps_the_model_confidence():
    """Bo loc phai HEP -- neu khong thi ca 873 muc vao hang doi va vo dung."""
    from perception.annotate import _name_is_opaque

    for ten in ("customer_id", "invoice", "birth_date", "total", "email", "region"):
        assert not _name_is_opaque(ten), f"{ten} co nghia, khong duoc ha oan"


def test_an_opaque_column_reaches_the_review_queue():
    """Diem cua ban sua: dua cot do RA TRUOC MAT analyst."""
    from perception.annotations import pending_review

    reply = (
        '{"table": {"text": "Quận huyện. ~ district", "confidence": "high"}, '
        '"columns": {"A11": {"text": "Mã xã. ~ ward code", "confidence": "high"}}}'
    )
    tables = (
        Table(name="district", columns=(Column("A11", "integer"),), primary_key=("A11",)),
    )
    ann, _ = annotate_schema(tables, {"district": []}, lambda s, u: reply)  # noqa: ARG005

    assert ("district", "A11") in pending_review(ann)


def test_an_opaque_table_name_is_low_too():
    reply = '{"table": {"text": "Bảng gì đó. ~ table", "confidence": "high"}, "columns": {}}'
    tables = (Table(name="T1", columns=(Column("x", "integer"),)),)
    ann, _ = annotate_schema(tables, {"T1": []}, lambda s, u: reply)  # noqa: ARG005

    assert ann.tables["T1"].confidence == "low"
