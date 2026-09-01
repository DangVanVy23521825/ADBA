"""Test bộ dịch SQL vàng SQLite -> Postgres.

Thuần tuý: schema đến từ `Table`, không cần database. Mỗi ca ở đây tương
ứng một họ lỗi đã đo được trên 1.534 câu BIRD thật.
"""

import pytest

from eval.sqlite_dialect import SchemaNames, backticks_to_double_quotes, translate
from perception.schema_model import Column, Table

# Sao lại đúng hình dạng gây rắc rối trong schema BIRD gộp:
#   - `CDSCode` giữ hoa thường -> tham chiếu trần bị Postgres gấp sai
#   - `zip` và `Zip` cùng tồn tại ở hai bảng -> nhập nhằng khi không định tính
#   - có cột tên `Count`, trùng tên hàm tổng hợp
TABLES = (
    Table(name="schools", columns=(
        Column("CDSCode", "text"), Column("Zip", "text"), Column("City", "text"),
    )),
    Table(name="member", columns=(
        Column("member_id", "integer"), Column("zip", "text"),
    )),
    Table(name="frpm", columns=(
        Column("CDSCode", "text"), Column("Count", "integer"),
        Column("Free Meal Count (K-12)", "real"),
    )),
)
NAMES = SchemaNames(TABLES)


def t(sql: str) -> str:
    return translate(sql, NAMES)


# --- Dấu huyền ---


def test_backtick_identifiers_become_double_quoted():
    assert backticks_to_double_quotes("SELECT `a b` FROM t") == 'SELECT "a b" FROM t'


def test_a_backtick_inside_a_string_literal_is_left_alone():
    """Dấu huyền có thể là dữ liệu thật. Đổi nó vừa hỏng chuỗi vừa đẻ ra
    một định danh không tồn tại."""
    sql = "SELECT x FROM t WHERE name = 'do`nt touch'"
    assert backticks_to_double_quotes(sql) == sql


def test_a_backticked_name_with_spaces_survives_translation():
    out = t("SELECT `Free Meal Count (K-12)` FROM frpm")
    assert '"Free Meal Count (K-12)"' in out
    assert "`" not in out


# --- Hoa thường ---


def test_an_unquoted_mixed_case_column_gets_quoted():
    """`CDSCode` không nháy bị Postgres gấp thành `cdscode` và không tìm
    thấy. Đây là họ lỗi lớn nhất: 588/1534 câu BIRD."""
    assert '"CDSCode"' in t("SELECT CDSCode FROM schools")


def test_an_all_lowercase_column_is_left_alone():
    """Không nháy thì Postgres gấp về chữ thường, mà tên thật đã là chữ
    thường — thêm nháy chỉ làm SQL rối mà không đổi nghĩa."""
    out = t("SELECT member_id FROM member")
    assert '"member_id"' not in out


def test_an_already_quoted_name_is_not_double_quoted_again():
    out = t('SELECT "CDSCode" FROM schools')
    assert '""' not in out


# --- Lời gọi hàm ---


def test_an_aggregate_call_is_not_mistaken_for_the_column_of_the_same_name():
    """Schema có cột `Count`. Bản đầu của bộ dịch bọc nháy nó thành
    `"Count"(...)`, khiến Postgres đi tìm một HÀM tên đúng như vậy — làm
    hỏng 347 câu vốn chạy tốt."""
    out = t("SELECT COUNT(CDSCode) FROM schools")
    assert '"Count"(' not in out
    assert '"CDSCode"' in out


@pytest.mark.parametrize(
    "sql", ["SELECT `Count` FROM frpm", "SELECT T1.Count FROM frpm AS T1"]
)
def test_the_column_named_count_is_quoted_in_the_forms_that_can_be_told_apart(sql):
    assert '"Count"' in t(sql)


def test_a_bare_column_named_like_a_function_is_a_known_limitation():
    """`Count` trần được sqlparse phân loại là Keyword, không phải Name, nên
    bộ dịch không chạm tới và câu đó sẽ hỏng ở tầng execute.

    KHÔNG sửa bằng cách coi Keyword là cột tiềm năng: như thế sẽ bọc nháy
    cả từ khoá thật, và hỏng nhiều hơn hẳn số cứu được. Trên 1.534 câu BIRD
    không có ca nào viết dạng này — chúng đều dùng dấu huyền hoặc định
    tính, hai dạng ở test trên đều xử lý đúng.

    Khoá lại để nếu ai đó "sửa" nó thì thấy ngay đây là đánh đổi có chủ ý.
    """
    assert t("SELECT Count FROM frpm") == "SELECT Count FROM frpm"


# --- Nhập nhằng và bí danh ---


def test_an_ambiguous_bare_name_is_left_alone():
    """`zip` (member) và `Zip` (schools) cùng tồn tại. Đoán bừa thì sửa
    được bảng này và phá bảng kia, nên thà để nguyên."""
    out = t("SELECT zip FROM member")
    assert '"Zip"' not in out


@pytest.mark.parametrize(
    "sql, want",
    [
        ("SELECT T2.zip FROM schools AS T2", '"Zip"'),
        ("SELECT T2.zip FROM schools T2", '"Zip"'),        # không có AS
        ("SELECT schools.zip FROM schools", '"Zip"'),      # tên bảng làm định tính
    ],
)
def test_a_qualified_ambiguous_name_is_resolved_through_the_alias(sql, want):
    """Biết bảng thì hết nhập nhằng. Đây là thứ đưa BIRD từ 62% lên 78%."""
    assert want in t(sql)


def test_the_same_bare_name_resolves_differently_under_two_aliases():
    out = t(
        "SELECT T1.zip, T2.zip FROM member AS T1 "
        "JOIN schools AS T2 ON T1.member_id = T2.CDSCode"
    )
    assert 'T1.zip' in out          # member.zip là chữ thường, giữ nguyên
    assert 'T2."Zip"' in out        # schools.Zip phải được bọc


def test_an_alias_is_never_itself_quoted():
    out = t("SELECT T1.CDSCode FROM schools AS T1")
    assert '"T1"' not in out


def test_a_keyword_after_the_table_name_is_not_taken_as_an_alias():
    """`FROM schools WHERE ...` — `WHERE` không phải bí danh. Nhận nhầm thì
    bảng thật mất bí danh và mọi cột có định tính của nó hỏng theo."""
    out = t("SELECT CDSCode FROM schools WHERE City = 'x'")
    assert '"CDSCode"' in out
    assert '"City"' in out


def test_a_join_without_aliases_still_resolves(sql=None):
    out = t("SELECT schools.Zip FROM schools JOIN frpm ON schools.CDSCode = frpm.CDSCode")
    assert out.count('"CDSCode"') == 2


# --- Điều bộ dịch KHÔNG hứa ---


def test_a_sqlite_only_function_passes_through_untouched():
    """Bộ dịch chỉ sửa ĐỊNH DANH. `strftime` vẫn hỏng, và hỏng ở tầng
    execute rồi vào bucket `gold_error` — đúng chỗ, vì đó là lỗi dữ liệu
    chứ không phải lỗi hệ thống."""
    assert "strftime" in t("SELECT strftime('%Y', d) FROM member")


def test_translation_is_idempotent():
    """Dịch hai lần không được đẻ ra nháy lồng nháy."""
    once = t("SELECT T1.CDSCode FROM schools AS T1")
    assert translate(once, NAMES) == once


def test_a_string_literal_matching_a_column_name_is_not_quoted():
    """`'CDSCode'` là dữ liệu, không phải định danh."""
    out = t("SELECT 1 FROM schools WHERE City = 'CDSCode'")
    assert "'CDSCode'" in out
