"""Sinh chú giải ngữ nghĩa cho từng bảng bằng model LOCAL.

Ràng buộc tuyệt đối: model local, không gọi API ngoài. Schema của khách
không được rời khỏi mạng của họ ở bước đầu tiên của quá trình cài đặt —
đó là thứ duy nhất khiến họ chọn on-prem.

Mọi thứ sinh ra ở đây đều mang `reviewed_by="llm"`. Chỉ giao diện duyệt mới
được đặt `"human"`. Nhầm chỗ này sẽ khiến chú giải máy đoán được coi là đã
xác nhận, và không ai nhìn lại nó nữa.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence

from model.model_config import AGENT_MAX_TOKENS
from perception.annotations import LLM, Annotation, SchemaAnnotations
from perception.schema_model import Column, Table

_ALLOWED_CONFIDENCE = {"high", "low"}
_FENCE = re.compile(r"```(?:json)?\s*|\s*```")

# Chỉ những dấu `{` mở ra một object JSON mới đáng thử: sau nó phải là một
# khoá dạng chuỗi, hoặc object rỗng. Dấu ngoặc trong văn xuôi (`{key:val}`)
# bị loại ngay, nên vòng quét không tốn một lần `raw_decode` thất bại cho
# mỗi dấu ngoặc trong câu trả lời.
_OBJECT_START = re.compile(r'\{\s*(?:"|\})')

# Chặn trên số ứng viên. `json.JSONDecodeError` tính lineno/colno bằng cách
# quét lại toàn bộ văn bản, nên N lần thử hỏng là O(N × độ dài). Model local
# lặp vô hạn một ký tự là chuyện có thật; không có trần thì một câu trả lời
# rác treo luôn bảng đó.
_MAX_PARSE_CANDIDATES = 32
_MAX_CELL_LEN = 200

_RULES = """\
Quy tắc:
- Viết bằng tiếng Việt, mỗi mô tả một câu ngắn.
- BẮT BUỘC: mỗi `text` gồm câu tiếng Việt, dấu `~`, rồi 2-5 từ khoá tiếng
  Anh. Từ khoá là thuật ngữ NGHIỆP VỤ người ta dùng để hỏi, không phải dịch
  máy móc tên cột — bảng `invoice` phải có `revenue`, vì người ta hỏi doanh
  thu chứ không hỏi hoá đơn. Thiếu phần `~` là câu trả lời SAI.
- confidence = "low" khi bạn phải đoán. Tên viết tắt không rõ nghĩa, cột cờ
  không có ngữ cảnh, mã nội bộ — đều là "low".
- Thà nhận không chắc còn hơn đoán nghe hợp lý. Một mô tả sai được đánh dấu
  "high" sẽ không ai kiểm lại, và mọi truy vấn dựa vào nó sẽ sai âm thầm.
- Chỉ mô tả cột nào bạn thực sự suy ra được; bỏ qua cột hiển nhiên như id.
- CHỈ trả về đúng những cột liệt kê trong mục "Cột:" ở trên — không thêm
  cột nào khác vào "columns" dù bạn đoán được ý nghĩa của nó. Một bảng
  rộng có thể được hỏi qua NHIỀU lượt riêng biệt, mỗi lượt chỉ mang một
  phần cột; tự ý mô tả thêm sẽ đẩy câu trả lời vượt giới hạn độ dài và bị
  cắt cụt giữa chừng, hỏng luôn cả những cột đúng lẽ ra phải trả lời được.
"""

SYSTEM_PROMPT = """\
Bạn mô tả ý nghĩa nghiệp vụ của bảng và cột trong một cơ sở dữ liệu.

Trả về DUY NHẤT một object JSON, không kèm giải thích:

{"table": {"text": "<mô tả tiếng Việt> ~ <từ khoá tiếng Anh>", "confidence": "high|low"},
 "columns": {"<tên cột>": {"text": "<mô tả tiếng Việt> ~ <từ khoá tiếng Anh>",
                           "confidence": "high|low"}}}

Ví dụ ĐÚNG, cho một bảng giả định tên `tram_quan_trac`:

{"table": {"text": "Trạm quan trắc môi trường, mỗi dòng một trạm. ~ monitoring station, sensor site, measurement point",
           "confidence": "high"},
 "columns": {"do_am": {"text": "Độ ẩm không khí đo được. ~ humidity, moisture, air humidity",
                       "confidence": "high"}}}

Ví dụ trên chỉ để minh hoạ ĐỊNH DẠNG. Đừng chép lại nội dung của nó; hãy mô
tả đúng bảng và cột đang được đưa cho bạn.

""" + _RULES

# Dùng cho các lô cột SAU lô đầu tiên của một bảng rộng (xem
# `_column_batches`/`annotate_schema`). Mô tả bảng đã được sinh từ lô đầu
# rồi — hỏi lại "table" ở đây chỉ tốn token đầu ra một cách vô ích, đúng thứ
# ngân sách này đang phải tiết kiệm. Phần "Quy tắc" giữ NGUYÊN VĂN với
# `SYSTEM_PROMPT` (biến `_RULES` dùng chung) để chất lượng mô tả cột không
# đổi tuỳ theo cột đó rơi vào lô nào.
SYSTEM_PROMPT_COLUMNS_ONLY = """\
Bạn mô tả ý nghĩa nghiệp vụ của MỘT NHÓM CỘT thuộc một bảng cơ sở dữ liệu.
Bảng này đã được mô tả ở một lượt gọi khác — lượt này CHỈ mô tả cột, ĐỪNG
lặp lại mô tả bảng.

Trả về DUY NHẤT một object JSON, không kèm giải thích:

{"columns": {"<tên cột>": {"text": "<mô tả tiếng Việt> ~ <từ khoá tiếng Anh>",
                           "confidence": "high|low"}}}

""" + _RULES


def _truncate(value):
    """Cắt một giá trị cell về tối đa `_MAX_CELL_LEN` ký tự.

    Bảng ERP thật có cột note/xml/json chứa hàng chục KB. Ba dòng mẫu như
    vậy tràn context của một model 7B. Chỉ cắt chuỗi — số, bool, None giữ
    nguyên vì chúng vốn đã ngắn và json.dumps xử lý được trực tiếp.
    """
    if isinstance(value, str) and len(value) > _MAX_CELL_LEN:
        return value[:_MAX_CELL_LEN] + "…"
    return value


def _truncate_row(row: Mapping, keep: frozenset[str] | None = None) -> dict:
    """`keep=None` giữ mọi khoá (hành vi cũ, bảng hẹp/một lô). `keep` khác
    `None` chỉ giữ những khoá có trong đó — xem lý do trong
    `build_annotation_prompt`.
    """
    items = row.items() if keep is None else ((k, v) for k, v in row.items() if k in keep)
    return {k: _truncate(v) for k, v in items}


def build_annotation_prompt(
    table: Table, samples: Sequence[Mapping], columns: Sequence[Column] | None = None
) -> str:
    """Prompt cho một bảng: cấu trúc cộng vài dòng thật.

    Dòng mẫu là thứ phân biệt "cột boolean tên flg_tt" với "cờ đánh dấu đơn
    đã thanh toán" — không có chúng thì model chỉ đoán từ tên.

    `columns`: liệt kê cột nào trong phần "Cột:" của prompt VÀ lọc dòng
    mẫu chỉ còn đúng các khoá đó. Mặc định `None` giữ TOÀN BỘ
    `table.columns`/toàn bộ khoá của dòng mẫu — hành vi cũ, không đổi cho
    bảng hẹp (một lô duy nhất).

    Bảng rộng bị chia lô (`_column_batches`) PHẢI lọc cả hai, không chỉ
    "Cột:". Đo được trên bảng `Match` thật (bird_all, 115 cột): lô đầu chỉ
    xin 7 cột trong "Cột:", nhưng `sample_rows` trả về DÒNG ĐẦY ĐỦ (cả 115
    cột) — dòng mẫu vẫn phơi ra `goal`, `shoton`, `home_player_9`, ... dù
    chúng không nằm trong lô. Model bám vào những khoá đó trong JSON mẫu
    và tự ý mô tả thêm cột NGOÀI lô, đẩy số mục vượt xa cỡ lô đã tính toán
    (7 mục) — output vượt ngân sách và bị Ollama cắt ngang giữa chừng,
    đúng lỗi gốc mà chunking phải sửa, chỉ khác là cắt ở nội dung sample
    thay vì ở "Cột:". Hậu quả cụ thể: JSON không bao giờ đóng, `_parse` trả
    `{}`, mô tả bảng (đi kèm lô này) mất trắng dù cả 7 cột lẫn "table" đều
    ĐÃ xuất hiện đầy đủ trong phần output kịp sinh ra trước khi bị cắt.
    Lọc dòng mẫu đúng theo `columns` của lô đóng luôn cả input lẫn tín hiệu
    khiến model lạc đề, không chỉ input.
    """
    cols_list = table.columns if columns is None else columns
    cols = "\n".join(
        f"  {c.name} {c.data_type}"
        + (" [GENERATED]" if c.is_generated else "")
        + (f" [FK → {table.foreign_keys[c.name]}]" if c.name in table.foreign_keys else "")
        for c in cols_list
    )
    pk = ", ".join(table.primary_key) or "(không có)"
    row_count = "(chưa biết)" if table.row_count is None else table.row_count
    keep = None if columns is None else frozenset(c.name for c in cols_list)
    truncated_rows = [_truncate_row(row, keep) for row in list(samples)[:3]]
    rows = json.dumps(truncated_rows, ensure_ascii=False, default=str, indent=1)
    return (
        f"Bảng: {table.name}\n"
        f"Khoá chính: {pk}\n"
        f"Số dòng (ước lượng): {row_count}\n"
        f"Cột:\n{cols}\n\n"
        f"Vài dòng dữ liệu thật:\n{rows}\n\n"
        f"Hãy mô tả bằng tiếng Việt.\n"
    )


def _parse(raw: str) -> dict:
    """Tách một object JSON khỏi phần văn xuôi có thể bao quanh nó.

    Model local hay trả kiểu "Đây là kết quả: {...} Hy vọng giúp ích." —
    `json.loads(text[start:])` đòi mọi thứ sau dấu `{` đầu tiên phải là JSON
    hợp lệ nên sẽ ném lỗi vì đoạn văn xuôi cuối câu. `raw_decode` chỉ đọc
    đúng một giá trị JSON và bỏ qua phần còn lại.

    Không dừng lại ở dấu `{` đầu tiên trong chuỗi: một câu dẫn kiểu
    "Kết quả (dạng {key:val}): {"table": ...}" có một dấu `{` giả trong
    phần văn xuôi trước JSON thật. Nếu khoá vào dấu đó, `raw_decode` sẽ
    hỏng và cả câu trả lời — vốn chứa JSON hợp lệ — bị vứt bỏ oan. Thay vào
    đó, quét từng dấu `{` từ trái sang phải và dùng kết quả đầu tiên giải
    mã thành công (thành một object, không phải số hay chuỗi); chỉ trả về
    `{}` khi mọi ứng viên đều thất bại.
    """
    text = _FENCE.sub("", raw).strip()
    decoder = json.JSONDecoder()
    for tried, match in enumerate(_OBJECT_START.finditer(text)):
        if tried >= _MAX_PARSE_CANDIDATES:
            break
        try:
            obj, _end = decoder.raw_decode(text, match.start())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


# Hán, Nhật (hiragana/katakana), Hàn. Prompt yêu cầu tiếng Việt, nên ký tự
# thuộc các dải này trong mô tả nghĩa là model vừa trượt sang ngôn ngữ khác.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


# Tên KHÔNG mang thông tin ngữ nghĩa. Ba dạng, đo trên 798 cột của
# bird_all: 61 cột (7%) khớp — đủ nhỏ để hàng đợi duyệt vẫn dùng được.
#
#   `^[A-Za-z]{1,3}\d+$`  A2, A11, col3, f1 — mã thuần, không mang nghĩa
#   không nguyên âm       B365D, CPK, CRP, BSH — viết tắt đã nén hết nguyên âm
#   một chữ cái           x, n — không thể suy ra gì
#
# Cả ba đều là tên mà THÔNG TIN KHÔNG TỒN TẠI trong đầu vào: không ở tên
# cột, không ở kiểu dữ liệu, và giá trị mẫu là số nên không phân biệt được
# "lương trung bình" với "mã xã". Model buộc phải đoán.
_CODE_LIKE = re.compile(r"^[A-Za-z]{1,3}\d+$")


def _name_is_opaque(name: str) -> bool:
    """Tên này có tự nói lên nó là gì không?"""
    if not name:
        return False
    if _CODE_LIKE.match(name):
        return True
    chu = re.sub(r"[^A-Za-z]", "", name).lower()
    if len(chu) == 1:
        return True
    return len(chu) >= 2 and not set(chu) & set("aeiouy")


def _confidence(value, text: str = "", name: str = "") -> str:
    """Độ tự tin của một mục chú giải, hạ xuống `low` khi có dấu hiệu hỏng.

    Model local là Qwen — huấn luyện chủ yếu trên tiếng Trung — và nó thỉnh
    thoảng rò token Hán vào giữa một từ tiếng Việt: `trường` thành `tr阿森`.
    Đo được trên BIRD formula_1: 10/89 mục dính, tức 11%.

    Một mô tả như thế vẫn qua được mọi kiểm tra khác — nó có text, parse
    được, model tự nhận `high` — nên nếu không bắt ở đây thì nó lọt thẳng
    vào prompt sinh SQL, và analyst không bao giờ thấy nó trong hàng đợi
    duyệt (`pending_review` chỉ lấy mục `low`).

    Hạ xuống `low` chứ không loại bỏ: phần còn lại của câu thường vẫn dùng
    được, và người sửa một từ nhanh hơn viết lại từ đầu. Đây cũng là lý do
    không tính nó là thất bại — bảng vẫn có chú giải, chỉ là cần người xem.

    Đánh đổi đã cân nhắc: một mô tả tiếng Việt hợp lệ có chứa chữ Hán —
    bảng về khách hàng Trung Quốc chẳng hạn — sẽ bị hạ oan xuống `low`.
    Hậu quả là nó vào hàng đợi duyệt. Đó là cái giá rẻ, ngược lại thì một
    mô tả hỏng đi thẳng vào prompt mà không ai biết.

    `name` — tên bảng/cột — hạ xuống `low` khi nó không mang thông tin gì
    (`_name_is_opaque`). Đo được trên BIRD `financial.district`, bảng có cột
    `A2`..`A16`: model bịa ra "A11 = Mã xã" (thật ra là LƯƠNG TRUNG BÌNH),
    "A12 = Tỷ lệ dân số" (thật ra là TỶ LỆ THẤT NGHIỆP 1995), và tự chấm
    `high` cho 7 trong 10 cột. Nó đúng ở chỗ dễ (`A9 = Số dân`) và sai ở chỗ
    khó — tức là ngược đúng chiều so với thứ cần.

    Không phải vì model kém: với `A11`, thông tin KHÔNG TỒN TẠI trong đầu
    vào. Không ở tên, không ở kiểu, và giá trị mẫu là số nên không phân biệt
    được lương với mã xã. Buộc phải đoán, nên `confidence` của chính nó
    không dùng được ở đúng những cột cần nhất.

    Bản sửa này không làm model đoán đúng hơn. Nó chỉ đưa 10/10 cột đó vào
    hàng đợi duyệt thay vì 3 — và người là chỗ duy nhất sửa được.
    """
    if name and _name_is_opaque(name):
        return "low"
    if text and _CJK.search(text):
        return "low"
    return value if value in _ALLOWED_CONFIDENCE else "low"


# --- Chia cột của bảng rộng thành nhiều lô ----------------------------------
#
# `_local_invoke` gọi model với `agent_type="annotate"`, một ngân sách token
# ĐẦU RA cố định (`AGENT_MAX_TOKENS["annotate"]`, model/model_config.py).
# Bảng càng nhiều cột, JSON trả lời càng dài — vượt ngân sách thì Ollama cắt
# NGANG câu trả lời (không phải lỗi, không phải chuỗi rỗng: một JSON dở dang
# kết thúc giữa chừng một khoá mới). `_parse` vẫn tìm được MỘT object nhỏ
# hơn, đầy đủ, nằm lọt trong phần dở dang đó và trả về nó — nên trông như
# parse thành công — nhưng object đó thường thiếu khoá "table" hoặc có
# "table" rỗng, và bảng bị tính là thất bại một cách âm thầm. Đo được trên
# BIRD bird_all: 5/75 bảng (21-115 cột) trượt kiểu này ở MỌI lượt chạy.
#
# Input không phải vấn đề (prompt của bảng 115 cột chỉ ~3.1K token, dưới xa
# OLLAMA_NUM_CTX=4096) — chỉ output. Và tăng ngân sách output không giải
# quyết được: thử num_predict=3000 cho một bảng 44 cột bị giết sau 5 phút
# trên một model 7B chạy local. Cách sửa đúng là chia cột bảng rộng thành
# nhiều LÔ, mỗi lô một lượt gọi `invoke` riêng, rồi ghép cột lại — mỗi lượt
# gọi luôn nằm trong ngân sách nên không bao giờ bị cắt, và luôn nhanh vì
# lô nhỏ.
#
# Cỡ lô suy ra từ ngân sách, không hardcode:
#
#   Đo thật qua Ollama (qwen2.5-coder:7b-instruct-q5_K_M), bảng `Laboratory`
#   thật (44 cột, bird_all), một lô 10 cột + 1 mục mô tả bảng: phản hồi tốn
#   483 token đầu ra (`eval_count` của Ollama) cho 11 mục — tức ~44
#   token/mục. Làm tròn lên 45 để chừa biên cho những mục dài hơn trung
#   bình (`_TOKENS_PER_ENTRY_ESTIMATE`).
#
#   Chỉ dùng 75% ngân sách làm hạn mức THẬT SỰ (`_ENTRY_BUDGET_SAFETY`) —
#   phần còn lại chừa cho khung JSON (dấu ngoặc, khoá, thụt lề) và biến
#   động tự nhiên quanh mức trung bình 45 token/mục. Thà lãng phí một lượt
#   gọi thừa (vài giây) còn hơn một lô bị cắt cụt (mất trắng cả lô) — đúng
#   nguyên tắc failure-mode mà toàn bộ hàm `annotate_schema` đã theo.
#
#       entries_per_call = floor(budget × 0.75 / 45)
#       budget = AGENT_MAX_TOKENS["annotate"] = 512
#       → floor(384 / 45) = 8
#
#   Lô ĐẦU TIÊN của một bảng còn phải chừa đúng 1 chỗ cho mục "table" (xem
#   lựa chọn thiết kế trong docstring `annotate_schema`): 8 − 1 = 7 cột. Lô
#   SAU chỉ mang cột, dùng nguyên 8.
#
#   Khi budget hoặc ngôn ngữ mô tả đổi (tiếng Anh sẽ rẻ hơn ~44 token/mục
#   đáng kể — xem module docstring `perception/annotate.py` về chi phí
#   dấu tiếng Việt), tính lại theo ĐÚNG công thức trên — đừng sửa tay hai
#   con số 7/8 mà không tính lại `entries_per_call`.
#
#   Ngưỡng 7 cột này khớp với phân bố thật: 51/75 bảng của bird_all có ≤7
#   cột, nên phần lớn bảng vẫn chỉ tốn đúng MỘT lượt gọi — không có gì đổi
#   so với trước khi có chunking.
_TOKENS_PER_ENTRY_ESTIMATE = 45
_ENTRY_BUDGET_SAFETY = 0.75


def _entries_per_call() -> int:
    budget = AGENT_MAX_TOKENS.get("annotate", 512)
    return max(1, int(budget * _ENTRY_BUDGET_SAFETY // _TOKENS_PER_ENTRY_ESTIMATE))


def _column_batches(columns: Sequence[Column]) -> list[tuple[Column, ...]]:
    """Chia `columns` thành các lô vừa ngân sách token đầu ra của một lượt
    gọi (xem khối chú thích ngay trên). Lô đầu chừa 1 chỗ cho mục "table";
    các lô sau dùng trọn cỡ lô. Bảng hẹp (vừa trong lô đầu) trả về đúng MỘT
    lô — `annotate_schema` nhờ đó vẫn gọi `invoke` đúng một lần cho bảng
    hẹp, không đổi hành vi so với trước khi có chunking.
    """
    per_call = _entries_per_call()
    first_size = max(1, per_call - 1)
    rest_size = max(1, per_call)
    cols = tuple(columns)
    if len(cols) <= first_size:
        return [cols]
    batches = [cols[:first_size]]
    remainder = cols[first_size:]
    for start in range(0, len(remainder), rest_size):
        batches.append(remainder[start : start + rest_size])
    return batches


def annotate_schema(
    tables: Sequence[Table],
    samples: Mapping[str, list[dict]],
    invoke: Callable[[str, str], str],
    *,
    on_table_start: Callable[[Table, int, int], None] | None = None,
) -> tuple[SchemaAnnotations, int]:
    """Chú giải từng bảng một; bảng rộng được chia thành nhiều lô cột.

    Từng bảng chứ không phải cả schema một lượt: 150 bảng không vừa context,
    và một lần trả lời hỏng chỉ làm hỏng một bảng thay vì cả lượt chạy. Điều
    đó chỉ đúng nếu `invoke` ném lỗi (Ollama sập, timeout, hết bộ nhớ) cũng
    được xử lý như một lần trả lời không đọc được — nếu không, một bảng lỗi
    hạ tầng ở giữa lượt chạy sẽ làm mất công 36 bảng đã xong trước đó. Cùng
    nguyên tắc áp dụng ở cấp LÔ bên trong một bảng rộng (xem bên dưới): một
    lô lỗi không được xoá cột của các lô khác thuộc CÙNG bảng đó.

    Một bảng tính là lỗi khi mô tả bảng thu được rỗng — bất kể nguyên nhân
    là `invoke` ném lỗi, câu trả lời không đọc được, hay câu trả lời là JSON
    hợp lệ nhưng rỗng (`{"table": {}, "columns": {}}`). Model local suy
    thoái về JSON hợp lệ-nhưng-rỗng là một kiểu lỗi thật, khác lỗi kết nối
    nhưng cùng hậu quả: bảng đó không được chú giải. Đếm theo văn bản rỗng
    thay vì theo "parse được hay không" mới trung thực với đúng câu hỏi mà
    con số này trả lời cho người vận hành.

    Mô tả BẢNG được sinh đúng MỘT LẦN, đi kèm LÔ ĐẦU TIÊN — không phải một
    lượt gọi riêng, và không phải mỗi lô một lần. Hai lý do: (1) không tốn
    thêm lượt gọi nào — lượt gọi là tài nguyên chậm nhất trong toàn hệ
    thống này; (2) mô tả bảng chỉ cần metadata dùng chung cho MỌI lô (tên
    bảng, khoá chính, số dòng, vài dòng mẫu — `build_annotation_prompt`
    đưa nguyên vẹn vào mọi lô bất kể lô đó mang cột nào), nên tập con cột
    của riêng lô đầu không ảnh hưởng chất lượng mô tả. Nếu chính lượt gọi
    lô đầu thất bại (`invoke` ném lỗi, hoặc trả JSON không có "table"),
    bảng đó bị tính là thất bại đúng theo định nghĩa ở trên — CÁC LÔ SAU
    vẫn được gọi bình thường và cột của chúng vẫn được ghép vào, chỉ riêng
    mô tả bảng là rỗng. Không có cơ chế "thử lại mô tả bảng ở lô sau": lô
    sau dùng `SYSTEM_PROMPT_COLUMNS_ONLY`, vốn không hỏi "table", nên dù
    model có tự ý trả về khoá đó, giá trị của nó bị bỏ qua có chủ đích —
    tránh một model lỡ trả "table" khác nhau ở mỗi lô ghi đè lẫn nhau theo
    thứ tự lô, một hành vi không xác định được trước.

    `on_table_start`, nếu có, được gọi đúng MỘT LẦN cho mỗi bảng — trước lô
    đầu tiên của bảng đó — với `(table, chỉ_số_1_based, tổng_số_bảng)`.
    Đây là chỗ neo đúng cho một bộ đếm tiến độ theo BẢNG (xem
    `onboard._with_progress`): số lượt gọi `invoke` cho một bảng giờ thay
    đổi theo độ rộng của nó, nên một bộ đếm tiến độ dựa trên số lần gọi
    `invoke` (cách làm cũ) chạy vượt quá tổng số bảng ngay khi gặp bảng
    rộng đầu tiên.
    """
    table_ann: dict[str, Annotation] = {}
    column_ann: dict[str, dict[str, Annotation]] = {}
    failures = 0
    total = len(tables)

    for index, table in enumerate(tables, start=1):
        if on_table_start is not None:
            on_table_start(table, index, total)

        table_samples = samples.get(table.name, [])
        known = {c.name for c in table.columns}
        batches = _column_batches(table.columns)

        text = ""
        confidence_raw: object = None
        merged_columns: dict[str, Annotation] = {}

        for batch_index, batch_cols in enumerate(batches):
            is_first = batch_index == 0
            system = SYSTEM_PROMPT if is_first else SYSTEM_PROMPT_COLUMNS_ONLY
            prompt = build_annotation_prompt(table, table_samples, batch_cols)
            try:
                raw = invoke(system, prompt)
            except Exception:
                raw = ""
            parsed = _parse(raw)

            if is_first:
                t = parsed.get("table") or {}
                text = t.get("text", "")
                confidence_raw = t.get("confidence")

            cols = parsed.get("columns") or {}
            for name, body in cols.items():
                if (
                    name in known
                    and name not in merged_columns
                    and isinstance(body, dict)
                    and body.get("text")
                ):
                    merged_columns[name] = Annotation(
                        text=body.get("text", ""),
                        reviewed_by=LLM,
                        confidence=_confidence(
                            body.get("confidence"), body.get("text", ""), name
                        ),
                    )

        table_ann[table.name] = Annotation(
            text=text,
            reviewed_by=LLM,
            confidence=(
                _confidence(confidence_raw, text, table.name) if text else "low"
            ),
        )
        if not text:
            failures += 1

        if merged_columns:
            column_ann[table.name] = merged_columns

    return SchemaAnnotations(tables=table_ann, columns=column_ann), failures
