"""Dựng lát schema đưa vào prompt cho một câu hỏi cụ thể.

Hàm thuần: không gọi LLM, không chạm database, không có kiểu lỗi cần retry.
Vì thế nó KHÔNG phải node LangGraph — biến nó thành node là mua thêm một
hop và một nhánh lỗi để đổi lấy không gì cả (spec mục 10.4). Nó chạy đúng
một lần mỗi câu hỏi, trước make_initial_state().
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from perception.connection_profile import ConnectionProfile
from perception.render_schema import render_schema
from perception.retrieval import Retriever, expand_by_foreign_keys


@dataclass(frozen=True)
class SchemaContext:
    """Nội dung prompt cho một câu hỏi. KHÔNG mang thông tin quyền.

    Cố ý không có trường `allowed_tables`: nếu có, ai đó sẽ truyền nó xuống
    execute_sql và biến guard thành thứ do bên gọi tự khai (spec 3.4.1).
    """

    retrieved_tables: tuple[str, ...] = ()
    rendered_text: str = ""
    few_shots: tuple[dict, ...] = field(default_factory=tuple)


def resolve_schema_context(
    profile: ConnectionProfile,
    question: str,
    permitted: frozenset[str],
    retriever: Retriever | None = None,
    k: int = 8,
    must_include: Sequence[str] = (),
) -> SchemaContext:
    """Chọn bảng và kết xuất text schema cho một câu hỏi.

    Args:
        profile: hồ sơ kết nối, mang schema và công tắc schema_mode.
        question: câu hỏi của người dùng.
        permitted: tập bảng người dùng được phép — dẫn ra ĐỘC LẬP bởi
            connection_profile.permitted_tables(). Mọi đường ra khỏi hàm
            này đều bị chặn bởi tập đó, kể cả must_include.
        retriever: bắt buộc khi profile.schema_mode == "retrieval".
        k: số bảng lấy trước khi mở rộng theo khóa ngoại.
        must_include: bảng buộc phải có mặt. Dùng khi reflector báo
            schema_mismatch (spec mục 5.2). Không vượt được `permitted`.

    Raises:
        ValueError: khi schema_mode là "retrieval" mà không có retriever.
    """
    if not permitted:
        return SchemaContext()

    if profile.schema_mode == "full":
        chosen = permitted
    else:
        if retriever is None:
            raise ValueError(
                "schema_mode='retrieval' cần một retriever. "
                "Dùng FullRetriever nếu muốn hành vi của chế độ full."
            )
        hits = retriever.search(question, k=k, candidates=permitted)
        chosen = expand_by_foreign_keys(hits, profile.tables) | set(must_include)
        # Ranh giới quyền THẬT SỰ vẫn nằm ở đây, không phải ở `candidates`
        # phía trên. FK expansion và must_include có thể kéo vào bảng ngoài
        # `permitted` sau khi retriever đã chọn xong — nếu bỏ dòng này, dữ
        # liệu ngoài quyền sẽ lọt vào rendered_text. `candidates` chỉ tối ưu
        # độ liên quan (tránh top-K bị lấp đầy bởi bảng người dùng không
        # thấy), nó không thay thế được phép giao cuối cùng này.
        chosen &= permitted

    by_name = profile.by_name()
    # Giữ thứ tự theo profile, không theo điểm số: đầu ra phải ổn định
    # giữa các câu hỏi thì prefix caching mới dùng được.
    ordered = tuple(t.name for t in profile.tables if t.name in chosen)

    return SchemaContext(
        retrieved_tables=ordered,
        rendered_text=render_schema([by_name[n] for n in ordered]),
        few_shots=(),
    )
