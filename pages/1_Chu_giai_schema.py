"""Trang duyệt chú giải schema — dành cho analyst nghiệp vụ, không phải kỹ sư."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from perception.annotations import load_annotations, save_annotations
from perception.profile_store import SCHEMA_YAML, STRUCTURE_JSON, structure_from_plain
from perception.review_state import (
    apply_edit,
    filter_rows,
    review_progress,
    review_rows,
    widget_key,
)

st.set_page_config(page_title="Chú giải schema", page_icon="📝", layout="wide")
st.title("📝 Chú giải schema")
st.caption(
    "Mô tả bảng và cột bằng ngôn ngữ nghiệp vụ. Chất lượng ở đây quyết định "
    "hệ thống trả lời đúng tới đâu — quan trọng hơn mọi thiết lập kỹ thuật."
)

profile_dir = Path(os.environ.get("ADBA_PROFILE_DIR", "profile"))

if not (profile_dir / STRUCTURE_JSON).exists():
    st.warning(
        f"Chưa có `{STRUCTURE_JSON}` trong `{profile_dir}`. "
        "Chạy `python onboard.py extract --dsn ...` trước."
    )
    st.stop()

tables = structure_from_plain(
    json.loads((profile_dir / STRUCTURE_JSON).read_text(encoding="utf-8"))
)
ann = load_annotations(profile_dir / SCHEMA_YAML)

done, total = review_progress(ann, tables=tables)
st.progress(done / total if total else 0.0, text=f"Đã duyệt {done}/{total} mục")

only_pending = st.checkbox("Chỉ hiện mục cần duyệt", value=True)
rows, hidden_confident = filter_rows(review_rows(tables, ann), only_pending)

if hidden_confident:
    st.info(
        f"Đang ẩn {hidden_confident} mục model tự tin — bỏ chọn "
        '"Chỉ hiện mục cần duyệt" để xem lại. Một chú giải sai mà model tự '
        "nhận là chắc chắn thì không ai kiểm lại, nên nó vẫn là rủi ro."
    )

if not rows:
    # Không nói "đã duyệt xong" khi vẫn còn mục bị bộ lọc giấu đi — đó là
    # đúng cái hiểu nhầm khiến người duyệt dừng lại quá sớm.
    if hidden_confident:
        st.warning(
            "Không còn mục nào model tự nhận là không chắc, nhưng vẫn còn "
            f"{hidden_confident} mục chưa ai duyệt. Bỏ chọn bộ lọc để xem."
        )
    else:
        st.success("Đã duyệt hết mọi mục.")
    st.stop()

by_table: dict[str, list] = {}
for r in rows:
    by_table.setdefault(r.table, []).append(r)

for table_name, table_rows in by_table.items():
    with st.expander(f"**{table_name}**  ({len(table_rows)} mục)", expanded=True):
        for r in table_rows:
            label = f"Bảng `{r.table}`" if r.column is None else f"Cột `{r.column}` — {r.type_hint}"
            key = widget_key(r.table, r.column)
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                new = st.text_input(label, value=r.current_text, key=key)
            with col_btn:
                st.write("")
                if st.button("Lưu", key=f"btn::{key}"):
                    ann = apply_edit(ann, r.table, r.column, new)
                    save_annotations(ann, profile_dir / SCHEMA_YAML)
                    st.rerun()
            if r.confidence == "low" and r.reviewed_by != "human":
                st.caption("⚠️ Model không chắc về mục này")
