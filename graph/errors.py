"""Nhãn lỗi — một nguồn sự thật cho `last_error["error_type"]`.

Reflector chẩn đoán theo nhãn, nên nhãn sai dẫn tới `corrected_context`
sai, và lần thử lại hỏng đúng như lần đầu. Vì thế tập này là đóng.

SAI LỆCH CÓ CHỦ Ý so với spec 2026-08-12 mục 5.6: spec liệt kê 9 nhãn
nhưng không có nhãn nào cho lỗi lập kế hoạch của supervisor, lỗi sinh
insight, bước vắng mặt khỏi kế hoạch, hay dữ liệu đầu vào thiếu — trong
khi code đã và đang sinh ra cả bốn lớp lỗi đó. Ép chúng vào 9 nhãn kia là
dán nhãn sai. Nên tập ở đây là 9 + 4.

"Tập đóng" chỉ có nghĩa khi mọi nơi sinh lỗi đều DÙNG tập này. Trước bản
sửa này thì không: mười chỗ đặt `error_type` trong bốn file agent còn
viết chuỗi trần, và hai chuỗi trong số đó (`"missing_step"`,
`"missing_data"`) thậm chí không phải thành viên của `ERROR_LABELS` —
tức là tính chất "một nguồn sự thật" đã hỏng sẵn. `tests/unit/test_errors.py`
canh cho điều đó không tái diễn.
"""

from __future__ import annotations

# Chín nhãn của spec 5.6.
SQL_GENERATION = "sql_generation"
SQL_EXECUTION = "sql_execution"
SQL_TIMEOUT = "sql_timeout"
PYTHON_RUNTIME = "python_runtime"
CHART_ERROR = "chart_error"
SCHEMA_MISMATCH = "schema_mismatch"
MODEL_TIMEOUT = "model_timeout"
BUDGET_EXCEEDED = "budget_exceeded"
TOOL_UNAVAILABLE = "tool_unavailable"

# Bốn nhãn thêm — xem docstring module.
PLANNING_FAILURE = "planning_failure"
INSIGHT_GENERATION = "insight_generation"

# Node được định tuyến tới nhưng kế hoạch không có bước nào cho nó. Đây là
# lỗi định tuyến/lập kế hoạch, không phải lỗi công cụ — nhét nó vào
# `tool_unavailable` sẽ khiến reflector đi sửa nhầm chỗ.
MISSING_STEP = "missing_step"

# Node chạy đúng lúc nhưng đầu vào nó cần chưa có (vd. `shared_dataframe`
# rỗng vì bước sql chưa chạy). Khác `missing_step`: kế hoạch đúng, thứ tự
# hoặc bước phía trước mới là chỗ hỏng.
MISSING_DATA = "missing_data"

ERROR_LABELS = frozenset({
    SQL_GENERATION, SQL_EXECUTION, SQL_TIMEOUT, PYTHON_RUNTIME, CHART_ERROR,
    SCHEMA_MISMATCH, MODEL_TIMEOUT, BUDGET_EXCEEDED, TOOL_UNAVAILABLE,
    PLANNING_FAILURE, INSIGHT_GENERATION, MISSING_STEP, MISSING_DATA,
})
