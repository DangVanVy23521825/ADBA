"""Nhãn lỗi — một nguồn sự thật cho `last_error["error_type"]`.

Reflector chẩn đoán theo nhãn, nên nhãn sai dẫn tới `corrected_context`
sai, và lần thử lại hỏng đúng như lần đầu. Vì thế tập này là đóng.

SAI LỆCH CÓ CHỦ Ý so với spec 2026-08-12 mục 5.6: spec liệt kê 9 nhãn
nhưng không có nhãn nào cho lỗi lập kế hoạch của supervisor hay lỗi sinh
insight — trong khi code đã và đang sinh ra cả hai lớp lỗi đó. Ép chúng
vào 9 nhãn kia là dán nhãn sai. Nên tập ở đây là 9 + 2.
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

# Hai nhãn thêm — xem docstring module.
PLANNING_FAILURE = "planning_failure"
INSIGHT_GENERATION = "insight_generation"

ERROR_LABELS = frozenset({
    SQL_GENERATION, SQL_EXECUTION, SQL_TIMEOUT, PYTHON_RUNTIME, CHART_ERROR,
    SCHEMA_MISMATCH, MODEL_TIMEOUT, BUDGET_EXCEEDED, TOOL_UNAVAILABLE,
    PLANNING_FAILURE, INSIGHT_GENERATION,
})
