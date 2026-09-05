"""Nhãn lỗi là một tập ĐÓNG — và tập đó phải khớp với thực tế code phát ra.

Reflector chẩn đoán theo nhãn, nên một nhãn nằm ngoài tập đóng không chỉ
xấu về hình thức: nó đi qua reflector như một danh mục lạ, sinh
`corrected_context` sai, và lần thử lại hỏng đúng như lần đầu.

Trước bản sửa này, tính chất "một nguồn sự thật" đã hỏng sẵn mà không ai
thấy: mười chỗ đặt `error_type` trong bốn file agent viết chuỗi trần thay
vì hằng số, và hai trong số đó (`"missing_step"`, `"missing_data"`) không
phải thành viên của `ERROR_LABELS`.

Danh sách dưới đây duy trì BẰNG TAY, cố ý. Một bản quét AST qua
`graph/agents/*.py` sẽ bắt được nhiều hơn nhưng vỡ mỗi lần ai đó đổi cách
dựng dict; danh sách tay thì buộc người thêm nhãn mới phải chạm vào đúng
hai chỗ — `graph/errors.py` và file này — và đó chính là bước dừng ta muốn.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph import errors
from graph.errors import ERROR_LABELS

# Mọi giá trị `error_type` mà codebase này được biết là có phát ra, kèm
# nơi phát ra. Thêm một nhãn mới ở agent thì thêm một dòng ở đây.
EMITTED_LABELS = {
    "planning_failure": "graph/agents/supervisor.py — hết retry lập kế hoạch",
    "sql_execution": "graph/agents/sql_agent.py — hết retry sinh/chạy SQL",
    "python_runtime": "graph/agents/python_agent.py — hết retry chạy pandas",
    "chart_error": "graph/agents/viz_agent.py — hết retry vẽ biểu đồ",
    "insight_generation": "graph/agents/insight_agent.py — sinh/validate insight hỏng",
    "missing_step": "sql/python/viz/insight — kế hoạch không có bước cho node này",
    "missing_data": "python/viz — shared_dataframe rỗng, bước trước chưa chạy",
    "budget_exceeded": "sql/python/viz/insight — BudgetExceededError, hết ngân sách",
}


def test_every_label_the_code_emits_is_a_member_of_the_closed_set():
    outside = sorted(set(EMITTED_LABELS) - ERROR_LABELS)
    assert not outside, (
        f"nhãn được phát ra nhưng không có trong ERROR_LABELS: {outside}. "
        "Thêm nó vào graph/errors.py, đừng nới lỏng test này."
    )


def test_every_emitted_label_has_a_named_constant():
    """Chuỗi trần ở nơi phát ra làm hỏng đúng cái tính chất tập đóng mua
    được: đổi tên nhãn ở `graph/errors.py` không kéo theo chỗ dùng, và hai
    bên trôi ra xa nhau trong im lặng."""
    constants = {
        value for name, value in vars(errors).items()
        if name.isupper() and isinstance(value, str)
    }
    missing = sorted(set(EMITTED_LABELS) - constants)
    assert not missing, f"nhãn không có hằng số tương ứng trong graph.errors: {missing}"


def test_the_budget_label_is_actually_emitted_somewhere():
    """`budget_exceeded` là nhãn chữ ký của chính plan này. Nó từng có mặt
    trong ERROR_LABELS mà không nơi nào phát ra — một nhãn chết, khiến
    không thể phân biệt "hết giờ" với "công cụ hỏng" trong log."""
    assert errors.BUDGET_EXCEEDED in EMITTED_LABELS


def test_a_budget_exceeded_call_is_labelled_as_such_not_as_a_tool_failure():
    """Kiểm HÀNH VI, không chỉ khai báo.

    `BudgetExceededError` kế thừa `RuntimeError`, nên một `except Exception`
    trần sẽ nuốt nó và node dán nhãn thất bại thường của mình
    (`sql_execution`). Log khi đó nói "SQL hỏng" về một lượt mà model còn
    chưa được gọi — đúng thứ mà nhãn `budget_exceeded` sinh ra để tránh."""
    from unittest.mock import MagicMock, patch

    from graph.agents.sql_agent import sql_agent_node
    from graph.state import make_initial_state
    from model.model_client import BudgetExceededError
    from perception.schema_context import SchemaContext

    state = make_initial_state("q", SchemaContext(), budget_s=600)
    state["execution_plan"] = [
        {"step": 1, "agent": "sql", "task": "t", "depends_on": [], "skill_type": "text-to-sql"},
    ]
    state["completed_agents"] = ["supervisor"]
    state["status"] = "running"
    state["shared_metadata"] = {"profile": object(), "user": "u"}

    client = MagicMock()
    client.invoke.side_effect = BudgetExceededError("còn 1.0s, cần khoảng 15s")

    with patch("graph.agents.sql_agent.ModelClient", return_value=client):
        out = sql_agent_node(state)

    assert out["last_error"]["error_type"] == errors.BUDGET_EXCEEDED, (
        "hết ngân sách bị dán nhãn như một lỗi công cụ: "
        f"{out['last_error']['error_type']}"
    )


def test_the_declared_set_has_no_unreachable_extras_beyond_the_known_spec_nine():
    """Nhãn khai mà chưa nơi nào phát ra thì phải là nhãn của spec 5.6 còn
    để dành, không phải rác tích tụ."""
    spec_reserved = {
        errors.SQL_GENERATION, errors.SQL_TIMEOUT,
        errors.SCHEMA_MISMATCH, errors.MODEL_TIMEOUT, errors.TOOL_UNAVAILABLE,
    }
    unaccounted = sorted(ERROR_LABELS - set(EMITTED_LABELS) - spec_reserved)
    assert not unaccounted, f"nhãn không ai phát ra và cũng không phải của spec: {unaccounted}"
