import inspect
from unittest.mock import MagicMock, patch

import pandas as pd

from graph import multi_agent
from graph.agents import sql_agent, supervisor
from graph.multi_agent import run_graph
from graph.state import make_initial_state
from perception.connection_profile import ALL_TABLES, build_profile
from perception.schema_context import resolve_schema_context
from tests.fixtures.mini_schema import MINI_TABLES
from tests.integration import VALID_INSIGHT

ALL = frozenset(t.name for t in MINI_TABLES)


def _ctx():
    p = build_profile(dsn="postgresql://u:p@h:5432/d", tables=MINI_TABLES,
                      grants={"admin": frozenset({ALL_TABLES})})
    return resolve_schema_context(p, "doanh thu", permitted=ALL)


def test_run_graph_takes_a_schema_context_not_an_info_box():
    params = list(inspect.signature(multi_agent.run_graph).parameters)
    assert "schema_context" in params
    assert "info_box" not in params


def test_initial_state_carries_the_schema_context():
    state = make_initial_state("câu hỏi", _ctx())
    assert state["schema_context"].rendered_text.startswith("--") or \
           state["schema_context"].rendered_text.startswith("CREATE TABLE")
    assert "info_box" not in state


def test_sql_prompt_contains_rendered_ddl_not_json():
    prompt = sql_agent.build_system_prompt(_ctx())
    assert "CREATE TABLE orders (" in prompt
    assert '"table_name"' not in prompt  # không còn json.dumps


def test_supervisor_prompt_contains_rendered_ddl_not_json():
    prompt = supervisor.build_system_prompt(_ctx())
    assert "CREATE TABLE orders (" in prompt
    assert '"table_name"' not in prompt


def test_prompt_omits_tables_outside_the_context():
    p = build_profile(dsn="d", tables=MINI_TABLES,
                      grants={"sales": frozenset({"orders", "customers"})})
    ctx = resolve_schema_context(p, "doanh thu", permitted=frozenset({"orders", "customers"}))
    prompt = sql_agent.build_system_prompt(ctx)
    assert "payroll" not in prompt


def test_run_graph_puts_profile_and_user_into_shared_metadata():
    """Regression test for gap P1 (Task 8/10): sql_agent_node reads
    state["shared_metadata"]["profile"]/["user"] to call execute_sql(), and
    run_graph() is the only thing that puts them there
    (graph/multi_agent.py: `initial["shared_metadata"] = {"profile": profile,
    "user": user}`). This test calls run_graph() itself — not
    build_multi_agent_graph().invoke() with a hand-built state, like the
    other integration tests do — because those bypass the exact line this
    test exists to guard. Every ModelClient the graph can reach is mocked,
    including the reflector's, so this never makes a real model call and
    stays fast even if something goes wrong and the reflector gets invoked.

    Verified to actually catch a regression: temporarily deleted the
    `initial["shared_metadata"] = {...}` line from graph/multi_agent.py,
    reran this test, watched it fail with
    `KeyError: 'profile'` (raised inside sql_agent_node's
    `meta["profile"]` lookup), then restored the line and reran to confirm
    it passes again. See task-10-report.md addendum for the exact output.
    """
    p = build_profile(dsn="postgresql://u:p@h:5432/d", tables=MINI_TABLES,
                      grants={"tester": frozenset({ALL_TABLES})})
    ctx = resolve_schema_context(p, "doanh thu theo region", permitted=ALL)

    plan = [
        {"step": 1, "agent": "sql", "task": "Lấy tổng doanh thu theo region",
         "depends_on": [], "skill_type": "text-to-sql"},
        {"step": 2, "agent": "insight", "task": "Nhận xét",
         "depends_on": ["sql"], "skill_type": "insight-generation"},
    ]

    supervisor_mock = MagicMock()
    supervisor_mock.invoke_json.return_value = {"plan_summary": "test", "steps": plan}
    supervisor_mock.calls_made = 1

    sql_mock = MagicMock()
    sql_mock.invoke.return_value = (
        "SELECT region, SUM(amount) AS total FROM orders GROUP BY region"
    )
    sql_mock.calls_made = 1

    insight_mock = MagicMock()
    insight_mock.invoke_json.return_value = VALID_INSIGHT
    insight_mock.calls_made = 1

    reflector_mock = MagicMock()  # must not be needed on this happy path

    df = pd.DataFrame({"region": ["North"], "total": [100]})

    with patch("graph.agents.supervisor.ModelClient", return_value=supervisor_mock), \
         patch("graph.agents.sql_agent.ModelClient", return_value=sql_mock), \
         patch("graph.agents.sql_agent.execute_sql", return_value=df), \
         patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
         patch("graph.agents.insight_agent.ModelClient", return_value=insight_mock), \
         patch("graph.agents.reflector_agent.ModelClient", return_value=reflector_mock):
        result = run_graph(query="doanh thu theo region", schema_context=ctx,
                           profile=p, user="tester")

    assert result["status"] == "success"
    assert result["shared_metadata"]["profile"] is p
    assert result["shared_metadata"]["user"] == "tester"
    assert reflector_mock.invoke_json.call_count == 0
