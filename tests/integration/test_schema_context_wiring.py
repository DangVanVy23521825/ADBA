import inspect

from graph import multi_agent
from graph.agents import sql_agent, supervisor
from graph.state import make_initial_state
from perception.connection_profile import ALL_TABLES, build_profile
from perception.schema_context import resolve_schema_context
from tests.fixtures.mini_schema import MINI_TABLES

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
