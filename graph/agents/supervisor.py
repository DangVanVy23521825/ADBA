"""
Supervisor Agent — parse query, produce ExecutionPlan, validate via Pydantic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, Literal

from graph.state import MultiAgentState
from graph.utils import append_trace
from model.model_client import ModelClient
from perception.schema_context import SchemaContext
from schemas.plan_schema import ExecutionPlan

logger = logging.getLogger(__name__)

# Max reflector invocations targeted at the same specialist agent — avoids infinite
# sql ↔ reflector loops while still allowing a new diagnosis each time that agent's
# agent_error_counts increases after a retry.
MAX_REFLECTOR_PASSES_PER_AGENT = 8
MAX_SUPERVISOR_RETRIES = 3
SPECIALIST_AGENTS = {"sql", "python", "viz", "insight"}

# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "supervisor_routing.txt"

SUPERVISOR_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


def build_system_prompt(schema_context: SchemaContext) -> str:
    """Render the prompt template with this turn's rendered schema slice.

    The template's trailing "User query: {query}" line
    (prompts/supervisor_routing.txt) is dropped rather than substituted:
    supervisor_node always sends the real query as the user turn
    (`user_prompt=query` below), so leaving the placeholder unsubstituted
    would send a literal "{query}" to the model, and substituting it here
    would send the query twice (system prompt and user turn). Dropping it
    means the query reaches the model exactly once, via the user turn.
    """
    rendered = SUPERVISOR_SYSTEM_PROMPT.replace("{schema}", schema_context.rendered_text)
    return rendered.replace("User query: {query}\n\n", "")


def build_dependency_graph(plan: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return JSON-safe dependency graph keyed by agent name."""
    return {str(step.get("agent", "")): list(step.get("depends_on", [])) for step in plan}


def resolve_ready_agents(
    plan: list[dict[str, Any]],
    completed_agents: list[str],
    blocked_agents: set[str] | None = None,
) -> list[str]:
    """Return all dependency-satisfied agents in stable step order.

    The graph can expose multiple ready agents for future parallel dispatch. The
    current LangGraph wiring still consumes the first item for sequential runs.
    """
    completed = set(completed_agents)
    blocked = blocked_agents or set()
    ready: list[str] = []
    for step in sorted(plan, key=lambda s: int(s.get("step", 0))):
        agent = str(step.get("agent", ""))
        if agent in completed or agent in blocked:
            continue
        deps = step.get("depends_on", [])
        if all(dep in completed for dep in deps):
            ready.append(agent)
    return ready


def validate_dependency_graph(plan: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Validate a full DAG from plain dict plan and return its adjacency map."""
    agents = [str(step.get("agent", "")) for step in plan]
    known = set(agents)
    graph = build_dependency_graph(plan)

    for agent, deps in graph.items():
        unknown = set(deps) - known
        if unknown:
            raise ValueError(f"agent '{agent}' depends on unknown agents: {sorted(unknown)}")

    indegree = {agent: 0 for agent in agents}
    successors: dict[str, list[str]] = defaultdict(list)
    for agent, deps in graph.items():
        for dep in deps:
            indegree[agent] += 1
            successors[dep].append(agent)

    queue = deque([agent for agent in agents if indegree[agent] == 0])
    visited: list[str] = []
    while queue:
        agent = queue.popleft()
        visited.append(agent)
        for successor in successors[agent]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)

    if len(visited) != len(agents):
        raise ValueError("execution_plan contains a dependency cycle")
    return graph


# ── Node ──────────────────────────────────────────────────────────────────────

def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    """Parse user query → call ModelClient → validate ExecutionPlan → update state.

    Errors are caught and surfaced through state["last_error"] rather than
    raising, so the graph can route to the reflector if needed.
    """
    query: str = state["query"]
    schema_context = state["schema_context"]
    logger.info("Supervisor: planning for query '%s'", query)

    trace = append_trace(state, "supervisor", "parse_intent",
                         f"Planning query: {query}", "started")

    client = ModelClient(agent_type="supervisor")
    system_prompt = build_system_prompt(schema_context)
    last_exc: Exception | None = None

    for attempt in range(1, MAX_SUPERVISOR_RETRIES + 1):
        try:
            raw_plan: dict = client.invoke_json(
                system_prompt=system_prompt,
                user_prompt=query,
            )

            # Pydantic validation — catches bad structure, missing fields,
            # cycles, forward deps, wrong skill_type, etc.
            validated = ExecutionPlan.model_validate(raw_plan)
            break
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Supervisor planning attempt %d/%d failed: %s",
                attempt, MAX_SUPERVISOR_RETRIES, exc,
            )
            if attempt < MAX_SUPERVISOR_RETRIES:
                trace = append_trace(
                    state,
                    "supervisor",
                    "parse_intent",
                    f"Attempt {attempt} failed: {exc}",
                    "error",
                )
                state = {**state, "action_trace": trace}
                continue

            error_msg = f"Supervisor failed after {MAX_SUPERVISOR_RETRIES} attempts: {exc}"
            logger.error(error_msg)
            trace = append_trace(state, "supervisor", "parse_intent", error_msg, "error")

            return {
                **state,
                "current_agent": "supervisor",
                "agent_outputs": {
                    **state.get("agent_outputs", {}),
                    "supervisor": {"status": "error", "error": error_msg},
                },
                "error_count": state.get("error_count", 0) + 1,
                "agent_error_counts": {
                    **state.get("agent_error_counts", {}),
                    "supervisor": state.get("agent_error_counts", {}).get("supervisor", 0) + 1,
                },
                "last_error": {
                    "agent": "supervisor",
                    "error_type": "planning_failure",
                    "traceback": error_msg,
                },
                "action_trace": trace,
                "status": "failed",
            }

    if last_exc is not None and "validated" not in locals():
        raise last_exc

    # Convert back to plain dicts for LangGraph state (must be JSON-safe).
    plan_dicts = [step.model_dump() for step in validated.steps]
    dependency_graph = validate_dependency_graph(plan_dicts)
    ready_agents = resolve_ready_agents(plan_dicts, state.get("completed_agents", []) + ["supervisor"])

    trace = append_trace(state, "supervisor", "parse_intent",
                         f"Plan generated: {validated.agents_sequence()}", "ok")
    logger.info("Supervisor: plan %s", validated.agents_sequence())

    return {
        **state,
        "execution_plan": plan_dicts,
        "dependency_graph": dependency_graph,
        "ready_agents": ready_agents,
        "current_agent": "supervisor",
        "completed_agents": state.get("completed_agents", []) + ["supervisor"],
        "agent_outputs": {
            **state.get("agent_outputs", {}),
            "supervisor": {"status": "ok", "output": validated.model_dump()},
        },
        "action_trace": trace,
        "status": "running",
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_next_agent(state: MultiAgentState) -> Literal[
    "sql", "python", "viz", "insight", "reflector", "__end__"
]:
    """Conditional edge — decide which agent runs next.

    Rules:
      1. status == 'failed' or 'success' → END.
      2. Resolve the full dependency graph and expose all currently ready agents
         in state/shared_metadata for future parallel dispatch.
      3. Sequentially dispatch the first ready agent by step order.
      4. Specialist stuck failing: agent_outputs[name].status == 'error' OR
         agent_error_counts[name] >= 3. Then decide whether to call reflector
         again vs skip the step — see shared_metadata.reflect_error_snapshot and
         reflect_passes_per_agent (maintained by reflector_agent_node). A new
         failure tick (incrementing agent_error_counts) clears staleness relative
         to the previous snapshot so reflector runs again until the budget is
         exhausted.
      5. No runnable agent → END.
    """
    if state["status"] in {"failed", "success"}:
        from langgraph.graph import END
        return END

    plan = state.get("execution_plan", [])
    completed = set(state.get("completed_agents", []))
    error_counts = state.get("agent_error_counts", {})
    blocked_agents: set[str] = set()

    try:
        dependency_graph = validate_dependency_graph(plan)
    except Exception as exc:
        logger.error("Invalid dependency graph: %s", exc)
        from langgraph.graph import END
        return END

    for step in plan:
        agent: str = step.get("agent", "")
        if agent in completed:
            continue

        agent_outputs = state.get("agent_outputs", {})
        meta = state.get("shared_metadata", {})
        passes_map: dict[str, int] = dict(meta.get("reflect_passes_per_agent", {}))
        snapshot: dict[str, int] = dict(meta.get("reflect_error_snapshot", {}))
        seen_at_reflect = snapshot.get(agent, -1)
        curr = error_counts.get(agent, 0)
        passes = passes_map.get(agent, 0)

        specialist_err = agent_outputs.get(agent, {}).get("status") == "error"
        error_budget = curr >= 3

        if specialist_err or error_budget:
            stale = curr <= seen_at_reflect
            needs_reflector = ("reflector" not in agent_outputs) or (not stale)

            if needs_reflector and passes < MAX_REFLECTOR_PASSES_PER_AGENT:
                return "reflector"

            if needs_reflector:
                logger.warning(
                    "Agent '%s': reflector cap (%d) reached — skipping stuck step",
                    agent, MAX_REFLECTOR_PASSES_PER_AGENT,
                )
            elif stale and "reflector" in agent_outputs:
                logger.warning(
                    "Agent '%s' still failing with no new error_count since last reflector — skipping",
                    agent,
                )
            blocked_agents.add(agent)
            continue

    ready_agents = resolve_ready_agents(plan, list(completed), blocked_agents)
    state["dependency_graph"] = dependency_graph
    state["ready_agents"] = ready_agents
    state["shared_metadata"] = {
        **state.get("shared_metadata", {}),
        "dependency_graph": dependency_graph,
        "ready_agents": ready_agents,
        "parallel_ready": len(ready_agents) > 1,
    }

    for agent in ready_agents:
        if agent in SPECIALIST_AGENTS:
            return agent  # type: ignore[return-value]

    from langgraph.graph import END
    return END


def route_reflector_return(state: MultiAgentState) -> str:
    """After reflector diagnoses, route back to the failed agent."""
    from langgraph.graph import END

    last_error = state.get("last_error") or {}
    failed_agent = last_error.get("agent", "")
    if failed_agent in {"sql", "python", "viz", "insight"}:
        return failed_agent
    logger.warning("Reflector: no valid agent target → END")
    return END
