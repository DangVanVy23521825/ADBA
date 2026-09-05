"""
Reflector Agent — diagnose errors and produce corrected context for retry.
"""

from __future__ import annotations

import logging

from graph.budget import calls_remaining, node_may_run
from graph.state import MultiAgentState
from graph.utils import append_trace
from model.model_client import ModelClient

logger = logging.getLogger(__name__)

REFLECTOR_SYSTEM_PROMPT = """\
## ROLE
You are an error diagnosis specialist for a multi-agent data pipeline.
A specialist agent failed. Diagnose the error and provide a corrected context.

## ERROR CATEGORIES
- sql_syntax: typo, missing keyword, wrong function name
- sql_logic: wrong join, incorrect aggregation, bad filter
- python_runtime: NameError, TypeError, KeyError on DataFrame
- python_logic: wrong calculation, incorrect column selection
- data_quality: NaN, wrong dtypes, empty result
- schema_mismatch: column doesn't exist, wrong table name
- chart_error: matplotlib code error, missing import

## OUTPUT FORMAT
Return ONLY valid JSON:
{
  "root_cause": "<what actually went wrong>",
  "error_category": "<sql_syntax|sql_logic|python_runtime|python_logic|data_quality|schema_mismatch|chart_error>",
  "fix_strategy": "<1-sentence fix plan>",
  "corrected_context": "<the corrected instruction to pass back>"
}

## CONTEXT
Failed agent: {failed_agent}
Task that failed: {failed_task}
Error: {error_message}
Previous attempts: {previous_attempt}

## NOW DIAGNOSE

Output:
"""


def reflector_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Diagnose the last error and produce a corrected context for the failed agent."""
    may_run, reason = node_may_run(state, "reflector")
    if not may_run:
        logger.warning("Reflector Agent: %s", reason)
        return {
            **state,
            "current_agent": "reflector",
            "completed_agents": state.get("completed_agents", []) + ["reflector"],
            "degradation_reason": state.get("degradation_reason", []) + [reason],
            "action_trace": append_trace(state, "reflector", "budget_check", reason, "error"),
            "status": "running",
        }

    last_error = state.get("last_error") or {}
    failed_agent = last_error.get("agent", "unknown")
    error_message = last_error.get("traceback", str(last_error))

    # Find the task that failed
    failed_task = "unknown"
    plan = state.get("execution_plan", [])
    for step in plan:
        if step.get("agent") == failed_agent:
            failed_task = step.get("task", "unknown")
            break

    # Count previous attempts for this agent
    prev_attempt = state.get("agent_error_counts", {}).get(failed_agent, 0)

    system_prompt = (REFLECTOR_SYSTEM_PROMPT
                     .replace("{failed_agent}", failed_agent)
                     .replace("{failed_task}", failed_task)
                     .replace("{error_message}", error_message)
                     .replace("{previous_attempt}", str(prev_attempt)))

    trace = append_trace(state, "reflector", "diagnose_error",
                         f"Diagnosing {failed_agent}: {error_message[:100]}", "started")

    # Dựng NGOÀI try — cả nhánh thành công lẫn nhánh fallback bên dưới đều
    # hội tụ về CÙNG một return, đọc `client.calls_made` (xem ghi chú ở
    # model/model_client.py::ModelClient.__init__).
    client = ModelClient(
        agent_type="reflector",
        deadline_ts=state.get("deadline_ts"),
        call_budget_remaining=calls_remaining(state.get("llm_calls_used", 0)),
    )
    try:
        diagnosis = client.invoke_json(
            system_prompt=system_prompt,
            user_prompt=error_message,
        )
    except Exception as exc:
        diagnosis = {
            "root_cause": f"Reflector itself failed: {exc}",
            "error_category": "unknown",
            "fix_strategy": "Retry with original task",
            "corrected_context": failed_task,
        }
        logger.warning("Reflector recovery fallback: %s", exc)

    trace = append_trace(state, "reflector", "diagnose_error",
                         f"Root cause: {diagnosis.get('root_cause', '')[:100]}", "ok")

    meta = dict(state.get("shared_metadata", {}))
    meta["reflector_diagnosis"] = diagnosis

    specialist = {"sql", "python", "viz", "insight"}
    if failed_agent in specialist:
        current_fail_depth = state.get("agent_error_counts", {}).get(failed_agent, 0)
        snap = dict(meta.get("reflect_error_snapshot", {}))
        snap[failed_agent] = current_fail_depth
        passes = dict(meta.get("reflect_passes_per_agent", {}))
        passes[failed_agent] = passes.get(failed_agent, 0) + 1

        meta["reflect_error_snapshot"] = snap
        meta["reflect_passes_per_agent"] = passes

    return {
        **state,
        "current_agent": "reflector",
        "agent_outputs": {
            **state.get("agent_outputs", {}),
            "reflector": {"status": "ok", "output": diagnosis},
        },
        "shared_metadata": meta,
        "action_trace": trace,
        "status": "running",
        "llm_calls_used": state.get("llm_calls_used", 0) + client.calls_made,
    }
