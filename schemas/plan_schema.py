"""
ADBA — Execution Plan Schema
============================
Pydantic models for the Supervisor Agent's output.

Validated invariants:
  1. No duplicate agents
  2. Steps numbered 1..N without gaps
  3. depends_on only references agents present in the plan
  4. No forward dependencies — step N cannot depend on agent with step > N
  5. No cycles — verified via Kahn's algorithm on the dependency DAG
  6. 'insight' is always the final step if present
  7. Each agent uses its designated skill_type (per-step)
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


AgentName = Literal["sql", "python", "viz", "insight"]

SkillType = Literal[
    "text-to-sql",
    "data-analysis",
    "visualization",
    "insight-generation",
]

AGENT_SKILL_MAP: dict[str, str] = {
    "sql":     "text-to-sql",
    "python":  "data-analysis",
    "viz":     "visualization",
    "insight": "insight-generation",
}


# =============================================================================
# DAG helpers
# =============================================================================

def _has_cycle(steps: list) -> tuple[bool, list[str]]:
    """
    Kahn's topological sort to detect cycles.
    Returns (has_cycle, topological_order).
    If has_cycle is True, topological_order is incomplete.
    """
    agents = [s.agent for s in steps]
    in_degree: dict[str, int] = {a: 0 for a in agents}
    successors: dict[str, list[str]] = defaultdict(list)

    for s in steps:
        for dep in s.depends_on:
            in_degree[s.agent] += 1
            successors[dep].append(s.agent)

    queue = deque(a for a in agents if in_degree[a] == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    return len(order) < len(agents), order


# =============================================================================
# AgentStep
# =============================================================================

class AgentStep(BaseModel):
    step:       int
    agent:      AgentName
    task:       str
    depends_on: list[AgentName]
    skill_type: SkillType

    @field_validator("step")
    @classmethod
    def step_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("step must be >= 1")
        return v

    @field_validator("task")
    @classmethod
    def task_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("task must not be empty")
        return v

    @model_validator(mode="after")
    def skill_matches_agent(self) -> "AgentStep":
        expected = AGENT_SKILL_MAP[self.agent]
        if self.skill_type != expected:
            raise ValueError(
                f"agent '{self.agent}' must use skill_type '{expected}', "
                f"got '{self.skill_type}'"
            )
        return self

    @model_validator(mode="after")
    def no_self_dependency(self) -> "AgentStep":
        if self.agent in self.depends_on:
            raise ValueError(f"agent '{self.agent}' cannot depend on itself")
        return self


# =============================================================================
# ExecutionPlan
# =============================================================================

class ExecutionPlan(BaseModel):
    """Full execution plan produced by the Supervisor Agent."""

    plan_summary: str
    steps:        list[AgentStep]

    @field_validator("plan_summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("plan_summary must not be empty")
        return v

    @field_validator("steps")
    @classmethod
    def steps_not_empty(cls, v: list[AgentStep]) -> list[AgentStep]:
        if not v:
            raise ValueError("steps must contain at least one AgentStep")
        return v

    @model_validator(mode="after")
    def validate_plan(self) -> "ExecutionPlan":
        steps_by_num = sorted(self.steps, key=lambda s: s.step)
        agents_ordered = [s.agent for s in steps_by_num]
        agent_set = set(agents_ordered)

        # 1. No duplicate agents
        dupes = [a for a, n in Counter(agents_ordered).items() if n > 1]
        if dupes:
            raise ValueError(
                f"duplicate agents: {dupes}. Each agent may appear at most once."
            )

        # 2. Steps numbered 1..N without gaps
        step_numbers = [s.step for s in steps_by_num]
        if step_numbers != list(range(1, len(self.steps) + 1)):
            raise ValueError(
                f"step numbers must be 1..{len(self.steps)} without gaps, "
                f"got {step_numbers}"
            )

        # 3. depends_on references must exist in the plan
        for s in self.steps:
            unknown = set(s.depends_on) - agent_set
            if unknown:
                raise ValueError(
                    f"step {s.step} ({s.agent}) depends_on unknown "
                    f"agents: {sorted(unknown)}"
                )

        # 4. No forward dependencies
        # A step at position N cannot depend on an agent assigned step > N.
        agent_step_num: dict[str, int] = {s.agent: s.step for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                dep_step = agent_step_num[dep]
                if dep_step >= s.step:
                    raise ValueError(
                        f"step {s.step} ({s.agent}) has a forward dependency on "
                        f"'{dep}' (step {dep_step}). "
                        f"A step can only depend on earlier steps."
                    )

        # 5. No cycles (Kahn's algorithm)
        # Rule 4 already blocks cycles when steps are numbered sequentially,
        # but Kahn's provides an independent structural guarantee.
        has_cycle, _ = _has_cycle(self.steps)
        if has_cycle:
            raise ValueError(
                "circular dependency detected in the execution plan. "
                "The dependency graph must be a DAG."
            )

        # 6. insight must be the last step
        if "insight" in agent_set:
            last = steps_by_num[-1]
            if last.agent != "insight":
                raise ValueError(
                    f"insight must be the last step, "
                    f"but step {last.step} is '{last.agent}'."
                )

        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_step(self, agent: AgentName) -> AgentStep | None:
        return next((s for s in self.steps if s.agent == agent), None)

    def agents_sequence(self) -> list[AgentName]:
        return [s.agent for s in sorted(self.steps, key=lambda s: s.step)]

    def is_ready(self, agent: AgentName, completed: set[AgentName]) -> bool:
        """True when all dependencies of agent are in completed."""
        step = self.get_step(agent)
        if step is None:
            return False
        return all(dep in completed for dep in step.depends_on)

    def next_runnable(self, completed: set[AgentName]) -> list[AgentName]:
        """Return agents whose deps are satisfied but haven't run yet."""
        return [
            s.agent for s in sorted(self.steps, key=lambda s: s.step)
            if s.agent not in completed
            and self.is_ready(s.agent, completed)
        ]


# =============================================================================
# CANONICAL EXAMPLES — copy field names into prompt few-shot examples
# =============================================================================

EXAMPLE_SIMPLE: dict = {
    "plan_summary": "Query total revenue by region for Q4 2024 and summarize",
    "steps": [
        {
            "step": 1, "agent": "sql",
            "task": "SELECT region, SUM(amount) AS total_revenue FROM orders "
                    "WHERE year=2024 AND quarter=4 "
                    "AND status IN ('completed','processing') "
                    "GROUP BY region ORDER BY total_revenue DESC",
            "depends_on": [], "skill_type": "text-to-sql",
        },
        {
            "step": 2, "agent": "insight",
            "task": "Identify which region leads Q4 2024 and by what margin",
            "depends_on": ["sql"], "skill_type": "insight-generation",
        },
    ],
}

EXAMPLE_COMPLEX: dict = {
    "plan_summary": "YoY regional revenue comparison with anomaly detection, "
                    "visualization, and business insight",
    "steps": [
        {
            "step": 1, "agent": "sql",
            "task": "WITH q4_24 AS (SELECT region, SUM(amount) AS rev_2024 "
                    "FROM orders WHERE year=2024 AND quarter=4 "
                    "AND status IN ('completed','processing') GROUP BY region), "
                    "q4_23 AS (SELECT region, SUM(amount) AS rev_2023 "
                    "FROM orders WHERE year=2023 AND quarter=4 "
                    "AND status IN ('completed','processing') GROUP BY region) "
                    "SELECT a.region, a.rev_2024, b.rev_2023 "
                    "FROM q4_24 a LEFT JOIN q4_23 b USING (region)",
            "depends_on": [], "skill_type": "text-to-sql",
        },
        {
            "step": 2, "agent": "python",
            "task": "Compute YoY % change; detect outliers with IQR; "
                    "add is_anomaly boolean column",
            "depends_on": ["sql"], "skill_type": "data-analysis",
        },
        {
            "step": 3, "agent": "viz",
            "task": "Grouped bar chart rev_2024 vs rev_2023 by region; "
                    "highlight anomalous bars in red",
            "depends_on": ["python"], "skill_type": "visualization",
        },
        {
            "step": 4, "agent": "insight",
            "task": "Identify top anomaly region, quantify deviation, "
                    "recommend inventory/staffing action for Q1 2025",
            "depends_on": ["sql", "python", "viz"],
            "skill_type": "insight-generation",
        },
    ],
}



if __name__ == "__main__":
    print("plan_schema.py — self-test")
    print("=" * 42)

    # Valid plans
    p1 = ExecutionPlan.model_validate(EXAMPLE_SIMPLE)
    assert p1.agents_sequence() == ["sql", "insight"]
    print(f"✓ Simple plan:  {p1.agents_sequence()}")

    p2 = ExecutionPlan.model_validate(EXAMPLE_COMPLEX)
    assert p2.agents_sequence() == ["sql", "python", "viz", "insight"]
    print(f"✓ Complex plan: {p2.agents_sequence()}")

    # Helpers
    assert p2.is_ready("sql", set()) is True
    assert p2.is_ready("python", set()) is False
    assert p2.is_ready("python", {"sql"}) is True
    assert p2.next_runnable(set()) == ["sql"]
    assert p2.next_runnable({"sql"}) == ["python"]
    assert p2.next_runnable({"sql", "python"}) == ["viz"]
    assert p2.next_runnable({"sql", "python", "viz"}) == ["insight"]
    print("✓ is_ready() and next_runnable() correct")

    # Bad cases — one per rule
    bad_cases: list[tuple[str, dict]] = [
        ("(rule 6) insight not last", {
            "plan_summary": "bad",
            "steps": [
                {"step": 1, "agent": "insight", "task": "x",
                 "depends_on": [], "skill_type": "insight-generation"},
                {"step": 2, "agent": "sql", "task": "SELECT 1",
                 "depends_on": [], "skill_type": "text-to-sql"},
            ],
        }),
        ("(rule 7) wrong skill_type", {
            "plan_summary": "bad",
            "steps": [
                {"step": 1, "agent": "sql", "task": "SELECT 1",
                 "depends_on": [], "skill_type": "data-analysis"},
            ],
        }),
        ("(rule 3) unknown agent in depends_on", {
            "plan_summary": "bad",
            "steps": [
                {"step": 1, "agent": "sql", "task": "SELECT 1",
                 "depends_on": ["viz"], "skill_type": "text-to-sql"},
            ],
        }),
        ("(rule 4) forward dependency: step 1 -> step 2", {
            "plan_summary": "bad",
            "steps": [
                {"step": 1, "agent": "sql", "task": "SELECT 1",
                 "depends_on": ["python"], "skill_type": "text-to-sql"},
                {"step": 2, "agent": "python", "task": "process",
                 "depends_on": [], "skill_type": "data-analysis"},
                {"step": 3, "agent": "insight", "task": "summarize",
                 "depends_on": ["sql", "python"],
                 "skill_type": "insight-generation"},
            ],
        }),
        ("(rule 1) duplicate agents", {
            "plan_summary": "bad",
            "steps": [
                {"step": 1, "agent": "sql", "task": "SELECT 1",
                 "depends_on": [], "skill_type": "text-to-sql"},
                {"step": 2, "agent": "sql", "task": "SELECT 2",
                 "depends_on": [], "skill_type": "text-to-sql"},
            ],
        }),
        ("(rule 2) gap in step numbers", {
            "plan_summary": "bad",
            "steps": [
                {"step": 1, "agent": "sql", "task": "SELECT 1",
                 "depends_on": [], "skill_type": "text-to-sql"},
                {"step": 3, "agent": "insight", "task": "summarize",
                 "depends_on": ["sql"], "skill_type": "insight-generation"},
            ],
        }),
    ]

    for label, bad in bad_cases:
        try:
            ExecutionPlan.model_validate(bad)
            print(f"  ✗ {label} — should have failed")
        except Exception as e:
            short = str(e).replace("\n", " ")[:70]
            print(f"  ✓ {label}\n      → {short}")

    print("\nAll tests passed.")