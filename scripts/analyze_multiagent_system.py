#!/usr/bin/env python3
"""
Analyze how the ADBA multi-agent system coordinates its agents.

This script is intentionally static-analysis-first:
- No dependency on LangGraph runtime
- No model/database calls
- Reads the current source tree and emits a concise system report

Usage:
  python3 scripts/analyze_multiagent_system.py
  python3 scripts/analyze_multiagent_system.py --format json
  python3 scripts/analyze_multiagent_system.py --output docs/multiagent_analysis.md
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "graph"
AGENTS_DIR = GRAPH_DIR / "agents"
STATE_FILE = GRAPH_DIR / "state.py"
GRAPH_FILE = GRAPH_DIR / "multi_agent.py"


AGENT_SPECS = {
    "supervisor": {
        "module": AGENTS_DIR / "supervisor.py",
        "role": "Plans the execution order and validates the ExecutionPlan schema.",
        "outputs": ["execution_plan", "agent_outputs.supervisor", "status"],
        "consumes_from_agents": [],
    },
    "sql": {
        "module": AGENTS_DIR / "sql_agent.py",
        "role": "Generates SQL, executes it, and serializes the result into shared state.",
        "outputs": ["shared_dataframe", "sql_result", "shared_metadata.sql_query"],
        "consumes_from_agents": ["supervisor"],
    },
    "python": {
        "module": AGENTS_DIR / "python_agent.py",
        "role": "Runs pandas transformations on the shared DataFrame.",
        "outputs": ["shared_dataframe", "python_result", "shared_metadata.python_stats"],
        "consumes_from_agents": ["sql", "reflector"],
    },
    "viz": {
        "module": AGENTS_DIR / "viz_agent.py",
        "role": "Generates and executes chart code, returning base64 PNG metadata.",
        "outputs": ["chart_b64", "chart_metadata", "shared_metadata.chart_type"],
        "consumes_from_agents": ["sql or python", "reflector"],
    },
    "insight": {
        "module": AGENTS_DIR / "insight_agent.py",
        "role": "Builds the final structured business insight from prior outputs.",
        "outputs": ["insight", "status=success"],
        "consumes_from_agents": ["sql", "python", "viz"],
    },
    "reflector": {
        "module": AGENTS_DIR / "reflector_agent.py",
        "role": "Diagnoses failed specialist steps and injects corrected retry context.",
        "outputs": ["shared_metadata.reflector_diagnosis", "agent_outputs.reflector"],
        "consumes_from_agents": ["sql/python/viz/insight failure"],
    },
}


@dataclass
class ModuleAnalysis:
    path: str
    reads: list[str]
    writes: list[str]
    constants: dict[str, Any]


def _parse_ast(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class StateAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.constants: dict[str, Any] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                try:
                    self.constants[name] = ast.literal_eval(node.value)
                except Exception:
                    pass
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "state":
            key = _literal_key(node.slice)
            if key:
                self.reads.add(key)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "state":
                if node.func.attr == "get" and node.args:
                    key = _literal_key(node.args[0])
                    if key:
                        self.reads.add(key)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                key_text = _literal_key(key)
                if key_text:
                    self.writes.add(key_text)
        self.generic_visit(node)


def analyze_module(path: Path) -> ModuleAnalysis:
    tree = _parse_ast(path)
    analyzer = StateAnalyzer()
    analyzer.visit(tree)
    return ModuleAnalysis(
        path=str(path),
        reads=sorted(analyzer.reads),
        writes=sorted(analyzer.writes),
        constants=analyzer.constants,
    )


def extract_state_fields(path: Path) -> list[str]:
    tree = _parse_ast(path)
    fields: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MultiAgentState":
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    fields.append(child.target.id)
    return fields


def build_report() -> dict[str, Any]:
    state_fields = extract_state_fields(STATE_FILE)

    modules: dict[str, ModuleAnalysis] = {
        name: analyze_module(spec["module"])
        for name, spec in AGENT_SPECS.items()
    }

    graph_analysis = analyze_module(GRAPH_FILE)

    normal_flow = [
        "supervisor -> sql -> insight",
        "supervisor -> sql -> python -> insight",
        "supervisor -> sql -> viz -> insight",
        "supervisor -> sql -> python -> viz -> insight",
    ]
    recovery_flow = [
        "specialist error -> supervisor.route_next_agent() -> reflector",
        "reflector -> failed specialist (sql/python/viz/insight)",
        "reflector hints are injected via shared_metadata.reflector_diagnosis.corrected_context",
        "stale or over-budget failures are skipped by the router",
    ]

    handoffs = [
        {
            "from": "supervisor",
            "to": "sql/python/viz/insight",
            "artifact": "execution_plan",
            "notes": "Dependency order is plan-driven, not hardcoded per query.",
        },
        {
            "from": "sql",
            "to": "python/viz/insight",
            "artifact": "shared_dataframe + sql_result + shared_metadata.sql_query",
            "notes": "DataFrame is serialized through df_to_state().",
        },
        {
            "from": "python",
            "to": "viz/insight",
            "artifact": "shared_dataframe + python_result + shared_metadata.python_stats",
            "notes": "Python can replace the shared DataFrame with transformed output.",
        },
        {
            "from": "viz",
            "to": "insight/UI",
            "artifact": "chart_b64 + chart_metadata",
            "notes": "Visualization is optional and does not overwrite shared_dataframe.",
        },
        {
            "from": "reflector",
            "to": "sql/python/viz/insight",
            "artifact": "shared_metadata.reflector_diagnosis",
            "notes": "Retry hint is appended to the next user prompt for the failed specialist.",
        },
    ]

    return {
        "project_root": str(ROOT),
        "state_fields": state_fields,
        "graph": {
            "module": str(GRAPH_FILE),
            "reads": graph_analysis.reads,
            "writes": graph_analysis.writes,
            "entry_point": "supervisor",
            "terminal_agents": ["insight", "__end__"],
            "normal_flow_patterns": normal_flow,
            "recovery_flow_patterns": recovery_flow,
        },
        "agents": {
            name: {
                "module": str(AGENT_SPECS[name]["module"]),
                "role": AGENT_SPECS[name]["role"],
                "reads_state": modules[name].reads,
                "writes_state": modules[name].writes,
                "retry_config": {
                    key: value
                    for key, value in modules[name].constants.items()
                    if key in {"MAX_RETRIES", "MAX_REFLECTOR_PASSES_PER_AGENT"}
                },
                "consumes_from_agents": AGENT_SPECS[name]["consumes_from_agents"],
                "produces_artifacts": AGENT_SPECS[name]["outputs"],
            }
            for name in AGENT_SPECS
        },
        "handoffs": handoffs,
        "mermaid": build_mermaid(),
    }


def build_mermaid() -> str:
    return "\n".join(
        [
            "flowchart TD",
            '    U["User Query"] --> S["Supervisor"]',
            '    S -->|ExecutionPlan| SQL["SQL Agent"]',
            '    S -->|ExecutionPlan| PY["Python Agent"]',
            '    S -->|ExecutionPlan| VZ["Viz Agent"]',
            '    S -->|ExecutionPlan| IN["Insight Agent"]',
            '    SQL -->|shared_dataframe + sql_result| PY',
            '    SQL -->|shared_dataframe + sql_query| VZ',
            '    SQL -->|sql_result| IN',
            '    PY -->|transformed shared_dataframe + stats| VZ',
            '    PY -->|python_result + stats| IN',
            '    VZ -->|chart_b64 + chart_metadata| IN',
            '    SQL -. failure .-> RF["Reflector"]',
            '    PY -. failure .-> RF',
            '    VZ -. failure .-> RF',
            '    IN -. failure .-> RF',
            '    RF -->|corrected_context| SQL',
            '    RF -->|corrected_context| PY',
            '    RF -->|corrected_context| VZ',
            '    RF -->|corrected_context| IN',
            '    IN --> O["Structured Insight / UI"]',
        ]
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Multi-Agent Interaction Analysis")
    lines.append("")
    lines.append(f"- Project root: `{report['project_root']}`")
    lines.append(f"- Graph module: `{report['graph']['module']}`")
    lines.append(f"- Entry point: `{report['graph']['entry_point']}`")
    lines.append("")
    lines.append("## State Contract")
    lines.append("")
    lines.append(", ".join(f"`{field}`" for field in report["state_fields"]))
    lines.append("")
    lines.append("## Normal Flow")
    lines.append("")
    for flow in report["graph"]["normal_flow_patterns"]:
        lines.append(f"- `{flow}`")
    lines.append("")
    lines.append("## Recovery Flow")
    lines.append("")
    for flow in report["graph"]["recovery_flow_patterns"]:
        lines.append(f"- {flow}")
    lines.append("")
    lines.append("## Agent Roles")
    lines.append("")
    for name, info in report["agents"].items():
        lines.append(f"### `{name}`")
        lines.append(f"- Module: `{info['module']}`")
        lines.append(f"- Role: {info['role']}")
        lines.append(f"- Reads state: {', '.join(f'`{x}`' for x in info['reads_state']) or '(none detected)'}")
        lines.append(f"- Writes state: {', '.join(f'`{x}`' for x in info['writes_state']) or '(none detected)'}")
        if info["retry_config"]:
            retry_str = ", ".join(f"`{k}={v}`" for k, v in info["retry_config"].items())
            lines.append(f"- Retry config: {retry_str}")
        else:
            lines.append("- Retry config: `(none)`")
        lines.append(f"- Consumes from: {', '.join(info['consumes_from_agents']) or '(none)'}")
        lines.append(f"- Produces: {', '.join(info['produces_artifacts'])}")
        lines.append("")
    lines.append("## Handoffs")
    lines.append("")
    for handoff in report["handoffs"]:
        lines.append(
            f"- `{handoff['from']}` -> `{handoff['to']}` via `{handoff['artifact']}`: {handoff['notes']}"
        )
    lines.append("")
    lines.append("## Mermaid")
    lines.append("")
    lines.append("```mermaid")
    lines.append(report["mermaid"])
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the ADBA multi-agent flow.")
    parser.add_argument(
        "--format",
        choices=("md", "json"),
        default="md",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write the report. Defaults to stdout.",
    )
    args = parser.parse_args()

    report = build_report()
    rendered = (
        render_markdown(report)
        if args.format == "md"
        else json.dumps(report, ensure_ascii=False, indent=2)
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
