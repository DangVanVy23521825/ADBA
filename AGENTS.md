# AGENTS.md — ADBA

## Project state

The `.cursorrules` file is the authoritative architecture spec — it describes the full multi-agent LangGraph system. **The `graph/` directory is empty stubs.** What is implemented:

- `model/` — Ollama-first LLM client with OpenAI fallback
- `schemas/` — Pydantic v2 models for `ExecutionPlan` and `InsightOutput`
- `perception/` — PostgreSQL schema introspection → compact `info_box` JSON
- `training/` — end-to-end pipeline: generate SFT data → validate → format → MLX-LM fine-tune
- `eval/` — standalone evaluation runner with baseline results
- `data/` — synthetic seed data with 15 injected cross-domain anomalies (S1–S5, I1–I5, H1–H5)
- `graph/` — LangGraph skeleton: `MultiAgentState`, supervisor agent with routing logic, specialist node stubs
- `tests/unit/` — pytest tests (run with `PYTHONPATH=. pytest tests/ -v`)
- No `app.py`, no specialist agent implementations yet

## Environment

- macOS M2, Python 3.11+, 16GB RAM
- Ollama with `OLLAMA_NUM_CTX=4096` (default 2048 is too small for agent prompts)
- Primary model: `qwen2.5-coder:7b-instruct-q5_K_M`
- PostgreSQL 15 via Docker (`docker compose up -d`)

## Key commands

```bash
# Database
docker compose up -d                                          # start PostgreSQL
./scripts/apply_schemas_docker.sh                             # apply DDL schemas
python data/seed/seed_data.py                                 # generate synthetic data
docker exec -it adba-postgres psql -U adba_user -d adba_db    # interactive psql

# Schema introspection
python perception/extract_info_box.py                         # generates info_box_*.json

# Training pipeline (must run in order)
python training/generate_data.py                              # GPT-4o-mini → raw_dataset.jsonl
python training/validate_dataset.py                           # SQL EXPLAIN + AST + Pydantic validation
python training/format_sharegpt.py                            # format → train.jsonl / valid.jsonl / test.jsonl
python training/train_mlx.py                                  # MLX-LM LoRA fine-tuning

# Evaluation
python eval/eval_runner.py                                    # baseline eval against test.jsonl

# Quick smoke test
python testing.py                                             # ModelClient + PostgreSQL connectivity

# Tests
PYTHONPATH=. pytest tests/ -v                                 # run all unit tests
```

## Import conventions

No `pyproject.toml` or package install. All scripts add the project root to `sys.path`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.model_client import ModelClient, safe_parse_json
from schemas.plan_schema import ExecutionPlan
from schemas.insight_schema import InsightOutput
```

## Critical conventions

- **Always use `ModelClient(agent_type="...")`** — never call `ollama.chat()` directly. Agent type sets temperature, max_tokens, and timeout automatically. Valid types: `supervisor`, `sql`, `python`, `viz`, `insight`, `reflector`.

- **Always strip markdown fences before parsing JSON** — use `safe_parse_json()` from `model.model_client`. LLMs frequently wrap JSON in ` ```json ... ``` ` fences.

- **`ModelClient.invoke_json()`** combines invoke + safe_parse_json in one call. Prefer this for structured outputs.

- **Parameterized SQL queries** — use `cursor.execute("... WHERE x = %s", (val,))`, never f-strings.

- **Environments variables** in `.env`:
  - `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`
  - `POSTGRES_URL` (full connection string)
  - `PRIMARY_MODEL`, `BACKUP_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_NUM_CTX`
  - `OPENAI_API_KEY` (optional, for fallback)
  - `ENABLE_OPENAI_FALLBACK` (default `1`; set to `0` to disable)
  - `MODEL_MAX_RETRIES` (default `3`)

- **OpenAI fallback** is enabled by default but only for `supervisor` and `sql` agent types. Other agents fail if Ollama is unavailable.

## .cursorrules as spec

When implementing new agents, agent state, or the LangGraph graph, consult `.cursorrules`. It defines:
- `MultiAgentState` TypedDict structure
- Agent system prompts with few-shot examples
- ExecutionPlan and InsightOutput JSON formats
- Routing logic and error handling conventions
- DataFrame serialization rules (never store DataFrame directly in LangGraph state — use JSON-serializable dicts)
