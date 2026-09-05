# Cứng hóa — ngân sách thời gian và cô lập sandbox — Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đóng hai lỗ hổng chặn việc mang ADBA vào mạng khách hàng — không có ngân sách thời gian toàn cục (L1) và sandbox Python cô lập bằng namespace thay vì tiến trình (L3) — để một câu hỏi bất kỳ luôn trả về thứ gì đó trong giới hạn đã biết, và code do model sinh không đọc được credential của tiến trình cha.

**Architecture:** Ngân sách là một `deadline_ts` mang trong LangGraph state, cấp phát theo dự trữ chứ không chia đều, và mọi đường ra khỏi graph đi qua một node `finalize` duy nhất phân loại kết quả thành `success`/`partial`/`failed`. Sandbox đổi từ `multiprocessing` `fork` sang `spawn` với môi trường bị xoá trước khi `exec`, khiến ranh giới trở thành tiến trình chứ không phải namespace.

**Tech Stack:** Python 3.12, LangGraph, psycopg2, sqlparse, pandas, pytest. **Không thêm phụ thuộc mới** — `sqlparse` đã có trong `requirements.txt`.

**Spec:** `docs/superpowers/specs/2026-08-12-adba-mcp-hardening-design.md` — phần còn lại của pha 1, cộng toàn bộ pha 2.
**Lộ trình:** `docs/superpowers/plans/2026-08-16-lo-trinh-toi-dong-goi.md` — Plan B.

---

## Global Constraints

- **Baseline đo được, không lấy theo trí nhớ: 547 pass + 12 skip** (unit 521+9, integration 26+3), chạy hết ~5 giây tổng. Không test nào đang pass được phép fail. 12 test skip đều cần `DATABASE_URL` trỏ Postgres thật (`test_introspect.py`, `test_load_sqlite_to_postgres.py`, `test_onboard_flow.py`) — chúng vẫn skip là đúng. **Con số này được đo lại ngày 2026-09-01 sau khi Plan A (đường onboarding) merge vào `main`** — baseline cũ 227+8 ghi lúc viết plan lần đầu đã lỗi thời, giữ lại ở lịch sử git commit `70dba92` nếu cần đối chiếu.
- **Chạy test:** `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest <dir> -q`, chạy **ở foreground**, hai lệnh riêng cho `tests/unit` và `tests/integration`. Không đẩy vào nền.
- **Số test trong mục "Expected" là phép cộng dồn, dùng như checksum chứ không phải hợp đồng.** Chúng tính từ 547 pass + 12 skip cộng số test mỗi task thêm vào. Lệch một hai con số vì bạn tách hay gộp một test là bình thường — điều thực sự phải đúng là **không có fail**, và delta khớp với số test task đó thêm. Từ Task 3 trở đi, số skip là 19 (12 nền, cần `DATABASE_URL` + 7 readonly, cần `ADBA_READONLY_URL`) trong lần chạy không đặt biến môi trường nào trong hai biến đó.
- **Trước khi bắt đầu Task 1: merge `origin/main` vào nhánh này.** Đã làm một lần ngày 2026-09-01 (commit merge `d1011a9`) — Plan A đã vào `main` (PR #1, #2, #3), mang theo thay đổi ở `model/model_client.py`, `model/model_config.py`, `app.py` và `tests/unit/test_sql_tool_guard.py`. Task 7 và Task 12 dưới đây đã được viết lại theo state hậu-merge; nếu `main` trôi tiếp trước khi thực thi, merge lại và đối chiếu file thật trước khi chạy Step 3 của hai task đó — patch viết sẵn giả định đúng cấu trúc file tại commit `d1011a9`.
- **Tuyệt đối không có lời gọi model thật trong test.** Bộ test hiện tại mock sạch `ModelClient`; logic deadline test bằng **clock tiêm vào**, không bằng `time.sleep`. Nếu một suite vượt ~60 giây thì có lời gọi model thật hoặc một `sleep` lọt vào — dừng và báo.
- **Không chạy `docker compose` trong worktree này.** Container `adba-postgres` ở `localhost:5432` là tài nguyên **dùng chung với worktree Plan A đang chạy**. Chỉ được `psql` vào nó bằng lệnh cộng thêm và idempotent. Cấm: `REVOKE` bất cứ quyền nào của `adba_user`, `DROP` bất cứ thứ gì, `docker compose down`, cờ `-v`.
- **Không chạm `perception/`, `onboard.py`, `pages/`** — đó là lãnh thổ Plan A, đang được sửa song song ở worktree khác. Đụng vào là tạo merge conflict không cần thiết. Plan B sở hữu `graph/`, `model/`, `scripts/`, `docker-compose.yml`.
- **`app.py` là file chung duy nhất giữa hai plan.** Mọi thay đổi `app.py` dồn vào Task 12, làm cuối cùng.
- **Ngân sách mặc định 45s qua `ADBA_QUERY_BUDGET_S`.** 45 là SLO của spec, đo ở cấu hình vLLM/PEFT (12,4s mỗi call). Cấu hình dev hiện tại là Ollama q5 ở **75,8s mỗi call**, nên trên máy này phải đặt `ADBA_QUERY_BUDGET_S=400` để chạy thử end-to-end. **Cổng ra `slo_hit_rate` đo ở cấu hình vLLM, không đo trên Ollama q5** — đừng báo cáo con số đo trên Ollama như thể nó là SLO.
- **Không đụng bất kỳ prompt nào** trong `prompts/`, và không đổi định dạng output của model. Fine-tune gắn với format prompt→text theo 5 skill; đổi định dạng là làm mất hiệu lực nó (spec mục 1).
- **`permitted_tables` là ranh giới bảo mật, dẫn ra bên trong bộ thực thi.** Không biến nó thành tham số của hàm nào (spec 3.4.1).
- Bốn assertion `route_next_agent(...) == END` ở `tests/unit/test_supervisor.py` dòng 127, 132, 142, 176 **sẽ được sửa có chủ ý** thành `== "finalize"` ở Task 9. Đó là thay đổi hợp đồng định trước, không phải hồi quy.
- 19 cảnh báo `multiprocessing` DeprecationWarning (9 ở unit, 10 ở integration) là do `fork()`. Task 1 sẽ xoá hết. **Số cảnh báo về 0 là tiêu chí nghiệm thu của Task 1**, không phải hiệu ứng phụ.
- **`ModelClient.__init__` giờ có thêm logic chế độ triển khai** (`deployment_mode()`/`egress_allowed()` từ `model/model_config.py`, nhánh Plan A merge vào sau khi Task 7 được thiết kế lần đầu — spec liên quan: `ADBA_DEPLOYMENT=onprem|hybrid`, mặc định `onprem` chặn mọi fallback OpenAI). Đây là một cơ chế **độc lập** với ngân sách thời gian: `deployment_mode` quyết fallback có được phép về mặt hợp đồng khách hàng; `deadline_ts` (Task 7) quyết có còn thời gian để thử không. Task 7 chèn `deadline_ts`/`clock` sau `openai_model` trong chữ ký `__init__` và gán `self.deadline_ts`/`self.clock` ở cuối thân hàm — vị trí đó vẫn đúng sau khi thân `__init__` dài thêm ~45 dòng vì đoạn deployment-mode, vì hai đoạn không đụng nhau. `_assert_budget()` được gọi trước `_invoke_ollama` trong `invoke()`, nên `BudgetExceededError` luôn thắng trước khi luồng chạm tới logic fallback theo `deployment_mode` — đúng ý đồ ban đầu (hết ngân sách không được đi vòng qua OpenAI, bất kể `hybrid` hay `onprem`).

---

## File Structure

**Tạo mới:**

| File | Trách nhiệm |
|---|---|
| `graph/budget.py` | Ngân sách thời gian: hằng số, `deadline_ts`, `time_left`, `has_room_for`. Hàm thuần, clock tiêm được. Không import gì từ `graph/`. |
| `graph/errors.py` | Tập nhãn lỗi hợp lệ. Một nguồn sự thật cho `last_error["error_type"]`. |
| `graph/agents/finalize.py` | Node cuối duy nhất. Gom kết quả đang có, phân loại `success`/`partial`/`failed`, ghi `degradation_reason`. |
| `scripts/create_readonly_role.sql` | Tạo role `adba_readonly`. Idempotent. |
| `tests/unit/test_budget.py` | |
| `tests/unit/test_python_sandbox_isolation.py` | |
| `tests/unit/test_finalize.py` | |
| `tests/unit/test_routing_is_pure.py` | |
| `tests/unit/test_trace_jsonl.py` | |
| `tests/integration/test_readonly_role.py` | Test duy nhất chạm Postgres thật. |
| `tests/integration/test_budget_end_to_end.py` | |

**Sửa:**

| File | Sửa gì |
|---|---|
| `graph/tools/python_tool.py` | `fork` → `spawn`; preset namespace thay vì truyền module/lambda qua ranh giới tiến trình; xoá env trước `exec` |
| `graph/agents/viz_agent.py` | `run_dict_safe` → `run_chart_safe`; bỏ 4 import chỉ phục vụ namespace cũ |
| `graph/agents/sql_agent.py` | `_extract_sql` fail closed; `MAX_RETRIES` 3 → 2; đếm `llm_calls_used`; nhận `deadline_ts` |
| `graph/tools/sql_tool.py` | `SQL_TIMEOUT_MS` 30000 → 10000 |
| `graph/state.py` | Thêm `query_id`, `deadline_ts`, `llm_calls_used`, `degradation_reason`; `make_initial_state` nhận `budget_s` |
| `graph/agents/supervisor.py` | `route_next_agent` thành hàm thuần; `routing_snapshot` tách ra; `MAX_REFLECTOR_PASSES_PER_AGENT` 8 → 1; đếm call; nhận deadline |
| `graph/agents/python_agent.py`, `viz_agent.py`, `insight_agent.py`, `reflector_agent.py` | Đếm `llm_calls_used`; nhận `deadline_ts`; nhãn lỗi từ `graph/errors.py` |
| `graph/multi_agent.py` | Thêm node `finalize`; mọi `END` → `finalize`; `run_graph` nhận `budget_s` |
| `graph/utils.py` | `append_trace` mang `query_id`/`llm_calls`/`deadline_remaining_ms`/`duration_ms`; ghi JSONL |
| `model/model_client.py` | Nhận `deadline_ts`; `BudgetExceededError`; siết `timeout_s` theo thời gian còn lại |
| `app.py` | Hiển thị `partial` và `degradation_reason` |

**Thứ tự:** an toàn (Task 1–3) → ngân sách (Task 4–11) → UI (Task 12). Ngược thứ tự pha trong spec, có chủ ý: ba task an toàn nhỏ và độc lập, nên nếu dừng giữa chừng thì L3 — lỗ hổng đáng sợ hơn ở mạng khách — đã đóng.

---

# PHẦN A — An toàn (spec pha 1, phần còn lại)

## Task 1: Sandbox — ranh giới là tiến trình, không phải namespace

Đây là task lớn nhất phần A, và **không phải đổi một chuỗi `"fork"` thành `"spawn"`**. Bản hiện tại truyền qua ranh giới tiến trình hai thứ mà `spawn` không pickle được:

- `result_picker` là **lambda đóng biến `df`** (`python_tool.py:162`) — `PicklingError`
- `viz_agent.py:115-116` truyền **module object** `matplotlib`, `plt`, `io`, `base64` vào `local_vars` — `TypeError: cannot pickle 'module' object`

`fork` che được cả hai vì child kế thừa memory. `spawn` thì không. Nên `preset` phải là một **chuỗi** và namespace phải được dựng **bên trong child**.

**Files:**
- Modify: `graph/tools/python_tool.py` (viết lại phần thực thi)
- Modify: `graph/agents/viz_agent.py:7-19, 113-117`
- Test: `tests/unit/test_python_sandbox_isolation.py` (tạo mới)
- Test: `tests/unit/test_python_agent.py` (phải pass không sửa)

**Interfaces:**
- Consumes: `pandas`, `numpy`, `scipy.stats`, `matplotlib`
- Produces:
  - `graph.tools.python_tool.run_pandas_safe(code: str, df: pd.DataFrame, timeout_seconds: float = PANDAS_EXEC_TIMEOUT_SECONDS) -> pd.DataFrame` — **chữ ký không đổi**
  - `graph.tools.python_tool.run_chart_safe(code: str, df: pd.DataFrame, timeout_seconds: float = PANDAS_EXEC_TIMEOUT_SECONDS) -> dict` — **thay** `run_dict_safe(code, local_vars)`
  - `graph.tools.python_tool.PRESET_PANDAS = "pandas"`, `PRESET_CHART = "chart"`
  - `graph.tools.python_tool.PYTHON_SAFE_NAMESPACE: dict` — giữ tên cũ, `tests/unit/test_python_agent.py:28` đang import

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_python_sandbox_isolation.py`:

```python
"""Sandbox: ranh giới là tiến trình.

Test rò rỉ env dùng `pd.io.common.os` — module `os` THẬT, lấy qua pandas
mà không đi qua `__import__`, nên whitelist import không nhìn thấy nó.
Đây chính là điểm yếu 1 trong spec mục 4.2, và nó là đường rò có thật
chứ không phải mẹo escape mong manh: đã kiểm trên pandas 2.3.3, truy được
34 biến môi trường của tiến trình cha trước bản sửa này.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.tools.python_tool import run_chart_safe, run_pandas_safe

DF = pd.DataFrame({"a": [1, 2, 3]})


def test_sandbox_cannot_read_parent_environment(monkeypatch):
    """Ràng buộc spec 4.2: dù thoát khỏi namespace, không đọc được credential."""
    monkeypatch.setenv("ADBA_SECRET_PROBE", "leaked-credential")
    code = (
        "import pandas as pd\n"
        "env = pd.io.common.os.environ\n"
        "result = {'probe': env.get('ADBA_SECRET_PROBE', '<absent>')}\n"
    )
    assert run_chart_safe(code, DF)["probe"] == "<absent>"


def test_sandbox_environment_is_emptied_not_just_filtered():
    """Xoá cả env, không phải chỉ giấu vài khoá đã biết tên."""
    code = (
        "import pandas as pd\n"
        "result = {'size': len(pd.io.common.os.environ)}\n"
    )
    assert run_chart_safe(code, DF)["size"] == 0


def test_sandbox_runs_in_a_separate_process():
    """PID của child khác PID của cha — namespace hạn chế không tự cho điều này."""
    code = (
        "import pandas as pd\n"
        "result = {'pid': pd.io.common.os.getpid()}\n"
    )
    assert run_chart_safe(code, DF)["pid"] != __import__("os").getpid()


def test_chart_namespace_provides_plt_io_base64_without_being_passed_them():
    """Module không pickle được, nên child phải tự import — không nhận qua dây."""
    code = (
        "fig = plt.figure()\n"
        "buf = io.BytesIO()\n"
        "fig.savefig(buf, format='png')\n"
        "plt.close(fig)\n"
        "result = {'chart_b64': base64.b64encode(buf.getvalue()).decode()}\n"
    )
    assert len(run_chart_safe(code, DF)["chart_b64"]) > 100


def test_pandas_preset_has_no_plt():
    """Preset pandas không được kéo matplotlib vào — mỗi preset đúng phạm vi nó cần."""
    with pytest.raises(RuntimeError):
        run_pandas_safe("df = df.copy()\ndf['x'] = plt.figure()\ndf", DF)


def test_timeout_still_kills_the_child():
    code = "import pandas as pd\nwhile True:\n    pass\n"
    with pytest.raises(RuntimeError, match="timed out"):
        run_pandas_safe(code, DF, timeout_seconds=2)
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_python_sandbox_isolation.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_chart_safe' from 'graph.tools.python_tool'`

- [ ] **Step 3: Viết lại phần thực thi của `graph/tools/python_tool.py`**

Thay toàn bộ nội dung từ dòng `def _sandbox_worker(` tới hết file, và sửa phần namespace ở trên nó. File sau khi sửa:

```python
"""Bộ thực thi Python bị giới hạn cho Data Analysis Agent.

Ranh giới là TIẾN TRÌNH, không phải namespace.

Namespace hạn chế bên dưới vẫn giữ làm phòng thủ nhiều lớp, nhưng nó
thoát được và đừng tin vào nó: `pd.io.common.os` là module `os` thật,
lấy qua pandas mà không đi qua `__import__`, nên whitelist không thấy.
Bịt từng hàm I/O của pandas là whack-a-mole (spec 4.2) — không làm.

Thứ thực sự giới hạn thiệt hại là hai điều dưới đây, và chúng đi cùng
nhau:

  1. `spawn` thay `fork` — child là interpreter mới, không kế thừa
     memory, file descriptor hay network namespace của cha.
  2. `os.environ.clear()` ngay trước `exec` — `spawn` cho phép làm điều
     mà `fork` không cho: code của model chạy trong một môi trường rỗng,
     nên dù thoát khỏi namespace nó cũng không còn `DATABASE_URL` để đọc.

Hệ quả về kiểu dữ liệu: mọi tham số đi qua ranh giới này phải pickle
được. Lambda và module object thì không. Đó là lý do preset là một CHUỖI
và namespace được dựng bên trong child, thay vì truyền callable/module
xuống như bản dùng `fork` trước đây.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)
PANDAS_EXEC_TIMEOUT_SECONDS = float(os.getenv("PANDAS_EXEC_TIMEOUT_SECONDS", "10"))

PRESET_PANDAS = "pandas"
PRESET_CHART = "chart"

# Whitelist of modules that can be imported inside the sandbox.
_ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "scipy", "scipy.stats",
})


def _restricted_import(name: str, *args, **kwargs):
    """Only allow imports of whitelisted modules (pandas, numpy, scipy)."""
    top_level = name.split(".")[0]
    if name in _ALLOWED_IMPORTS or top_level in _ALLOWED_IMPORTS:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"Module '{name}' is not available in the safe sandbox.")


_RESTRICTED_BUILTINS: dict[str, Any] = {
    "__import__": _restricted_import,
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "KeyError": KeyError,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "zip": zip,
}


def _base_namespace() -> dict[str, Any]:
    """Namespace cho phân tích pandas thuần."""
    return {
        "__builtins__": dict(_RESTRICTED_BUILTINS),
        "pd": pd,
        "np": np,
        "stats": stats,
    }


def _chart_namespace() -> dict[str, Any]:
    """Namespace cho vẽ biểu đồ.

    matplotlib import ở đây chứ không ở đầu file: chỉ tiến trình con vẽ
    biểu đồ mới cần nó, và tiến trình cha không còn lý do gì để nạp một
    thư viện đồ hoạ.
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace = _base_namespace()
    namespace.update({"matplotlib": matplotlib, "plt": plt, "io": io, "base64": base64})
    return namespace


_NAMESPACE_BUILDERS = {
    PRESET_PANDAS: _base_namespace,
    PRESET_CHART: _chart_namespace,
}

# Giữ tên cũ cho tương thích ngược — tests/unit/test_python_agent.py import nó.
PYTHON_SAFE_NAMESPACE: dict[str, Any] = _base_namespace()


def _pick_dataframe_result(local_vars: dict[str, Any], original_df: pd.DataFrame) -> pd.DataFrame:
    for key, val in reversed(list(local_vars.items())):
        if isinstance(val, pd.DataFrame) and key != "df":
            return val

    result_df = local_vars.get("df")
    if isinstance(result_df, pd.DataFrame):
        if result_df.shape != original_df.shape or not result_df.equals(original_df):
            return result_df

    raise RuntimeError(
        "Code did not produce a DataFrame result. "
        "Ensure the last expression evaluates to a DataFrame."
    )


def _pick_dict_result(local_vars: dict[str, Any]) -> dict:
    result = local_vars.get("result")
    if isinstance(result, dict):
        return result

    for val in reversed(list(local_vars.values())):
        if isinstance(val, dict):
            return val

    raise RuntimeError("Code did not produce a dict result.")


def _sandbox_worker(code: str, df: pd.DataFrame, preset: str, result_queue: mp.Queue) -> None:
    """Chạy trong tiến trình con. Dưới `spawn` đây là một interpreter mới.

    Mọi tham số phải pickle được — đó là lý do `preset` là chuỗi chứ không
    phải callable.
    """
    try:
        namespace = _NAMESPACE_BUILDERS[preset]()
        original = df.copy()
        local_vars: dict[str, Any] = {"df": df}

        # Xoá env NGAY TRƯỚC exec, không sớm hơn: các import phía trên
        # (matplotlib tìm thư mục cấu hình qua HOME/MPLCONFIGDIR) cần env
        # còn nguyên. Sau dòng này, code của model không còn DATABASE_URL
        # để đọc, kể cả khi nó lấy được module `os` qua pd.io.common.
        os.environ.clear()

        global_vars = dict(namespace)
        global_vars.update(local_vars)
        compiled = compile(code, "<python_sandbox>", "exec")
        exec(compiled, global_vars, local_vars)

        if preset == PRESET_PANDAS:
            result = _pick_dataframe_result(local_vars, original)
        else:
            result = _pick_dict_result(local_vars)
        result_queue.put(("ok", result))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def _run_sandboxed(
    code: str,
    df: pd.DataFrame,
    preset: str,
    timeout_seconds: float,
) -> Any:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_sandbox_worker, args=(code, df, preset, result_queue))
    proc.start()

    try:
        status, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        proc.terminate()
        proc.join()
        raise RuntimeError(f"Python execution timed out after {timeout_seconds:g}s") from exc

    proc.join(timeout=1)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    if status == "error":
        raise RuntimeError(f"Python execution failed: {payload}")
    return payload


def run_pandas_safe(
    code: str,
    df: pd.DataFrame,
    timeout_seconds: float = PANDAS_EXEC_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Execute user-supplied Python code in a restricted sandbox.

    Args:
        code: Python code string. The last expression must evaluate to a DataFrame.
        df: Input DataFrame available as the variable `df` inside the sandbox.

    Returns:
        The resulting DataFrame from the last expression.

    Raises:
        RuntimeError: on any execution failure.
    """
    return _run_sandboxed(code, df.copy(), PRESET_PANDAS, timeout_seconds)


def run_chart_safe(
    code: str,
    df: pd.DataFrame,
    timeout_seconds: float = PANDAS_EXEC_TIMEOUT_SECONDS,
) -> dict:
    """Chạy code vẽ biểu đồ, trả dict do code đặt vào biến `result`.

    Thay cho `run_dict_safe(code, local_vars)`: bản cũ nhận namespace tuỳ ý
    từ bên gọi, mà bên gọi duy nhất truyền vào bốn MODULE — thứ không pickle
    được nên không qua nổi ranh giới `spawn`. Giờ child tự dựng namespace
    từ preset.
    """
    return _run_sandboxed(code, df.copy(), PRESET_CHART, timeout_seconds)
```

- [ ] **Step 4: Sửa `graph/agents/viz_agent.py`**

Ở đầu file, xoá bốn import chỉ tồn tại để nhồi namespace cũ. `matplotlib.use("Agg")` ở tiến trình cha cũng bỏ — biểu đồ giờ vẽ trong child, và child tự gọi `use("Agg")`.

Xoá các dòng 7-8 và 13-16:

```python
import base64
import io
```

```python
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
```

Đổi import ở dòng 19:

```python
from graph.tools.python_tool import run_chart_safe
```

Đổi lời gọi ở dòng 113-117 thành:

```python
            result = run_chart_safe(code, df)
```

- [ ] **Step 5: Chạy test cô lập**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_python_sandbox_isolation.py -q`
Expected: PASS, 6 passed

- [ ] **Step 6: Chạy cả hai thư mục — cảnh báo `fork` phải về 0**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit -q`
Expected: PASS, 527 passed, 9 skipped. **Không còn dòng `DeprecationWarning` nào nhắc `os.fork()`.**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/integration -q`
Expected: PASS, 26 passed, 3 skipped, **0 warnings**.

`spawn` khởi động interpreter mới mỗi lần nên chậm hơn `fork` rõ rệt: mỗi lần vào sandbox tốn thêm khoảng 0,5–1,5 giây cho việc nạp lại pandas (và matplotlib với preset chart). `tests/unit/test_viz_agent.py` **không mock sandbox** — nó chạy code biểu đồ thật — nên suite unit sẽ tăng từ ~3,6s lên khoảng 15–25 giây. **Đó là mức chấp nhận được.** Nếu vượt 60 giây thì có gì đó khác đang sai, dừng và báo.

- [ ] **Step 7: Commit**

```bash
git add graph/tools/python_tool.py graph/agents/viz_agent.py tests/unit/test_python_sandbox_isolation.py
git commit -m "fix(sandbox): spawn + env rỗng — ranh giới là tiến trình, không phải namespace"
```

---

## Task 2: Trần tài nguyên SQL — fail closed, timeout 10s, cap 50 000 dòng

**Files:**
- Modify: `graph/agents/sql_agent.py:28-40, 140`
- Modify: `graph/tools/sql_tool.py:23, 105-129`
- Test: `tests/unit/test_sql_agent.py` (bổ sung)
- Test: `tests/unit/test_sql_tool_guard.py` (bổ sung)

**Interfaces:**
- Produces: `graph.agents.sql_agent._extract_sql(text: str) -> str` — raise `ValueError` thay vì trả nguyên văn output model
- Produces: `graph.tools.sql_tool.SQL_TIMEOUT_MS` mặc định `10000`
- Produces: `graph.tools.sql_tool.SQL_MAX_ROWS` mặc định `50000`
- Produces: `execute_sql(...) -> pd.DataFrame` với `df.attrs["truncated"]: bool` và `df.attrs["row_cap"]: int` — kiểu trả về **không đổi**, nên mọi lời gọi hiện có vẫn chạy

Đây là lớp 4 của spec 4.1, "trần tài nguyên". Ba việc, cùng một mục đích: chặn một câu truy vấn hỏng ăn hết ngân sách của cả lượt.

Guard `sqlparse` ở `assert_tables_permitted` đã chặn được câu rác ở tầng tool, nên fail-closed không phải lỗ hổng còn hở. Nhưng spec mục 4.1 ("Sửa kèm") yêu cầu fail closed **tại chỗ trích xuất**, và lý do là chẩn đoán: khi model trả văn xuôi, lỗi hiện ra phải là "không trích được SQL" chứ không phải "không parse được tên bảng" — hai lớp lỗi khác nhau, và nhãn sai làm reflector chẩn đoán sai.

Row cap thì là lỗ hổng thật và chưa bịt: `cur.fetchall()` ở `sql_tool.py:123` nạp trọn result set vào RAM. Một `SELECT *` trên bảng chục triệu dòng — đúng thứ một model đang bối rối hay sinh ra — làm sập tiến trình trước khi `statement_timeout` kịp có ý kiến, vì Postgres trả dòng đúng hạn, chỗ chết là phía client.

**Cố ý bỏ:** "giới hạn connection pool" trong spec 4.1 lớp 4. Không có pool nào để giới hạn — `execute_sql` mở và đóng một connection mỗi lời gọi (`sql_tool.py:118, 129`). Thêm pool để rồi giới hạn nó là dựng một thứ mới chỉ để có cái mà siết.

- [ ] **Step 1: Viết test thất bại**

Thêm vào cuối `tests/unit/test_sql_agent.py`:

```python
class TestExtractSqlFailsClosed:
    def test_prose_without_sql_raises(self):
        from graph.agents.sql_agent import _extract_sql

        with pytest.raises(ValueError, match="Không trích được SQL"):
            _extract_sql("Xin lỗi, tôi không đủ thông tin để viết câu truy vấn này.")

    def test_empty_output_raises(self):
        from graph.agents.sql_agent import _extract_sql

        with pytest.raises(ValueError, match="Không trích được SQL"):
            _extract_sql("   ")

    def test_a_real_select_still_comes_through(self):
        from graph.agents.sql_agent import _extract_sql

        assert _extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_a_with_cte_still_comes_through(self):
        from graph.agents.sql_agent import _extract_sql

        out = _extract_sql("Đây là câu trả lời:\nWITH t AS (SELECT 1) SELECT * FROM t")
        assert out.startswith("WITH")


class TestSqlTimeoutCeiling:
    def test_default_statement_timeout_is_ten_seconds(self):
        """Spec 4.1 lớp 4: 30s → 10s, nằm trong ngân sách wall-clock mục 5."""
        import importlib

        import graph.tools.sql_tool as sql_tool

        importlib.reload(sql_tool)
        assert sql_tool.SQL_TIMEOUT_MS == 10000
```

Thêm vào cuối `tests/unit/test_sql_tool_guard.py`:

```python
class TestRowCap:
    """Trần dòng (spec 4.1 lớp 4). Không chạm DB thật — psycopg2 bị mock."""

    @staticmethod
    def _profile():
        from perception.connection_profile import ALL_TABLES, build_profile
        from tests.fixtures.mini_schema import MINI_TABLES

        return build_profile(
            dsn="postgresql://u:p@h:5432/d",
            tables=MINI_TABLES,
            grants={"u": frozenset({ALL_TABLES})},
        )

    @staticmethod
    def _patch_connect(rows):
        """Mock psycopg2.connect sao cho fetchmany(n) tôn trọng n."""
        from unittest.mock import MagicMock, patch

        cur = MagicMock()
        cur.description = [("id",)]
        cur.fetchmany.side_effect = lambda n: rows[:n]
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        conn = MagicMock()
        conn.cursor.return_value = cur
        return patch("graph.tools.sql_tool.psycopg2.connect", return_value=conn)

    def test_all_rows_come_back_when_under_the_cap(self):
        from graph.tools.sql_tool import execute_sql

        rows = [{"id": i} for i in range(5)]
        with self._patch_connect(rows):
            df = execute_sql("SELECT id FROM orders", self._profile(), "u", max_rows=10)
        assert len(df) == 5
        assert df.attrs["truncated"] is False

    def test_rows_are_capped_and_the_flag_is_set(self):
        from graph.tools.sql_tool import execute_sql

        rows = [{"id": i} for i in range(50)]
        with self._patch_connect(rows):
            df = execute_sql("SELECT id FROM orders", self._profile(), "u", max_rows=10)
        assert len(df) == 10
        assert df.attrs["truncated"] is True

    def test_the_cap_defaults_to_fifty_thousand(self):
        import importlib

        import graph.tools.sql_tool as sql_tool

        importlib.reload(sql_tool)
        assert sql_tool.SQL_MAX_ROWS == 50000
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_sql_agent.py -q`
Expected: FAIL — `DID NOT RAISE <class 'ValueError'>`, `assert 30000 == 10000`, và `TypeError: execute_sql() got an unexpected keyword argument 'max_rows'`

Chạy cả `tests/unit/test_sql_tool_guard.py` trong cùng lệnh để thấy đủ ba lớp fail.

- [ ] **Step 3: Sửa `_extract_sql`**

Trong `graph/agents/sql_agent.py`, thay thân hàm `_extract_sql`:

```python
def _extract_sql(text: str) -> str:
    """Strip markdown fences and extract the SQL query from model output.

    Fail closed: khi không tìm thấy SELECT/WITH, RAISE thay vì trả nguyên
    văn output của model. Bản trước trả nguyên văn, và chuỗi đó đi thẳng
    tới `execute_sql`. Guard sqlparse ở tầng tool vẫn chặn được nó, nhưng
    lỗi hiện ra khi đó là "không parse được tên bảng" — sai lớp. Reflector
    chẩn đoán theo nhãn lỗi, nên nhãn sai dẫn tới `corrected_context` sai,
    và lần thử lại hỏng đúng như lần đầu.
    """
    text = re.sub(r"```(?:sql)?\s*|\s*```", "", text).strip()
    # Scan for the earliest SQL keyword (WITH may start before SELECT inside a CTE)
    earliest = len(text)
    for kw in _SQL_KEYWORDS:
        idx = text.find(kw)
        if 0 <= idx < earliest:
            earliest = idx
    if earliest < len(text):
        return text[earliest:].strip()
    raise ValueError(f"Không trích được SQL từ output của model: {text[:200]!r}")
```

- [ ] **Step 4: Bắt `ValueError` trong vòng lặp retry**

Trong `sql_agent_node`, dòng 140 hiện là `sql = _extract_sql(raw)` không được bảo vệ — giờ nó raise được, và một `ValueError` thoát ra sẽ làm sập node thay vì đi vào đường retry. Bọc lại:

```python
        try:
            sql = _extract_sql(raw)
        except ValueError as exc:
            error_context = str(exc)
            logger.warning("SQL Agent attempt %d: %s", attempt, error_context)
            trace = append_trace(state, "sql", "generate_sql",
                                 f"Attempt {attempt}: {error_context}", "error")
            continue

        logger.info("SQL Agent attempt %d:\n%s", attempt, sql)
```

- [ ] **Step 5: Hạ `SQL_TIMEOUT_MS` và thêm `SQL_MAX_ROWS`**

Trong `graph/tools/sql_tool.py` dòng 23:

```python
# 10s, không phải 30s: trần này nằm BÊN TRONG ngân sách wall-clock 45s
# (spec mục 5). Một câu truy vấn chạy quá 10 giây trên schema BI cỡ này
# gần như luôn là join sai, và để nó chạy tiếp chỉ ăn hết phần thời gian
# lẽ ra dành cho bước insight.
SQL_TIMEOUT_MS = int(os.getenv("SQL_TIMEOUT_MS", "10000"))

# Trần dòng nạp về client. statement_timeout không cứu được trường hợp này:
# Postgres trả dòng đúng hạn, chỗ chết là fetchall() phía client nạp trọn
# result set vào RAM. Một SELECT * trên bảng chục triệu dòng làm sập tiến
# trình trước khi có ai kịp báo lỗi.
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "50000"))
```

- [ ] **Step 6: Cắt kết quả ở trần dòng**

Thay thân `execute_sql` trong `graph/tools/sql_tool.py` (dòng 105-129):

```python
def execute_sql(
    sql: str,
    profile: ConnectionProfile,
    user: str,
    params: tuple | None = None,
    timeout_ms: int = SQL_TIMEOUT_MS,
    max_rows: int = SQL_MAX_ROWS,
) -> pd.DataFrame:
    """Chạy một câu SELECT, trả DataFrame.

    Kiểm quyền trước khi mở kết nối. Không có tham số nào cho phép bên gọi
    nới rộng tập bảng được chạm (spec tiêu chí 11).

    Kết quả bị cắt ở `max_rows`. Cờ `truncated` đi qua `df.attrs` chứ không
    qua kiểu trả về: đổi kiểu trả về sẽ buộc sql_agent, app.py và toàn bộ
    test hiện có phải sửa, đổi lấy một bit thông tin.
    """
    assert_tables_permitted(sql, profile, user)
    conn = psycopg2.connect(profile.dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (timeout_ms,))
            cur.execute(sql, params)
            # Lấy dư một dòng: đó là cách duy nhất phân biệt "đúng max_rows
            # dòng" với "còn nữa" mà không phải đếm cả bảng.
            rows = cur.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            if truncated:
                logger.warning(
                    "Kết quả bị cắt ở %d dòng — câu truy vấn trả về nhiều hơn thế.", max_rows
                )
                rows = rows[:max_rows]

            if not rows:
                columns = [desc[0] for desc in cur.description] if cur.description else []
                df = pd.DataFrame(columns=columns)
            else:
                df = pd.DataFrame(rows)

            df.attrs["truncated"] = truncated
            df.attrs["row_cap"] = max_rows
            return df
    finally:
        conn.close()
```

Trong `graph/agents/sql_agent.py`, cho người dùng biết khi kết quả bị cắt — dòng 166-167:

```python
        truncated_note = " (BỊ CẮT ở trần dòng)" if df.attrs.get("truncated") else ""
        trace = append_trace(state, "sql", "execute_sql",
                             f"OK — {len(df)} rows{truncated_note}, cost ~{cost}", "ok")
```

- [ ] **Step 7: Chạy test và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: PASS, 561 passed, 12 skipped

```bash
git add graph/agents/sql_agent.py graph/tools/sql_tool.py tests/unit/test_sql_agent.py tests/unit/test_sql_tool_guard.py
git commit -m "fix(sql): fail closed, statement_timeout 30s→10s, trần 50k dòng"
```

---

## Task 3: Role `adba_readonly` ở tầng Postgres

Lớp duy nhất thực sự là bảo đảm (spec 4.1 lớp 1). Ba lớp còn lại chỉ phát hiện sớm.

**Files:**
- Create: `scripts/create_readonly_role.sql`
- Create: `tests/integration/test_readonly_role.py`
- Modify: `docker-compose.yml` (chỉ thêm biến môi trường tài liệu hoá, không đổi service)

**Interfaces:**
- Produces: role `adba_readonly` trong `adba_db`; DSN dùng nó đặt ở `ADBA_READONLY_URL`

**Ràng buộc tài nguyên dùng chung — đọc trước khi chạy bất cứ lệnh nào:** container `adba-postgres` đang phục vụ worktree Plan A song song. Script dưới đây **chỉ tạo và cấp**, không thu hồi gì của `adba_user`, không `DROP`. `REVOKE` duy nhất trong script nhắm vào chính `adba_readonly`, không nhắm ai khác.

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/integration/test_readonly_role.py`:

```python
"""Role chỉ-đọc ở tầng Postgres — lớp bảo đảm duy nhất (spec 4.1 lớp 1).

Test này là test DUY NHẤT của Plan B chạm database thật. Nó skip khi
không có ADBA_READONLY_URL, nên bộ test vẫn chạy được trên máy không có
container.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RO_DSN = os.environ.get("ADBA_READONLY_URL")
pytestmark = pytest.mark.skipif(
    not RO_DSN, reason="cần ADBA_READONLY_URL sau khi chạy scripts/create_readonly_role.sql"
)


def _run(sql: str):
    conn = psycopg2.connect(RO_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall() if cur.description else None
    finally:
        conn.rollback()
        conn.close()


def test_select_works():
    assert _run("SELECT 1")[0][0] == 1


def test_insert_is_refused_by_the_database():
    with pytest.raises(psycopg2.Error):
        _run("INSERT INTO orders (id) VALUES (-1)")


def test_update_is_refused_by_the_database():
    with pytest.raises(psycopg2.Error):
        _run("UPDATE orders SET id = id")


def test_delete_is_refused_by_the_database():
    with pytest.raises(psycopg2.Error):
        _run("DELETE FROM orders")


def test_data_modifying_cte_is_refused_by_the_database():
    """Câu đi qua được heuristic 'first token ∈ {SELECT, WITH}' của bản cũ.

    Guard sqlparse đã chặn nó ở tầng ứng dụng. Test này kiểm điều khác:
    KỂ CẢ khi guard đó thủng, database vẫn từ chối.
    """
    with pytest.raises(psycopg2.Error):
        _run("WITH gone AS (DELETE FROM orders RETURNING *) SELECT count(*) FROM gone")


def test_create_table_is_refused_by_the_database():
    with pytest.raises(psycopg2.Error):
        _run("CREATE TABLE adba_probe_should_not_exist (id int)")


def test_the_transaction_is_read_only_by_default():
    assert _run("SHOW default_transaction_read_only")[0][0] == "on"
```

- [ ] **Step 2: Chạy test để xác nhận skip (chưa có role)**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/integration/test_readonly_role.py -q`
Expected: 7 skipped — `cần ADBA_READONLY_URL`

- [ ] **Step 3: Viết script tạo role**

Tạo `scripts/create_readonly_role.sql`:

```sql
-- Role chỉ-đọc cho đường production (spec 2026-08-12 mục 4.1, lớp 1).
--
-- Đây là lớp bảo đảm DUY NHẤT. Ba lớp phía ứng dụng (chặn multi-statement,
-- quét DML trên statement đã parse, trần tài nguyên) chỉ phát hiện sớm; nếu
-- cả ba thủng thì lớp này vẫn khiến DELETE báo lỗi thay vì chạy.
--
-- IDEMPOTENT: chạy lại nhiều lần không sao.
--
-- KHÔNG thu hồi bất cứ quyền nào của adba_user. adba_user vẫn cần quyền ghi
-- cho data/seed/seed_data.py, và container này đang phục vụ nhiều nhánh phát
-- triển song song — lấy quyền của nó đi là làm hỏng việc của người khác.

\set ON_ERROR_STOP on

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'adba_readonly') THEN
        EXECUTE format('CREATE ROLE adba_readonly LOGIN PASSWORD %L', :'ro_password');
    ELSE
        EXECUTE format('ALTER ROLE adba_readonly LOGIN PASSWORD %L', :'ro_password');
    END IF;
END
$$;

GRANT CONNECT ON DATABASE adba_db TO adba_readonly;
GRANT USAGE ON SCHEMA public TO adba_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO adba_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO adba_readonly;

-- Chỉ thu hồi của CHÍNH adba_readonly, không của ai khác.
REVOKE CREATE ON SCHEMA public FROM adba_readonly;

-- Cái chốt cuối: mọi transaction của role này mở ở chế độ chỉ đọc, nên cả
-- data-modifying CTE — thứ bắt đầu bằng WITH và qua được mọi heuristic
-- "token đầu tiên" — cũng bị Postgres từ chối.
ALTER ROLE adba_readonly SET default_transaction_read_only = on;
```

- [ ] **Step 4: Áp script vào container đang chạy**

Đọc lại ràng buộc tài nguyên dùng chung ở đầu task trước khi chạy.

```bash
docker exec -i -e PGPASSWORD=adba_password adba-postgres \
  psql -v ON_ERROR_STOP=1 -v ro_password=adba_readonly_pw \
       -U adba_user -d adba_db < scripts/create_readonly_role.sql
```

Expected: `GRANT` / `ALTER ROLE` in ra, không có `ERROR`.

- [ ] **Step 5: Chạy test với DSN chỉ-đọc**

```bash
ADBA_READONLY_URL="postgresql://adba_readonly:adba_readonly_pw@localhost:5432/adba_db" \
  "/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" \
  -m pytest tests/integration/test_readonly_role.py -q
```

Expected: PASS, 7 passed

- [ ] **Step 6: Ghi lại cách dùng vào `docker-compose.yml`**

Thêm vào cuối `docker-compose.yml`, sau khối `volumes:`:

```yaml
# Role chỉ-đọc không dựng được bằng biến môi trường của image postgres —
# nó cần chạy sau khi schema tồn tại. Sau khi container healthy:
#
#   docker exec -i -e PGPASSWORD=$POSTGRES_PASSWORD adba-postgres \
#     psql -v ON_ERROR_STOP=1 -v ro_password="$ADBA_READONLY_PASSWORD" \
#          -U $POSTGRES_USER -d $POSTGRES_DB < scripts/create_readonly_role.sql
#
# Rồi trỏ ứng dụng vào role đó:
#   ADBA_READONLY_URL=postgresql://adba_readonly:<pw>@localhost:5432/adba_db
#
# adba_user giữ nguyên quyền ghi, và chỉ dùng cho data/seed/seed_data.py.
```

- [ ] **Step 7: Chạy cả bộ test và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: PASS, 561 passed, 19 skipped (12 nền, cần `DATABASE_URL` + 7 readonly, cần `ADBA_READONLY_URL`, vì không đặt biến nào trong lần chạy trần)

```bash
git add scripts/create_readonly_role.sql tests/integration/test_readonly_role.py docker-compose.yml
git commit -m "feat(db): role adba_readonly — lớp bảo đảm chỉ-đọc ở tầng Postgres"
```

---

# PHẦN B — Ngân sách thời gian (spec pha 2)

## Task 4: `graph/budget.py` và các trường ngân sách trong state

Nền của cả phần B. Hàm thuần, clock tiêm vào — nhờ vậy mọi test deadline chạy tức thời, không `sleep`.

**Files:**
- Create: `graph/budget.py`
- Create: `tests/unit/test_budget.py`
- Modify: `graph/state.py`

**Interfaces:**
- Produces:
  - `graph.budget.DEFAULT_BUDGET_S: float` (env `ADBA_QUERY_BUDGET_S`, mặc định 45)
  - `graph.budget.INSIGHT_RESERVE_S: float` (env `ADBA_INSIGHT_RESERVE_S`, mặc định 12)
  - `graph.budget.MAX_LLM_CALLS_PER_QUERY: int` (env `ADBA_MAX_LLM_CALLS`, mặc định 12)
  - `graph.budget.MODEL_CALL_ESTIMATE_S: float` (env `ADBA_MODEL_CALL_ESTIMATE_S`, mặc định 15)
  - `graph.budget.new_query_id() -> str`
  - `graph.budget.make_deadline(budget_s: float = DEFAULT_BUDGET_S, clock: Clock = time.time) -> float`
  - `graph.budget.time_left(deadline_ts: float, clock: Clock = time.time) -> float`
  - `graph.budget.is_expired(deadline_ts: float, clock: Clock = time.time) -> bool`
  - `graph.budget.has_room_for(deadline_ts: float, needed_s: float, *, reserve_s: float = 0.0, clock: Clock = time.time) -> bool`
  - `graph.budget.calls_exhausted(llm_calls_used: int) -> bool`
- Produces: `graph.state.make_initial_state(query, schema_context, budget_s=DEFAULT_BUDGET_S, clock=time.time)` — hai tham số mới đều có mặc định, nên mọi lời gọi hiện có vẫn chạy

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_budget.py`:

```python
"""Ngân sách thời gian — hàm thuần, clock tiêm vào.

Không test nào ở đây được dùng time.sleep. Nếu phải sleep để test một
deadline thì deadline đó đang đọc đồng hồ ở chỗ không tiêm được, và đó
mới là lỗi cần sửa.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.budget import (
    INSIGHT_RESERVE_S,
    MAX_LLM_CALLS_PER_QUERY,
    calls_exhausted,
    has_room_for,
    is_expired,
    make_deadline,
    new_query_id,
    time_left,
)
from graph.state import make_initial_state
from perception.schema_context import SchemaContext


def fake_clock(*ticks: float):
    """Đồng hồ trả lần lượt các mốc cho trước, giữ nguyên mốc cuối."""
    seq = list(ticks)

    def clock() -> float:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return clock


# ── deadline ────────────────────────────────────────────────────────────────

def test_deadline_is_now_plus_budget():
    assert make_deadline(45, clock=fake_clock(1000.0)) == 1045.0


def test_time_left_shrinks_as_the_clock_advances():
    assert time_left(1045.0, clock=fake_clock(1030.0)) == 15.0


def test_time_left_goes_negative_past_the_deadline():
    assert time_left(1045.0, clock=fake_clock(1060.0)) == -15.0


def test_not_expired_before_the_deadline():
    assert is_expired(1045.0, clock=fake_clock(1044.9)) is False


def test_expired_exactly_at_the_deadline():
    """Đúng mốc là hết. Biên nghiêng về phía an toàn."""
    assert is_expired(1045.0, clock=fake_clock(1045.0)) is True


# ── dự trữ ──────────────────────────────────────────────────────────────────

def test_room_when_plenty_of_time_left():
    assert has_room_for(1100.0, needed_s=15, clock=fake_clock(1000.0)) is True


def test_no_room_when_the_work_would_overrun():
    assert has_room_for(1010.0, needed_s=15, clock=fake_clock(1000.0)) is False


def test_reserve_is_subtracted_before_the_decision():
    """40s còn lại, cần 15s, dự trữ 30s → không đủ chỗ.

    Đây là quy tắc bảo vệ bước insight: node python/viz bị cắt trước, để
    người dùng luôn nhận được nhận xét nếu SQL đã chạy được (spec 5.2).
    """
    assert has_room_for(1040.0, needed_s=15, reserve_s=30, clock=fake_clock(1000.0)) is False


def test_without_a_reserve_the_same_call_has_room():
    assert has_room_for(1040.0, needed_s=15, clock=fake_clock(1000.0)) is True


def test_insight_reserve_defaults_to_twelve_seconds():
    assert INSIGHT_RESERVE_S == 12


# ── trần cứng số lời gọi ────────────────────────────────────────────────────

def test_call_ceiling_is_twelve():
    assert MAX_LLM_CALLS_PER_QUERY == 12


def test_calls_not_exhausted_below_the_ceiling():
    assert calls_exhausted(11) is False


def test_calls_exhausted_at_the_ceiling():
    assert calls_exhausted(12) is True


# ── query_id ────────────────────────────────────────────────────────────────

def test_query_ids_are_unique():
    assert len({new_query_id() for _ in range(100)}) == 100


# ── state ───────────────────────────────────────────────────────────────────

def test_initial_state_carries_a_deadline():
    s = make_initial_state("q", SchemaContext(), budget_s=45, clock=fake_clock(1000.0))
    assert s["deadline_ts"] == 1045.0


def test_initial_state_carries_a_query_id():
    s = make_initial_state("q", SchemaContext())
    assert s["query_id"]


def test_initial_state_starts_with_zero_llm_calls():
    assert make_initial_state("q", SchemaContext())["llm_calls_used"] == 0


def test_initial_state_starts_with_no_degradation():
    assert make_initial_state("q", SchemaContext())["degradation_reason"] == []


def test_old_two_argument_call_still_works():
    """Hàng chục test hiện có gọi make_initial_state(query, ctx). Đừng làm hỏng."""
    s = make_initial_state("q", SchemaContext())
    assert s["query"] == "q"
    assert s["deadline_ts"] > 0
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_budget.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'graph.budget'`

- [ ] **Step 3: Viết `graph/budget.py`**

```python
"""Ngân sách thời gian cho một lượt truy vấn.

Toàn bộ module là hàm thuần với đồng hồ tiêm vào. Lý do không phải là
thẩm mỹ: deadline chỉ kiểm được bằng test nhanh khi đồng hồ tiêm được,
và một bộ test phải `sleep` để kiểm ngân sách thì sẽ bị người ta tắt đi.

Module này KHÔNG import gì từ `graph/` — nó nằm dưới đáy cây phụ thuộc
để mọi node và cả `model/` dùng được mà không tạo vòng import.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable

Clock = Callable[[], float]

# 45s là SLO trong spec mục 1, đo ở cấu hình vLLM/PEFT (12,4s mỗi lời gọi
# model). Trên Ollama q5 — 75,8s mỗi lời gọi — con số này không đạt được,
# và đó là lý do nó có env override thay vì bị viết cứng: dev đặt
# ADBA_QUERY_BUDGET_S=400, production giữ 45.
DEFAULT_BUDGET_S = float(os.getenv("ADBA_QUERY_BUDGET_S", "45"))

# Phần cuối ngân sách giữ riêng cho bước insight. Chia đều sẽ bỏ đói bước
# cuối, mà bước cuối lại là thứ người dùng thực sự đọc (spec 5.2).
INSIGHT_RESERVE_S = float(os.getenv("ADBA_INSIGHT_RESERVE_S", "12"))

# Trần cứng, độc lập với thời gian. Deadline ép SLO; trần này chặn vòng
# lặp vô hạn khi đồng hồ vì lý do nào đó không cứu được (spec 5.4).
MAX_LLM_CALLS_PER_QUERY = int(os.getenv("ADBA_MAX_LLM_CALLS", "12"))

# Ước lượng thời gian một lời gọi model, dùng để quyết định CÓ khởi động
# lời gọi hay không. Cố ý bi quan: bỏ qua một node còn hơn khởi động một
# lời gọi chắc chắn không kịp về, vì lời gọi đó vẫn tiêu hết phần thời
# gian còn lại rồi mới thất bại.
MODEL_CALL_ESTIMATE_S = float(os.getenv("ADBA_MODEL_CALL_ESTIMATE_S", "15"))


def new_query_id() -> str:
    """Khoá vòng đời của một lượt truy vấn — dùng cho trace và log."""
    return uuid.uuid4().hex[:12]


def make_deadline(budget_s: float = DEFAULT_BUDGET_S, clock: Clock = time.time) -> float:
    return clock() + budget_s


def time_left(deadline_ts: float, clock: Clock = time.time) -> float:
    return deadline_ts - clock()


def is_expired(deadline_ts: float, clock: Clock = time.time) -> bool:
    """Đúng mốc deadline đã tính là hết — biên nghiêng về phía an toàn."""
    return time_left(deadline_ts, clock) <= 0


def has_room_for(
    deadline_ts: float,
    needed_s: float,
    *,
    reserve_s: float = 0.0,
    clock: Clock = time.time,
) -> bool:
    """Còn đủ chỗ cho một việc tốn `needed_s` mà không lấn vào phần dự trữ?"""
    return time_left(deadline_ts, clock) - reserve_s >= needed_s


def calls_exhausted(llm_calls_used: int) -> bool:
    return llm_calls_used >= MAX_LLM_CALLS_PER_QUERY
```

- [ ] **Step 4: Thêm trường vào `graph/state.py`**

Thêm import ở đầu file:

```python
import time

from graph.budget import DEFAULT_BUDGET_S, Clock, make_deadline, new_query_id
```

Thêm khối trường mới vào `MultiAgentState`, ngay sau khối `# ── Input ──`:

```python
    # ── Ngân sách ──────────────────────────────────────────
    query_id: str
    deadline_ts: float
    llm_calls_used: int
    degradation_reason: list[str]
```

Đổi chữ ký và thân `make_initial_state`:

```python
def make_initial_state(
    query: str,
    schema_context: SchemaContext,
    budget_s: float = DEFAULT_BUDGET_S,
    clock: Clock = time.time,
) -> MultiAgentState:
    """Return a fresh state dict for a new query.

    `budget_s` và `clock` đều có mặc định, nên mọi lời gọi hai tham số
    hiện có vẫn chạy nguyên. `clock` tồn tại để test deadline không phải
    sleep.
    """
    return MultiAgentState(
        query=query,
        schema_context=schema_context,
        query_id=new_query_id(),
        deadline_ts=make_deadline(budget_s, clock),
        llm_calls_used=0,
        degradation_reason=[],
        execution_plan=[],
        dependency_graph={},
        ready_agents=[],
        current_agent="supervisor",
        completed_agents=[],
        agent_outputs={},
        shared_dataframe=None,
        shared_metadata={},
        error_count=0,
        agent_error_counts={},
        last_error=None,
        sql_result=None,
        python_result=None,
        chart_b64=None,
        chart_metadata=None,
        insight=None,
        action_trace=[],
        status="planning",
    )
```

- [ ] **Step 5: Chạy test để xác nhận pass**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_budget.py -q`
Expected: PASS, 19 passed

- [ ] **Step 6: Chạy cả bộ và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: PASS, 580 passed, 19 skipped

```bash
git add graph/budget.py graph/state.py tests/unit/test_budget.py
git commit -m "feat(budget): deadline_ts mang trong state, clock tiêm được"
```

---

## Task 5: `route_next_agent` thành hàm thuần (bug spec 5.5)

`route_next_agent` hiện ghi vào `state["dependency_graph"]`, `state["ready_agents"]` và `state["shared_metadata"]` (supervisor.py:284-291) **bên trong một conditional-edge function**. LangGraph coi conditional edge là hàm thuần trả về tên route; những gán này không đi qua reducer nên không đảm bảo persist. Chúng đang là bug im lặng: state trông như được cập nhật khi đọc code, nhưng không có gì bảo đảm điều đó.

**Files:**
- Modify: `graph/agents/supervisor.py:212-298`
- Modify: `graph/utils.py` (thêm `with_routing`)
- Modify: `graph/agents/sql_agent.py`, `python_agent.py`, `viz_agent.py` (mỗi file một dòng)
- Test: `tests/unit/test_routing_is_pure.py` (tạo mới)

**Interfaces:**
- Produces: `graph.agents.supervisor.routing_snapshot(state: MultiAgentState) -> dict[str, Any]` — trả `{"dependency_graph": ..., "ready_agents": [...], "blocked_agents": [...]}`, không ghi gì
- Produces: `graph.utils.with_routing(state: MultiAgentState) -> MultiAgentState` — trả state mới có snapshot đã gắn
- `route_next_agent` giữ nguyên chữ ký, **không còn gán vào `state`**

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_routing_is_pure.py`:

```python
"""Conditional edge phải là hàm thuần (spec 5.5).

LangGraph gọi conditional edge để lấy TÊN ROUTE. Gán vào state bên trong
nó không đi qua reducer, nên thay đổi có thể mất — mà mất im lặng, vì
đọc code thì thấy như đã cập nhật.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.supervisor import route_next_agent, routing_snapshot
from graph.state import make_initial_state
from graph.utils import with_routing
from perception.schema_context import SchemaContext

PLAN = [
    {"step": 1, "agent": "sql", "task": "t1", "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "insight", "task": "t2", "depends_on": ["sql"],
     "skill_type": "insight-generation"},
]


def _state(**over):
    s = make_initial_state("q", SchemaContext())
    s["execution_plan"] = PLAN
    s["completed_agents"] = ["supervisor"]
    s["status"] = "running"
    s.update(over)
    return s


def test_route_does_not_mutate_the_state_it_is_given():
    s = _state()
    before = copy.deepcopy(s)
    route_next_agent(s)
    assert s == before, "conditional edge đã ghi vào state — xem spec 5.5"


def test_route_still_returns_the_right_agent():
    assert route_next_agent(_state()) == "sql"


def test_route_returns_insight_after_sql_completes():
    assert route_next_agent(_state(completed_agents=["supervisor", "sql"])) == "insight"


def test_snapshot_reports_ready_agents_without_writing():
    s = _state()
    before = copy.deepcopy(s)
    snap = routing_snapshot(s)
    assert snap["ready_agents"] == ["sql"]
    assert s == before


def test_snapshot_reports_the_dependency_graph():
    assert routing_snapshot(_state())["dependency_graph"] == {"sql": [], "insight": ["sql"]}


def test_snapshot_reports_blocked_agents():
    s = _state(
        agent_outputs={"sql": {"status": "error"}, "reflector": {"status": "ok"}},
        agent_error_counts={"sql": 5},
        shared_metadata={
            "reflect_passes_per_agent": {"sql": 9},
            "reflect_error_snapshot": {"sql": 5},
        },
    )
    assert "sql" in routing_snapshot(s)["blocked_agents"]


def test_with_routing_writes_the_snapshot_into_the_returned_state():
    """Node ghi — không phải conditional edge. Đây là chỗ ghi hợp lệ."""
    out = with_routing(_state())
    assert out["ready_agents"] == ["sql"]
    assert out["dependency_graph"] == {"sql": [], "insight": ["sql"]}
    assert out["shared_metadata"]["parallel_ready"] is False


def test_with_routing_does_not_mutate_its_input():
    s = _state()
    before = copy.deepcopy(s)
    with_routing(s)
    assert s == before
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_routing_is_pure.py -q`
Expected: FAIL — `ImportError: cannot import name 'routing_snapshot'`

- [ ] **Step 3: Tách `routing_snapshot` trong `graph/agents/supervisor.py`**

Thay toàn bộ `route_next_agent` (dòng 212-298) bằng hai hàm dưới đây:

```python
def _blocked_agents(state: MultiAgentState) -> tuple[set[str], str | None]:
    """Agent nào đang kẹt lỗi, và có cần gọi reflector ngay không.

    Trả `(blocked, "reflector" | None)`. Hàm thuần — không ghi gì.
    """
    plan = state.get("execution_plan", [])
    completed = set(state.get("completed_agents", []))
    error_counts = state.get("agent_error_counts", {})
    agent_outputs = state.get("agent_outputs", {})
    meta = state.get("shared_metadata", {})
    passes_map: dict[str, int] = dict(meta.get("reflect_passes_per_agent", {}))
    snapshot: dict[str, int] = dict(meta.get("reflect_error_snapshot", {}))

    blocked: set[str] = set()
    for step in plan:
        agent: str = step.get("agent", "")
        if agent in completed:
            continue

        seen_at_reflect = snapshot.get(agent, -1)
        curr = error_counts.get(agent, 0)
        passes = passes_map.get(agent, 0)

        specialist_err = agent_outputs.get(agent, {}).get("status") == "error"
        error_budget = curr >= 3

        if specialist_err or error_budget:
            stale = curr <= seen_at_reflect
            needs_reflector = ("reflector" not in agent_outputs) or (not stale)

            if needs_reflector and passes < MAX_REFLECTOR_PASSES_PER_AGENT:
                return blocked, "reflector"

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
            blocked.add(agent)

    return blocked, None


def routing_snapshot(state: MultiAgentState) -> dict[str, Any]:
    """Tính dependency_graph / ready_agents / blocked_agents — HÀM THUẦN.

    Node gọi hàm này rồi ghi kết quả vào state nó TRẢ VỀ; conditional edge
    chỉ đọc. Xem docstring của route_next_agent về lý do phân vai này.
    """
    plan = state.get("execution_plan", [])
    completed = list(state.get("completed_agents", []))
    try:
        dependency_graph = validate_dependency_graph(plan)
    except Exception as exc:
        logger.error("Invalid dependency graph: %s", exc)
        return {"dependency_graph": {}, "ready_agents": [], "blocked_agents": []}

    blocked, _ = _blocked_agents(state)
    return {
        "dependency_graph": dependency_graph,
        "ready_agents": resolve_ready_agents(plan, completed, blocked),
        "blocked_agents": sorted(blocked),
    }


def route_next_agent(state: MultiAgentState) -> str:
    """Conditional edge — quyết định node nào chạy tiếp. KHÔNG ghi vào state.

    Bản trước gán state["dependency_graph"], state["ready_agents"] và
    state["shared_metadata"] ngay tại đây. LangGraph coi conditional edge
    là hàm thuần trả về tên route: những gán đó không đi qua reducer nên
    không có gì bảo đảm chúng persist — một bug im lặng, vì đọc code thì
    thấy như đã cập nhật (spec 5.5). Việc ghi giờ thuộc về node, qua
    graph.utils.with_routing.

    Rules:
      1. status == 'failed' hoặc 'success' → finalize.
      2. Agent kẹt lỗi → reflector, tới khi hết ngân sách reflector.
      3. Agent sẵn sàng đầu tiên theo thứ tự step.
      4. Không còn gì chạy được → finalize.
    """
    if state["status"] in {"failed", "success"}:
        return "finalize"

    plan = state.get("execution_plan", [])
    try:
        validate_dependency_graph(plan)
    except Exception as exc:
        logger.error("Invalid dependency graph: %s", exc)
        return "finalize"

    blocked, reflector = _blocked_agents(state)
    if reflector is not None:
        return reflector

    for agent in resolve_ready_agents(plan, list(state.get("completed_agents", [])), blocked):
        if agent in SPECIALIST_AGENTS:
            return agent

    return "finalize"
```

Xoá `from typing import Any, Literal` cũ và thay bằng `from typing import Any` nếu `Literal` không còn dùng ở đâu khác trong file.

- [ ] **Step 4: Thêm `with_routing` vào `graph/utils.py`**

Thêm vào cuối file:

```python
def with_routing(state: MultiAgentState) -> MultiAgentState:
    """Gắn ảnh chụp định tuyến vào state mà node sắp trả về.

    Đây là chỗ ghi HỢP LỆ cho dependency_graph/ready_agents: node trả state
    qua reducer của LangGraph, conditional edge thì không (spec 5.5).
    Import cục bộ để tránh vòng import graph.utils ↔ graph.agents.supervisor.
    """
    from graph.agents.supervisor import routing_snapshot

    snapshot = routing_snapshot(state)
    return {
        **state,
        "dependency_graph": snapshot["dependency_graph"],
        "ready_agents": snapshot["ready_agents"],
        "shared_metadata": {
            **state.get("shared_metadata", {}),
            "dependency_graph": snapshot["dependency_graph"],
            "ready_agents": snapshot["ready_agents"],
            "parallel_ready": len(snapshot["ready_agents"]) > 1,
        },
    }
```

- [ ] **Step 5: Gọi `with_routing` ở ba node specialist**

Trong `graph/agents/sql_agent.py`, `python_agent.py`, `viz_agent.py`: thêm `with_routing` vào import từ `graph.utils`, rồi bọc **câu `return` trên đường thành công** của mỗi node.

`sql_agent.py` dòng 170 — `return {` trở thành `return with_routing({`, và dấu `}` đóng ở dòng 188 trở thành `})`. Làm y hệt cho `python_agent.py` và `viz_agent.py`.

Supervisor đã tự ghi `dependency_graph`/`ready_agents` ở dòng 197-198 nên không cần đổi.

- [ ] **Step 6: Chạy test**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_routing_is_pure.py -q`
Expected: PASS, 8 passed

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: 584 passed, **4 failed**, 19 skipped. Bốn fail nằm đúng ở `tests/unit/test_supervisor.py` dòng 127/132/142/176 — `assert 'finalize' == END`. **Đây là kỳ vọng, không phải hồi quy**; Task 9 sửa chúng khi node `finalize` tồn tại. Fail ở bất kỳ dòng nào khác thì dừng lại.

- [ ] **Step 7: Commit**

```bash
git add graph/agents/supervisor.py graph/utils.py graph/agents/sql_agent.py graph/agents/python_agent.py graph/agents/viz_agent.py tests/unit/test_routing_is_pure.py
git commit -m "fix(routing): conditional edge thành hàm thuần; node giữ việc ghi state (spec 5.5)"
```

---

## Task 6: Trần cứng thay cho đếm retry

**Files:**
- Modify: `graph/agents/supervisor.py:23`
- Modify: `graph/agents/sql_agent.py:23`
- Modify: `graph/agents/supervisor.py` (`route_next_agent` kiểm trần call)
- Test: `tests/unit/test_budget.py` (bổ sung)

**Interfaces:**
- Consumes: `graph.budget.calls_exhausted`
- Produces: `MAX_REFLECTOR_PASSES_PER_AGENT = 1`, `graph.agents.sql_agent.MAX_RETRIES = 2`

- [ ] **Step 1: Viết test thất bại**

Thêm vào cuối `tests/unit/test_budget.py`:

```python
def test_reflector_cap_is_one():
    """Spec 5.4: reflector sinh corrected_context. Một chẩn đoán + một lần
    thử lại không sửa được thì bảy lần nữa cũng vậy — model kẹt ở cùng lớp
    lỗi, và reflector_json_rate = 100% cho thấy nó không hỏng ở khâu sinh
    output."""
    from graph.agents.supervisor import MAX_REFLECTOR_PASSES_PER_AGENT

    assert MAX_REFLECTOR_PASSES_PER_AGENT == 1


def test_sql_node_retry_is_two():
    """Một lần sinh + một lần sửa theo error context."""
    from graph.agents.sql_agent import MAX_RETRIES

    assert MAX_RETRIES == 2


def test_router_goes_to_finalize_when_the_call_ceiling_is_hit():
    from graph.agents.supervisor import route_next_agent

    s = _state_for_ceiling()
    s["llm_calls_used"] = 12
    assert route_next_agent(s) == "finalize"


def test_router_still_routes_below_the_call_ceiling():
    from graph.agents.supervisor import route_next_agent

    s = _state_for_ceiling()
    s["llm_calls_used"] = 11
    assert route_next_agent(s) == "sql"


def _state_for_ceiling():
    from graph.state import make_initial_state
    from perception.schema_context import SchemaContext

    s = make_initial_state("q", SchemaContext())
    s["execution_plan"] = [
        {"step": 1, "agent": "sql", "task": "t", "depends_on": [], "skill_type": "text-to-sql"},
    ]
    s["completed_agents"] = ["supervisor"]
    s["status"] = "running"
    return s
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_budget.py -q`
Expected: FAIL — `assert 8 == 1`, `assert 3 == 2`, `assert 'sql' == 'finalize'`

- [ ] **Step 3: Hạ hai hằng số**

`graph/agents/supervisor.py` dòng 20-23:

```python
# Số lần reflector được gọi cho cùng một specialist. Bằng 1, không phải 8:
# reflector sinh `corrected_context`, và một chẩn đoán cộng một lần thử lại
# không sửa được thì bảy lần nữa cũng vậy — model kẹt ở cùng lớp lỗi.
# `reflector_json_rate = 100%` cho thấy reflector không hỏng ở khâu sinh
# output, nên lặp thêm chỉ đốt ngân sách (spec 5.4).
MAX_REFLECTOR_PASSES_PER_AGENT = 1
```

`graph/agents/sql_agent.py` dòng 23:

```python
# Một lần sinh + một lần sửa theo error context. Lần thứ ba trước đây chỉ
# lặp lại cùng một lớp lỗi với giá một lời gọi model (spec 5.4).
MAX_RETRIES = 2
```

- [ ] **Step 4: Kiểm trần call trong router**

Trong `graph/agents/supervisor.py`, thêm import:

```python
from graph.budget import calls_exhausted
```

Trong `route_next_agent`, ngay sau khối kiểm `status`:

```python
    if calls_exhausted(state.get("llm_calls_used", 0)):
        logger.warning(
            "Trần %d lời gọi model đã chạm — đi thẳng tới finalize",
            state.get("llm_calls_used", 0),
        )
        return "finalize"
```

- [ ] **Step 5: Chạy test và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_budget.py -q`
Expected: PASS, 23 passed

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: 588 passed, 4 failed, 19 skipped — vẫn đúng 4 fail đã biết ở `test_supervisor.py` (Task 9 sửa). Không fail mới.

```bash
git add graph/agents/supervisor.py graph/agents/sql_agent.py tests/unit/test_budget.py
git commit -m "feat(budget): trần cứng thay đếm retry — reflector 8→1, sql retry 3→2, trần 12 call"
```

---

## Task 7: `ModelClient` nhận deadline

Điểm kiểm thứ hai trong ba điểm của spec 5.1: **trước mỗi lời gọi model**, thời gian còn lại nhỏ hơn ước lượng thì không khởi động. Khởi động một lời gọi chắc chắn không kịp về là cách tệ nhất tiêu ngân sách: nó vẫn đốt hết phần còn lại rồi mới thất bại.

**Files:**
- Modify: `model/model_client.py`
- Test: `tests/unit/test_model_client_deadline.py` (tạo mới)

**Interfaces:**
- Consumes: `graph.budget.MODEL_CALL_ESTIMATE_S`, `has_room_for`, `time_left`, `Clock`
- Produces:
  - `model.model_client.BudgetExceededError(RuntimeError)`
  - `ModelClient(agent_type, ..., deadline_ts: float | None = None, clock: Clock = time.time)` — `deadline_ts=None` giữ nguyên hành vi cũ

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_model_client_deadline.py`:

```python
"""ModelClient tôn trọng deadline (spec 5.1, điểm kiểm 2)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.model_client import BudgetExceededError, ModelClient


def clock_at(t: float):
    return lambda: t


def test_no_deadline_means_no_budget_check():
    """Mặc định giữ nguyên hành vi cũ — hàng chục test hiện có dựa vào đó."""
    with patch("model.model_client.ollama.Client") as fake:
        fake.return_value.chat.return_value = {"message": {"content": "ok"}}
        assert ModelClient(agent_type="sql").invoke("s", "u") == "ok"


def test_call_is_refused_when_not_enough_time_remains():
    with patch("model.model_client.ollama.Client") as fake:
        client = ModelClient(agent_type="sql", deadline_ts=1005.0, clock=clock_at(1000.0))
        with pytest.raises(BudgetExceededError, match="ngân sách"):
            client.invoke("s", "u")
        fake.return_value.chat.assert_not_called()


def test_call_proceeds_when_there_is_room():
    with patch("model.model_client.ollama.Client") as fake:
        fake.return_value.chat.return_value = {"message": {"content": "ok"}}
        client = ModelClient(agent_type="sql", deadline_ts=1100.0, clock=clock_at(1000.0))
        assert client.invoke("s", "u") == "ok"


def test_call_is_refused_after_the_deadline_has_passed():
    with patch("model.model_client.ollama.Client") as fake:
        client = ModelClient(agent_type="sql", deadline_ts=900.0, clock=clock_at(1000.0))
        with pytest.raises(BudgetExceededError):
            client.invoke("s", "u")
        fake.return_value.chat.assert_not_called()


def test_timeout_is_clamped_to_the_time_actually_left():
    """AGENT_TIMEOUT_S['sql'] là 200s. Còn 30s thì timeout phải là 30, không phải 200."""
    with patch("model.model_client.ollama.Client"):
        client = ModelClient(agent_type="sql", deadline_ts=1030.0, clock=clock_at(1000.0))
        assert client.effective_timeout_s() == pytest.approx(30.0)


def test_timeout_is_unclamped_without_a_deadline():
    with patch("model.model_client.ollama.Client"):
        assert ModelClient(agent_type="sql").effective_timeout_s() == 200


def test_budget_error_does_not_trigger_the_openai_fallback():
    """Hết ngân sách không phải lỗi của Ollama. Gọi sang OpenAI chỉ tiêu thêm
    thời gian mình không có, và làm schema rời khỏi mạng khách."""
    with patch("model.model_client.ollama.Client"), \
         patch("model.model_client.ModelClient._invoke_openai") as fallback:
        client = ModelClient(
            agent_type="sql", deadline_ts=900.0, clock=clock_at(1000.0),
            enable_openai_fallback=True,
        )
        with pytest.raises(BudgetExceededError):
            client.invoke("s", "u")
        fallback.assert_not_called()
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_model_client_deadline.py -q`
Expected: FAIL — `ImportError: cannot import name 'BudgetExceededError'`

- [ ] **Step 3: Sửa `model/model_client.py`**

Thêm import ở đầu file:

```python
from graph.budget import MODEL_CALL_ESTIMATE_S, Clock, has_room_for, time_left
```

Thêm lớp lỗi ngay trước `class ModelClient`:

```python
class BudgetExceededError(RuntimeError):
    """Không còn đủ thời gian để khởi động một lời gọi model.

    Tách khỏi RuntimeError thường vì nó KHÔNG phải lỗi của Ollama, nên
    không được kích hoạt đường fallback sang OpenAI: gọi sang API ngoài
    lúc sắp hết giờ chỉ tiêu thêm thời gian mình không có, và làm schema
    của khách rời khỏi mạng của họ.
    """
```

Thêm hai tham số vào `__init__` (sau `openai_model`):

```python
        deadline_ts: float | None = None,
        clock: Clock = time.time,
```

và ở cuối thân `__init__`:

```python
        self.deadline_ts = deadline_ts
        self.clock = clock
```

Thêm hai phương thức mới ngay trước `invoke`:

```python
    def effective_timeout_s(self) -> float:
        """Timeout thực dùng: nhỏ hơn giữa timeout của agent và thời gian còn lại.

        AGENT_TIMEOUT_S['sql'] là 200s — lớn hơn cả ngân sách 45s của cả
        lượt. Không siết theo thời gian còn lại thì tham số deadline chỉ là
        trang trí.
        """
        if self.deadline_ts is None:
            return self.timeout_s
        return max(0.0, min(self.timeout_s, time_left(self.deadline_ts, self.clock)))

    def _assert_budget(self) -> None:
        if self.deadline_ts is None:
            return
        if not has_room_for(self.deadline_ts, MODEL_CALL_ESTIMATE_S, clock=self.clock):
            raise BudgetExceededError(
                f"Không khởi động lời gọi model cho agent '{self.agent_type}': còn "
                f"{time_left(self.deadline_ts, self.clock):.1f}s, cần khoảng "
                f"{MODEL_CALL_ESTIMATE_S:.0f}s — vượt ngân sách."
            )
```

Trong `_invoke_ollama`, đổi hai chỗ dùng `self.timeout_s` thành `self.effective_timeout_s()`:

```python
        last_error: Exception | None = None
        budget = self.effective_timeout_s()
        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                out = self._chat_once_ollama(messages)
                elapsed = time.time() - start
                if elapsed > budget:
                    raise TimeoutError(
                        f"Ollama response exceeded timeout: {elapsed:.1f}s > {budget:.1f}s"
                    )
```

Đổi đầu `invoke` để kiểm ngân sách trước, và để `BudgetExceededError` đi thẳng ra ngoài:

```python
    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """
        Invoke model text with local-first strategy.
        """
        self._assert_budget()
        try:
            return self._invoke_ollama(system_prompt=system_prompt, user_prompt=user_prompt)
        except BudgetExceededError:
            raise
        except Exception as ollama_error:  # noqa: BLE001
```

- [ ] **Step 4: Chạy test và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_model_client_deadline.py -q`
Expected: PASS, 7 passed

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: 595 passed, 4 failed, 19 skipped — vẫn đúng 4 fail đã biết ở `test_supervisor.py`. Không fail mới.

```bash
git add model/model_client.py tests/unit/test_model_client_deadline.py
git commit -m "feat(budget): ModelClient từ chối khởi động lời gọi không kịp deadline"
```

---

## Task 8: Node `finalize`

Thay đổi khiến hard timeout trở nên chấp nhận được. Hiện tại hết retry → `status: "failed"` → người dùng nhận **không gì cả**. Một bảng dữ liệu kèm dòng "phần insight bị bỏ do hết thời gian" có giá trị hơn hẳn một thông báo lỗi (spec 5.3).

**Files:**
- Create: `graph/agents/finalize.py`
- Create: `tests/unit/test_finalize.py`

**Interfaces:**
- Consumes: `graph.budget.is_expired`, `graph.utils.append_trace`
- Produces: `graph.agents.finalize.finalize_node(state: MultiAgentState) -> MultiAgentState` — đặt `status ∈ {"success","partial","failed"}` và `degradation_reason: list[str]`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_finalize.py`:

```python
"""Node finalize — thang suy giảm có kiểm soát (spec 5.3)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.finalize import finalize_node
from graph.state import make_initial_state
from perception.schema_context import SchemaContext

PLAN = [
    {"step": 1, "agent": "sql", "task": "t1", "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "viz", "task": "t2", "depends_on": ["sql"],
     "skill_type": "visualization"},
    {"step": 3, "agent": "insight", "task": "t3", "depends_on": ["sql"],
     "skill_type": "insight-generation"},
]


def _state(**over):
    s = make_initial_state("q", SchemaContext(), budget_s=45, clock=lambda: 1000.0)
    s["execution_plan"] = PLAN
    s["completed_agents"] = ["supervisor"]
    s["status"] = "running"
    s.update(over)
    return s


# ── success ─────────────────────────────────────────────────────────────────

def test_full_plan_completed_is_success():
    s = _state(
        completed_agents=["supervisor", "sql", "viz", "insight"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        insight={"summary": "x"},
    )
    assert finalize_node(s)["status"] == "success"


def test_success_has_no_degradation_reason():
    s = _state(
        completed_agents=["supervisor", "sql", "viz", "insight"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        insight={"summary": "x"},
    )
    assert finalize_node(s)["degradation_reason"] == []


# ── partial ─────────────────────────────────────────────────────────────────

def test_sql_without_insight_is_partial_not_failed():
    """Người dùng nhận bảng dữ liệu. Đó có giá trị hơn một thông báo lỗi."""
    s = _state(completed_agents=["supervisor", "sql"], sql_result={"sql": "SELECT 1", "row_count": 3})
    assert finalize_node(s)["status"] == "partial"


def test_partial_names_every_step_that_was_skipped():
    s = _state(completed_agents=["supervisor", "sql"], sql_result={"sql": "SELECT 1", "row_count": 3})
    reasons = " ".join(finalize_node(s)["degradation_reason"])
    assert "viz" in reasons
    assert "insight" in reasons


def test_expired_deadline_is_named_in_the_reason():
    s = _state(
        completed_agents=["supervisor", "sql"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        deadline_ts=900.0,
    )
    out = finalize_node(s)
    assert out["status"] == "partial"
    assert any("ngân sách" in r for r in out["degradation_reason"])


def test_call_ceiling_is_named_in_the_reason():
    s = _state(
        completed_agents=["supervisor", "sql"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        llm_calls_used=12,
    )
    assert any("lời gọi" in r for r in finalize_node(s)["degradation_reason"])


# ── failed ──────────────────────────────────────────────────────────────────

def test_no_sql_result_is_failed():
    """Không có gì dùng được thì gọi đúng tên nó là thất bại."""
    assert finalize_node(_state())["status"] == "failed"


def test_failed_keeps_the_last_error_for_the_trace():
    s = _state(last_error={"agent": "sql", "error_type": "sql_execution", "traceback": "boom"})
    out = finalize_node(s)
    assert out["status"] == "failed"
    assert out["last_error"]["traceback"] == "boom"


# ── bất biến ────────────────────────────────────────────────────────────────

def test_finalize_always_appends_a_trace_entry():
    before = len(_state()["action_trace"])
    assert len(finalize_node(_state())["action_trace"]) == before + 1


def test_finalize_never_raises_on_an_empty_state():
    """finalize LUÔN chạy — quá hạn, xong kế hoạch, hay lỗi chí mạng. Nó là
    node cuối duy nhất, nên nó ném lỗi là người dùng không nhận được gì."""
    s = make_initial_state("q", SchemaContext())
    assert finalize_node(s)["status"] in {"success", "partial", "failed"}


def test_finalize_does_not_mutate_its_input():
    import copy

    s = _state()
    before = copy.deepcopy(s)
    finalize_node(s)
    assert s == before
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_finalize.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'graph.agents.finalize'`

- [ ] **Step 3: Viết `graph/agents/finalize.py`**

```python
"""Node cuối duy nhất của graph.

Trước bản này, hết retry → status='failed' → người dùng nhận KHÔNG GÌ CẢ,
sau khi đã chờ. Một bảng dữ liệu kèm dòng "phần insight bị bỏ do hết thời
gian" có giá trị hơn hẳn một thông báo lỗi — và chính điều đó khiến việc
cắt cứng theo deadline trở thành chấp nhận được thay vì tàn nhẫn (spec 5.3).

`finalize` LUÔN chạy: quá hạn, xong kế hoạch, hay lỗi chí mạng. Vì thế nó
không được phép ném lỗi — nó là chỗ cuối cùng còn cơ hội trả về thứ gì đó.
"""

from __future__ import annotations

import logging

from graph.budget import calls_exhausted, is_expired
from graph.state import MultiAgentState
from graph.utils import append_trace

logger = logging.getLogger(__name__)


def _planned_agents(state: MultiAgentState) -> list[str]:
    return [str(step.get("agent", "")) for step in state.get("execution_plan", [])]


def _skipped_agents(state: MultiAgentState) -> list[str]:
    completed = set(state.get("completed_agents", []))
    return [agent for agent in _planned_agents(state) if agent not in completed]


def finalize_node(state: MultiAgentState) -> MultiAgentState:
    """Gom mọi thứ đang có và phân loại kết quả.

    | status    | Điều kiện                              |
    |-----------|----------------------------------------|
    | success   | Kế hoạch hoàn tất                      |
    | partial   | Có kết quả SQL, một số bước bị bỏ      |
    | failed    | Không có output nào dùng được          |

    `degradation_reason` ghi ra để UI hiển thị được — nuốt lỗi im lặng ở
    đây là cách chắc chắn nhất khiến không ai biết hệ thống đang cắt gì.
    """
    skipped = _skipped_agents(state)
    has_sql = bool(state.get("sql_result"))

    reasons: list[str] = list(state.get("degradation_reason", []))
    if is_expired(state.get("deadline_ts", 0.0)):
        reasons.append("Hết ngân sách thời gian trước khi kế hoạch chạy xong.")
    if calls_exhausted(state.get("llm_calls_used", 0)):
        reasons.append(
            f"Chạm trần {state.get('llm_calls_used', 0)} lời gọi model cho một câu hỏi."
        )
    if skipped:
        reasons.append("Bước bị bỏ: " + ", ".join(skipped) + ".")

    if not skipped and has_sql:
        status = "success"
        reasons = []
        observation = "Kế hoạch hoàn tất."
    elif has_sql:
        status = "partial"
        observation = "Kết quả một phần — " + " ".join(reasons)
    else:
        status = "failed"
        last_error = state.get("last_error") or {}
        observation = (
            "Không có output dùng được. "
            f"Lỗi cuối: {last_error.get('error_type', 'không rõ')}."
        )
        if not reasons:
            reasons = [observation]

    logger.info("Finalize: status=%s, skipped=%s", status, skipped)
    trace = append_trace(state, "finalize", "finalize", observation, status)

    return {
        **state,
        "current_agent": "finalize",
        "status": status,
        "degradation_reason": reasons,
        "action_trace": trace,
    }
```

- [ ] **Step 4: Chạy test và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_finalize.py -q`
Expected: PASS, 11 passed

```bash
git add graph/agents/finalize.py tests/unit/test_finalize.py
git commit -m "feat(graph): node finalize — thang success/partial/failed có lý do"
```

---

## Task 9: Nối `finalize` vào graph, thay mọi `END` trần

**Files:**
- Modify: `graph/multi_agent.py`
- Modify: `graph/agents/supervisor.py` (`route_reflector_return`)
- Modify: `tests/unit/test_supervisor.py:127, 132, 142, 176`

**Interfaces:**
- Consumes: `graph.agents.finalize.finalize_node`
- Produces: `graph.multi_agent.run_graph(query, schema_context, profile, user, budget_s=DEFAULT_BUDGET_S)`

- [ ] **Step 1: Sửa bốn assertion đã biết trong `tests/unit/test_supervisor.py`**

Ở dòng 127, 132, 142 và 176, đổi:

```python
        assert route_next_agent(s) == END
```

thành:

```python
        assert route_next_agent(s) == "finalize"
```

Nếu `END` không còn được dùng ở đâu khác trong file, xoá dòng `from langgraph.graph import END`.

- [ ] **Step 2: Chạy test để xác nhận pass ở tầng router**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_supervisor.py -q`
Expected: PASS — bốn fail đã biết biến mất.

- [ ] **Step 3: Viết test tích hợp cho đường đi mới**

Tạo `tests/integration/test_budget_end_to_end.py`:

```python
"""Ngân sách nhìn từ ngoài graph.

Mọi lời gọi model đều mock — không chạm Ollama, không chạm Postgres.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.multi_agent import build_multi_agent_graph
from graph.state import make_initial_state
from perception.connection_profile import ALL_TABLES, build_profile
from perception.schema_context import SchemaContext
from tests.fixtures.mini_schema import MINI_TABLES

PROFILE = build_profile(
    dsn="postgresql://u:p@h:5432/d",
    tables=MINI_TABLES,
    grants={"test_user": frozenset({ALL_TABLES})},
)

PLAN_JSON = {
    "plan_summary": "s",
    "steps": [
        {"step": 1, "agent": "sql", "task": "t1", "depends_on": [], "skill_type": "text-to-sql"},
        {"step": 2, "agent": "insight", "task": "t2", "depends_on": ["sql"],
         "skill_type": "insight-generation"},
    ],
}


def _initial(budget_s: float, now: float = 1000.0):
    s = make_initial_state("q", SchemaContext(), budget_s=budget_s, clock=lambda: now)
    s["shared_metadata"] = {"profile": PROFILE, "user": "test_user"}
    return s


def test_every_run_ends_at_finalize():
    """Không còn đường nào ra khỏi graph mà bỏ qua finalize."""
    with patch("graph.agents.supervisor.ModelClient") as sup:
        sup.return_value.invoke_json.return_value = PLAN_JSON
        out = build_multi_agent_graph().invoke(_initial(budget_s=0))
    assert out["current_agent"] == "finalize"


def test_an_exhausted_budget_never_reaches_the_sql_node():
    with patch("graph.agents.supervisor.ModelClient") as sup, \
         patch("graph.agents.sql_agent.ModelClient") as sql:
        sup.return_value.invoke_json.return_value = PLAN_JSON
        build_multi_agent_graph().invoke(_initial(budget_s=0))
        sql.return_value.invoke.assert_not_called()


def test_an_exhausted_budget_yields_a_reason_not_a_silent_failure():
    with patch("graph.agents.supervisor.ModelClient") as sup:
        sup.return_value.invoke_json.return_value = PLAN_JSON
        out = build_multi_agent_graph().invoke(_initial(budget_s=0))
    assert out["degradation_reason"], "phải nói ra vì sao bị cắt"
    assert out["status"] in {"partial", "failed"}
```

- [ ] **Step 4: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/integration/test_budget_end_to_end.py -q`
Expected: FAIL — `KeyError: 'finalize'` hoặc `assert '__end__' == 'finalize'`

- [ ] **Step 5: Nối node vào `graph/multi_agent.py`**

Thêm import:

```python
from graph.agents.finalize import finalize_node
from graph.budget import DEFAULT_BUDGET_S
```

Thêm node, sau `builder.add_node("reflector", reflector_agent_node)`:

```python
    builder.add_node("finalize", finalize_node)
```

Thay **cả ba** bản đồ route. Bản đồ dùng cho `supervisor` và cho vòng lặp specialist đổi khoá `END: END` thành `"finalize": "finalize"`:

```python
    _ROUTES = {
        "sql": "sql",
        "python": "python",
        "viz": "viz",
        "insight": "insight",
        "reflector": "reflector",
        "finalize": "finalize",
    }

    builder.add_conditional_edges("supervisor", route_next_agent, _ROUTES)

    for agent in ("sql", "python", "viz"):
        builder.add_conditional_edges(agent, route_next_agent, _ROUTES)
```

Insight không còn đi thẳng ra `END`:

```python
    # Insight xong vẫn đi qua finalize — finalize là nơi duy nhất đặt status,
    # nên không đường nào được vòng qua nó.
    builder.add_edge("insight", "finalize")
```

Bản đồ của reflector:

```python
    builder.add_conditional_edges(
        "reflector",
        route_reflector_return,
        {
            "sql": "sql",
            "python": "python",
            "viz": "viz",
            "insight": "insight",
            "finalize": "finalize",
        },
    )

    builder.add_edge("finalize", END)
```

Sửa `route_reflector_return` trong `graph/agents/supervisor.py` — thay `return END` cuối hàm:

```python
def route_reflector_return(state: MultiAgentState) -> str:
    """After reflector diagnoses, route back to the failed agent."""
    last_error = state.get("last_error") or {}
    failed_agent = last_error.get("agent", "")
    if failed_agent in {"sql", "python", "viz", "insight"}:
        return failed_agent
    logger.warning("Reflector: no valid agent target → finalize")
    return "finalize"
```

Xoá dòng `from langgraph.graph import END` cục bộ bên trong hàm đó.

Cuối cùng, `run_graph` nhận ngân sách:

```python
def run_graph(
    query: str,
    schema_context: SchemaContext,
    profile: ConnectionProfile,
    user: str,
    budget_s: float = DEFAULT_BUDGET_S,
) -> MultiAgentState:
    """Convenience: build the graph and invoke it in one shot.

    profile/user travel through shared_metadata so the sql node can pass
    them down to execute_sql — see graph/agents/sql_agent.py.
    """
    graph = build_multi_agent_graph()
    initial = make_initial_state(query, schema_context, budget_s=budget_s)
    initial["shared_metadata"] = {"profile": profile, "user": user}
    return graph.invoke(initial)
```

- [ ] **Step 6: Chạy test và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: PASS, 613 passed, 19 skipped. **Không còn fail nào** — bốn fail dự kiến từ Task 5 đã đóng.

```bash
git add graph/multi_agent.py graph/agents/supervisor.py tests/unit/test_supervisor.py tests/integration/test_budget_end_to_end.py
git commit -m "feat(graph): mọi đường ra đi qua finalize, không còn END trần"
```

---

## Task 10: Cấp phát theo dự trữ và chặn ở cửa node

Điểm kiểm thứ nhất và thứ ba trong ba điểm của spec 5.1. Thang ưu tiên: `sql` bắt buộc → `insight` được bảo vệ bằng dự trữ 12s → `python` và `viz` bị cắt trước tiên. Nếu SQL chạy được, người dùng **luôn** nhận được insight, kể cả khi biểu đồ bị bỏ (spec 5.2).

**Files:**
- Modify: `graph/agents/sql_agent.py`, `python_agent.py`, `viz_agent.py`, `insight_agent.py`, `reflector_agent.py`
- Test: `tests/unit/test_budget_gating.py` (tạo mới)

**Interfaces:**
- Consumes: `graph.budget.has_room_for`, `INSIGHT_RESERVE_S`, `MODEL_CALL_ESTIMATE_S`
- Produces: `graph.budget.node_may_run(state: MultiAgentState, agent: str) -> tuple[bool, str]` — `(chạy được?, lý do nếu không)`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_budget_gating.py`:

```python
"""Cấp phát theo dự trữ (spec 5.2).

Chia đều sẽ bỏ đói bước cuối, mà bước cuối lại là thứ người dùng đọc.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.budget import node_may_run
from graph.state import make_initial_state
from perception.schema_context import SchemaContext


def _state(seconds_left: float, **over):
    s = make_initial_state("q", SchemaContext(), budget_s=seconds_left, clock=lambda: 1000.0)
    s["_clock"] = lambda: 1000.0
    s.update(over)
    return s


def test_sql_may_run_with_plenty_of_time():
    ok, _ = node_may_run(_state(45), "sql")
    assert ok is True


def test_sql_may_run_even_inside_the_insight_reserve():
    """sql là bắt buộc: không có nó thì không có gì để nhận xét, nên nó
    được phép lấn vào phần dự trữ."""
    ok, _ = node_may_run(_state(16), "sql")
    assert ok is True


def test_viz_is_cut_before_the_reserve_is_touched():
    """Còn 20s, cần ~15s, dự trữ 12s → 20 - 12 = 8 < 15. Cắt viz."""
    ok, reason = node_may_run(_state(20), "viz")
    assert ok is False
    assert "viz" in reason


def test_python_is_cut_before_the_reserve_is_touched():
    ok, _ = node_may_run(_state(20), "python")
    assert ok is False


def test_viz_runs_when_there_is_room_beyond_the_reserve():
    """Còn 40s, dự trữ 12s → 28s dùng được, đủ cho một lời gọi ~15s."""
    ok, _ = node_may_run(_state(40), "viz")
    assert ok is True


def test_insight_may_spend_the_reserve_it_is_reserved_for():
    """Dự trữ dành riêng cho insight, nên insight không tự trừ nó của mình."""
    ok, _ = node_may_run(_state(16), "insight")
    assert ok is True


def test_insight_is_cut_when_even_the_reserve_is_gone():
    ok, reason = node_may_run(_state(5), "insight")
    assert ok is False
    assert "ngân sách" in reason


def test_nothing_may_run_past_the_deadline():
    for agent in ("sql", "python", "viz", "insight", "reflector"):
        ok, _ = node_may_run(_state(0), agent)
        assert ok is False, f"{agent} chạy sau deadline"


def test_reflector_is_cut_before_specialists_are():
    """Reflector là chẩn đoán, không phải kết quả. Nó đi trước trên đoạn ván."""
    ok, _ = node_may_run(_state(20), "reflector")
    assert ok is False


def test_the_call_ceiling_blocks_a_node_that_still_has_time():
    ok, reason = node_may_run(_state(45, llm_calls_used=12), "sql")
    assert ok is False
    assert "lời gọi" in reason
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_budget_gating.py -q`
Expected: FAIL — `ImportError: cannot import name 'node_may_run'`

- [ ] **Step 3: Thêm `node_may_run` vào `graph/budget.py`**

```python
# Node nào được lấn vào phần dự trữ dành cho insight.
#
#   sql       — bắt buộc. Không có kết quả SQL thì không có gì để nhận xét,
#               nên nó được lấn.
#   insight   — chính là chủ của phần dự trữ, nên không tự trừ nó của mình.
#   python/viz/reflector — bị cắt trước tiên. Người dùng mất biểu đồ nhưng
#               vẫn nhận được bảng và nhận xét (spec 5.2).
_MAY_USE_RESERVE = frozenset({"sql", "insight"})


def node_may_run(state, agent: str) -> tuple[bool, str]:
    """Node `agent` còn được phép chạy không, và nếu không thì vì sao.

    `state` là MultiAgentState nhưng không annotate ở đây: module này nằm
    dưới đáy cây phụ thuộc và import graph.state sẽ tạo vòng.

    Đọc `state["_clock"]` nếu có — chỉ để test tiêm đồng hồ; production
    không đặt khoá đó và rơi về time.time.
    """
    clock: Clock = state.get("_clock", time.time)
    deadline_ts = state.get("deadline_ts", 0.0)

    if calls_exhausted(state.get("llm_calls_used", 0)):
        return False, (
            f"Bỏ qua '{agent}': đã chạm trần "
            f"{state.get('llm_calls_used', 0)} lời gọi model cho câu hỏi này."
        )

    reserve = 0.0 if agent in _MAY_USE_RESERVE else INSIGHT_RESERVE_S
    if not has_room_for(deadline_ts, MODEL_CALL_ESTIMATE_S, reserve_s=reserve, clock=clock):
        return False, (
            f"Bỏ qua '{agent}': còn {time_left(deadline_ts, clock):.1f}s, "
            f"không đủ ngân sách cho một lời gọi model"
            + (f" mà không lấn vào {reserve:.0f}s dự trữ cho insight." if reserve else ".")
        )
    return True, ""
```

- [ ] **Step 4: Chặn ở cửa năm node**

Với mỗi node trong `sql_agent.py`, `python_agent.py`, `viz_agent.py`, `insight_agent.py`, `reflector_agent.py`: thêm import

```python
from graph.budget import node_may_run
```

và chèn khối dưới đây làm **việc đầu tiên** trong thân node, trước cả khi tìm step trong plan. Đổi `"sql"` thành tên agent tương ứng ở mỗi file.

```python
    may_run, reason = node_may_run(state, "sql")
    if not may_run:
        logger.warning("SQL Agent: %s", reason)
        return {
            **state,
            "current_agent": "sql",
            "completed_agents": state.get("completed_agents", []) + ["sql"],
            "degradation_reason": state.get("degradation_reason", []) + [reason],
            "action_trace": append_trace(state, "sql", "budget_check", reason, "error"),
            "status": "running",
        }
```

`completed_agents` được cộng thêm agent bị bỏ **có chủ ý**: nếu không, router sẽ chọn lại đúng node đó ở vòng sau và graph quay vòng cho tới khi chạm trần call. Bước bị bỏ vẫn hiện ra trong `degradation_reason`, nên "hoàn thành" ở đây không giấu được gì khỏi người dùng.

- [ ] **Step 5: Đếm `llm_calls_used` ở năm node**

Trần 12 lời gọi chỉ có tác dụng nếu có ai đếm. `ModelClient` không đếm được — nó không thấy state — nên việc đếm thuộc về node.

Trong `graph/agents/sql_agent.py`, mỗi vòng của vòng lặp retry là đúng một lời gọi model, nên `attempt` chính là số lượt đã đốt. Ở câu `return` **đường thành công** (dòng 170-188) thêm khoá:

```python
            "llm_calls_used": state.get("llm_calls_used", 0) + attempt,
```

Ở câu `return` **đường thất bại** (dòng 194-213) thêm:

```python
        "llm_calls_used": state.get("llm_calls_used", 0) + MAX_RETRIES,
```

Không cộng ở đường thất bại thì các lượt đã đốt biến mất khỏi sổ, và trần cứng đếm hụt đúng ở trường hợp nó cần đếm nhất.

Làm y hệt cho `python_agent.py` và `viz_agent.py` — cả hai đều có vòng `for attempt in range(1, MAX_RETRIES + 1)` và hai đường return tương ứng.

`insight_agent.py` và `reflector_agent.py` mỗi node đúng một lời gọi, không có vòng lặp. Thêm vào **cả** câu return thành công và câu return lỗi:

```python
            "llm_calls_used": state.get("llm_calls_used", 0) + 1,
```

- [ ] **Step 6: Truyền `deadline_ts` vào mọi `ModelClient`**

Ở năm node, thêm tham số vào chỗ dựng client. `sql_agent.py` dòng 118:

```python
    client = ModelClient(agent_type="sql", deadline_ts=state.get("deadline_ts"))
```

Tương tự với `agent_type="supervisor"` (`supervisor.py:125`), `"python"` (`python_agent.py`), `"viz"` (`viz_agent.py`), `"insight"` (`insight_agent.py:93`), `"reflector"` (`reflector_agent.py:77`).

Không cần thêm `except` cho `BudgetExceededError`: nó kế thừa `RuntimeError` nên các khối `except Exception` sẵn có trong mỗi vòng lặp đã bắt được, và chúng gán `error_context = f"ModelClient error: {exc}"` nên thông điệp ngân sách đi vào trace nguyên vẹn.

- [ ] **Step 7: Chạy test và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_budget_gating.py -q`
Expected: PASS, 10 passed

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: PASS, 623 passed, 19 skipped

```bash
git add graph/budget.py graph/agents/sql_agent.py graph/agents/python_agent.py graph/agents/viz_agent.py graph/agents/insight_agent.py graph/agents/reflector_agent.py tests/unit/test_budget_gating.py
git commit -m "feat(budget): cấp phát theo dự trữ — python/viz bị cắt trước, insight được bảo vệ"
```

---

## Task 11: Nhãn lỗi và trace JSONL

**Files:**
- Create: `graph/errors.py`
- Create: `tests/unit/test_trace_jsonl.py`
- Modify: `graph/utils.py`
- Modify: `graph/agents/supervisor.py`, `insight_agent.py` (dùng nhãn từ `graph/errors.py`)

**Interfaces:**
- Produces: `graph.errors.ERROR_LABELS: frozenset[str]` và hằng số cho từng nhãn
- Produces: `graph.utils.append_trace(state, agent, action, observation, status, *, duration_ms: int | None = None) -> list[dict]` — tham số mới là keyword-only có mặc định, mọi lời gọi hiện có vẫn chạy

**Sai lệch có chủ ý so với spec 5.6:** spec liệt kê 9 nhãn nhưng **không có nhãn nào cho lỗi lập kế hoạch của supervisor hay lỗi sinh insight**, trong khi code đang ghi `planning_failure` và `insight_generation_failure`. Ép chúng vào 9 nhãn kia sẽ là dán nhãn sai, và reflector chẩn đoán theo nhãn. Nên tập nhãn ở đây là **9 nhãn của spec cộng hai nhãn đó**, và sai lệch này được ghi ngay trong docstring của module để lần đọc spec sau không tưởng là code lệch chuẩn.

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_trace_jsonl.py`:

```python
"""Trace mang ngân sách, và ghi ra đĩa được (spec 5.6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.errors import ERROR_LABELS
from graph.state import make_initial_state
from graph.utils import append_trace
from perception.schema_context import SchemaContext


def _state():
    s = make_initial_state("q", SchemaContext(), budget_s=45, clock=lambda: 1000.0)
    s["llm_calls_used"] = 3
    return s


def test_trace_entry_carries_the_query_id():
    s = _state()
    entry = append_trace(s, "sql", "execute_sql", "ok", "ok")[-1]
    assert entry["query_id"] == s["query_id"]


def test_trace_entry_carries_the_llm_call_count():
    assert append_trace(_state(), "sql", "a", "o", "ok")[-1]["llm_calls"] == 3


def test_trace_entry_carries_remaining_budget():
    entry = append_trace(_state(), "sql", "a", "o", "ok")[-1]
    assert entry["deadline_remaining_ms"] is not None


def test_trace_entry_carries_duration_when_given():
    entry = append_trace(_state(), "sql", "a", "o", "ok", duration_ms=1234)[-1]
    assert entry["duration_ms"] == 1234


def test_duration_is_none_when_not_measured():
    assert append_trace(_state(), "sql", "a", "o", "ok")[-1]["duration_ms"] is None


def test_old_five_argument_call_still_works():
    """Mọi node đang gọi năm tham số vị trí. Đừng làm hỏng."""
    assert len(append_trace(_state(), "sql", "a", "o", "ok")) == 1


def test_entries_are_written_as_jsonl_keyed_by_query_id(tmp_path, monkeypatch):
    monkeypatch.setattr("graph.utils._TRACE_DIR", tmp_path)
    s = _state()
    append_trace(s, "sql", "execute_sql", "ok", "ok")
    lines = (tmp_path / f"{s['query_id']}.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["agent"] == "sql"


def test_a_write_failure_never_breaks_the_query(monkeypatch):
    """Log hỏng không được làm hỏng câu trả lời của người dùng."""
    monkeypatch.setattr("graph.utils._TRACE_DIR", Path("/khong/ton/tai/duoc"))
    assert len(append_trace(_state(), "sql", "a", "o", "ok")) == 1


def test_the_nine_spec_labels_are_all_present():
    assert {
        "sql_generation", "sql_execution", "sql_timeout", "python_runtime",
        "chart_error", "schema_mismatch", "model_timeout", "budget_exceeded",
        "tool_unavailable",
    } <= ERROR_LABELS


def test_planning_and_insight_labels_are_present_too():
    """Sai lệch có chủ ý so với spec 5.6 — xem docstring graph/errors.py."""
    assert {"planning_failure", "insight_generation"} <= ERROR_LABELS
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit/test_trace_jsonl.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'graph.errors'`

- [ ] **Step 3: Viết `graph/errors.py`**

```python
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
```

- [ ] **Step 4: Mở rộng `append_trace` trong `graph/utils.py`**

Thay phần đầu file và hàm `append_trace`:

```python
_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_TRACE_DIR = _LOG_DIR / "traces"


def _write_trace_line(entry: dict[str, Any]) -> None:
    """Nối một dòng JSONL vào logs/traces/<query_id>.jsonl.

    Nuốt mọi lỗi có chủ ý: đây là log, và một câu trả lời đúng không được
    mất chỉ vì đĩa đầy hay thư mục không ghi được.
    """
    query_id = entry.get("query_id")
    if not query_id:
        return
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        with (_TRACE_DIR / f"{query_id}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def append_trace(
    state: MultiAgentState,
    agent: str,
    action: str,
    observation: str,
    status: str,
    *,
    duration_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Log an action-trace entry and return the updated trace list.

    `duration_ms` là keyword-only có mặc định, nên hàng chục lời gọi năm
    tham số vị trí đang có vẫn chạy nguyên.
    """
    deadline_ts = state.get("deadline_ts")
    entry: dict[str, Any] = {
        "query_id": state.get("query_id", ""),
        "agent": agent,
        "action": action,
        "observation": observation,
        "status": status,
        "timestamp": time.time(),
        "llm_calls": state.get("llm_calls_used", 0),
        "deadline_remaining_ms": (
            None if deadline_ts is None else int((deadline_ts - time.time()) * 1000)
        ),
        "duration_ms": duration_ms,
    }
    _write_trace_line(entry)
    return state.get("action_trace", []) + [entry]
```

Thêm `import json` vào đầu file.

- [ ] **Step 5: Dùng nhãn chuẩn ở hai chỗ đang lệch**

`graph/agents/supervisor.py` — đổi `"error_type": "planning_failure"` thành `"error_type": PLANNING_FAILURE`, thêm import `from graph.errors import PLANNING_FAILURE`.

`graph/agents/insight_agent.py` — đổi `"error_type": "insight_generation_failure"` thành `"error_type": INSIGHT_GENERATION`, thêm import `from graph.errors import INSIGHT_GENERATION`.

- [ ] **Step 6: Thêm `logs/traces/` vào `.gitignore`**

Kiểm tra `logs/` đã được ignore chưa:

```bash
git check-ignore -v logs/traces
```

Nếu chưa, thêm dòng `logs/` vào `.gitignore`.

- [ ] **Step 7: Chạy test và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: PASS, 633 passed, 19 skipped

```bash
git add graph/errors.py graph/utils.py graph/agents/supervisor.py graph/agents/insight_agent.py tests/unit/test_trace_jsonl.py .gitignore
git commit -m "feat(trace): query_id + ngân sách trong trace, ghi JSONL; nhãn lỗi một nguồn"
```

---

## Task 12: `app.py` hiển thị kết quả một phần

**Task cuối cùng.** `app.py` là file chung duy nhất giữa hai plan. **Plan A đã merge vào `main` ngày 2026-09-01** (PR #1 `a2682cf`, cộng PR #2/#3 sau đó) — gate ở Step 1 của bản plan gốc đã qua, không còn là điều kiện chặn. Nhưng việc merge đó viết lại `app.py` khá nhiều: hàm hiển thị kết quả đổi tên từ `_render_results` thành `_display_result`, và phần đọc `ConnectionProfile` giờ đi qua `_load_profile()` (đọc từ đĩa, cache theo `@st.cache_resource`) thay vì dựng lại từ `info_box_all.json` mỗi câu hỏi. Các dòng/anchor dưới đây đã cập nhật theo state thật tại commit `d1011a9` (merge `origin/main` vào nhánh này) — đối chiếu lại nếu `main` trôi tiếp trước khi thực thi task này.

**Files:**
- Modify: `app.py:204-206` (đầu `_display_result`) và `app.py:348-354` (khối "Last Run" trong sidebar)

**Interfaces:**
- Consumes: `state["status"] ∈ {"success","partial","failed"}`, `state["degradation_reason"]: list[str]`, `state["llm_calls_used"]: int`

- [ ] **Step 1: Thêm dải trạng thái vào đầu `_display_result`**

Hàm hiện tại (`app.py:204-206`):

```python
def _display_result(result: MultiAgentState) -> None:
    """Display all result sections from a completed run."""
    st.divider()
```

Chèn ngay sau `st.divider()`:

```python
    # ── Trạng thái và lý do bị cắt ──
    status = result.get("status", "unknown")
    reasons = result.get("degradation_reason") or []
    if status == "partial":
        st.warning(
            "**Kết quả một phần.** " + " ".join(reasons)
            + "\n\nPhần hiển thị bên dưới là những gì chạy xong trong ngân sách thời gian."
        )
    elif status == "failed" and reasons:
        st.error("**Không hoàn thành.** " + " ".join(reasons))
```

Không nuốt lý do vào log: nếu người dùng không thấy hệ thống đã cắt gì, họ sẽ đọc một bảng thiếu như thể nó đầy đủ.

- [ ] **Step 2: Thêm số lời gọi model vào thanh metric trong sidebar**

Khối "Last Run" hiện tại (`app.py:348-354`), trong `with st.sidebar:`:

```python
    if st.session_state.last_result:
        st.divider()
        st.header("Last Run")
        result = st.session_state.last_result
        st.metric("Status", result.get("status", "unknown"))
        st.metric("Agents Completed", len(result.get("completed_agents", [])))
        st.metric("Errors", result.get("error_count", 0))
```

Thêm một dòng ngay sau `st.metric("Errors", ...)`:

```python
        st.metric("Lời gọi model", result.get("llm_calls_used", 0))
```

- [ ] **Step 3: Chạy test và commit**

Run: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest tests/unit tests/integration -q`
Expected: PASS, 633 passed, 19 skipped

```bash
git add app.py
git commit -m "feat(ui): hiển thị kết quả một phần và lý do bị cắt"
```

---

## Phần spec cố ý để ngoài Plan B

Ghi ra để lần đọc spec sau không tưởng là bị bỏ sót.

| Spec | Ở đâu | Vì sao không ở đây |
|---|---|---|
| 4.2 — container `adba-tools` riêng, non-root, `read_only` rootfs, chặn egress, `mem_limit`/`cpus`/`pids_limit` | **Plan C** (đóng gói on-prem) | Đó là cấu hình hạ tầng, không phải code. Nó cần `sandbox/Dockerfile` và một compose file cho bản giao khách — cả hai là hạng mục của Plan C. `spawn` + env rỗng ở Task 1 là phần code có thể làm ngay và cũng là phần chặn được rò credential; container thêm lớp chặn ghi file và egress. |
| 4.1 lớp 2 (chặn multi-statement) và lớp 3 (quét DML trên statement đã parse) | **Đã xong** | `assert_tables_permitted` dùng `sqlparse.parse` đòi đúng một statement và `get_type() == "SELECT"` (`sql_tool.py:83-90`). Data-modifying CTE bị chặn ở đây, và Task 3 thêm lớp bảo đảm ở tầng DB. |
| 4.1 — `explain_query_plan` qua cùng validator | **Đã xong** | `sql_tool.py:154` gọi `assert_tables_permitted` trước khi chạy EXPLAIN. |
| Mục 3 — `ConnectionProfile`, `allowed_tables` từ introspection | **Đã xong** | `perception/connection_profile.py` tồn tại; `_ALLOWED_TABLES` hardcode đã biến mất. |
| Mục 6 — `eval/eval_e2e.py`, golden set, subset BEAVER/BIRD | **Pha 0, chưa làm** | Xem ghi chú cuối trang. |
| Pha 3 — MCP server, handle pattern, `DatasetStore` | **Hoãn** (roadmap: "Ranh giới MCP") | Refactor nội bộ, không mở khoá gì trên đường tới khách hàng đầu tiên. |

## Cổng ra của Plan B

Ba câu hỏi từ roadmap, kèm cách kiểm từng câu:

| Cổng | Kiểm bằng |
|---|---|
| Một câu hỏi bất kỳ không bao giờ vượt ngân sách, và lượt bị cắt luôn nói ra vì sao | `tests/integration/test_budget_end_to_end.py::test_a_zero_budget_never_reaches_the_sql_node` (graph với `budget_s=0` kết thúc ở `finalize`, không chạm node SQL) và `::test_sql_succeeds_then_the_budget_runs_out_before_insight` (SQL xong, ngân sách cạn giữa chừng → `partial` kèm lý do, không phải `success` im lặng) |
| Code trong sandbox không còn đọc được credential qua environment dict CỦA CHÍNH NÓ — **đóng một phần, xem ghi chú bên dưới** | `tests/unit/test_python_sandbox_isolation.py::test_sandbox_cannot_read_parent_environment` và `::test_sandbox_environment_is_emptied_not_just_filtered` |
| Vai trò DB không thực thi được lệnh ghi | `tests/integration/test_readonly_role.py` — 7 test, gồm cả data-modifying CTE |

**Ghi chú cổng 2 — phạm vi thật của nó.** Hai test kia chứng minh đúng một điều: sau `os.environ.clear()`, environment dict **của chính tiến trình con** rỗng, nên `pd.io.common.os.environ` không còn `DATABASE_URL`. Đó là lỗ hổng cụ thể mà bản `fork` cũ để hở, và bịt nó là cải thiện thật.

Chúng **không** chứng minh rằng environment của tiến trình **cha** là bất khả xâm phạm. Trên **Linux** — môi trường khách hàng thật; máy dev macOS không có `/proc` nên không lộ ra điều này — code trong sandbox vẫn lấy được module `os` thật qua `pd.io.common.os`, rồi `os.getppid()` + đọc `/proc/<ppid>/environ` để phục hồi **nguyên vẹn** env của cha, gồm cả `DATABASE_URL`. `os.environ.clear()` không chạm tới bộ nhớ của cha nên không chặn được đường này. Cùng đường đó còn cho đọc file bất kỳ và sinh tiến trình con.

Chặn phần còn lại cần cô lập ở mức namespace/container (non-root, rootfs chỉ-đọc, không thấy `/proc` của cha, chặn egress) — tức là spec 4.2, **đã cố ý hoãn sang Plan C** ở bảng ngay phía trên. Cho tới khi Plan C xong: code chạy trong sandbox là code KHÔNG đáng tin chạy với quyền của tiến trình cha, chỉ bớt được đúng một đường rò credential. Caveat này được ghi lại ở docstring đầu `graph/tools/python_tool.py` để người đọc code thấy nó mà không phải tìm tới plan.

**Không nằm trong Plan B, và đừng báo cáo như thể có:** `slo_hit_rate`, `answer_accuracy`, `partial_rate` là số của `eval/eval_e2e.py` — thuộc pha 0 của spec, chưa tồn tại. Plan B chứng minh **cơ chế** cắt đúng, không chứng minh hệ thống đạt 45s. Con số đó chỉ đo được ở cấu hình vLLM và cần harness của pha 0.
