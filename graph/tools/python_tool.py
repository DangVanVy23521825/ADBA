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

ĐIỀU HAI ĐIỀU TRÊN *KHÔNG* BẢO ĐẢM — đọc kỹ trước khi trích dẫn module
này như một bảo đảm cô lập:

  * Chúng xoá **environment dict của chính child**. Đó là lỗ hổng cụ thể,
    có thật, mà bản `fork` trước đây để hở, và bịt nó là một cải thiện
    thật — nhưng phạm vi của nó chỉ đến đó.
  * Trên **Linux** (môi trường khách hàng thật; máy dev macOS không có
    `/proc` nên không thấy điều này), code trong sandbox vẫn lấy được
    module `os` thật qua `pd.io.common.os`, rồi `os.getppid()` +
    đọc `/proc/<ppid>/environ` để lấy **environment của tiến trình CHA**
    — nguyên vẹn, gồm cả `DATABASE_URL`. `os.environ.clear()` không chạm
    tới bộ nhớ của tiến trình cha, nên nó không chặn đường này.
  * Cùng đường đó cho phép **đọc file bất kỳ** và **sinh tiến trình con**
    (`os.system`, `subprocess` qua `os`).

Cô lập tiến trình đầy đủ — chạy non-root, rootfs chỉ-đọc, không thấy
`/proc` của cha, chặn egress, giới hạn mem/cpu/pids — là chuyện của một
container riêng, và nó CỐ Ý nằm ngoài phạm vi plan này: xem mục "Phần
spec cố ý để ngoài Plan B" trong
`docs/superpowers/plans/2026-08-17-cung-hoa-ngan-sach-sandbox.md`, nơi
spec 4.2 (container `adba-tools`) được hoãn sang **Plan C**. Cho tới khi
Plan C xong, hãy coi code chạy ở đây là code KHÔNG đáng tin chạy với
quyền của tiến trình cha, chỉ bớt được đúng một đường rò credential.

Hệ quả về kiểu dữ liệu: mọi tham số đi qua ranh giới này phải pickle
được. Lambda và module object thì không. Đó là lý do preset là một CHUỖI
và namespace được dựng bên trong child, thay vì truyền callable/module
xuống như bản dùng `fork` trước đây.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import queue
import sys
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)

# 25s, không phải 10s. `spawn` dựng một interpreter MỚI mỗi lần gọi và
# nạp lại pandas/numpy (và matplotlib ở preset chart) từ đầu — cái giá cố
# định đó lớn hơn hẳn bản `fork` cũ, nơi child thừa hưởng sẵn mọi module
# đã nạp. Trần 10s là con số của thời `fork`, và giữ nguyên nó sau khi đổi
# sang `spawn` khiến sandbox báo "timed out" cho cả code đúng và nhanh.
# Đó nhiều khả năng chính là hai lần "flaky timeout" mà người làm Task 10
# tự ghi nhận rồi quy cho tải máy.
PANDAS_EXEC_TIMEOUT_SECONDS = float(os.getenv("PANDAS_EXEC_TIMEOUT_SECONDS", "25"))

# Preset chart còn nạp thêm matplotlib trong child, nên nó có trần riêng
# — mặc định bằng trần pandas, nhưng chỉnh được độc lập khi cần.
CHART_EXEC_TIMEOUT_SECONDS = float(
    os.getenv("CHART_EXEC_TIMEOUT_SECONDS", str(PANDAS_EXEC_TIMEOUT_SECONDS))
)

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
        # còn nguyên. Sau dòng này, environment dict CỦA CHÍNH TIẾN TRÌNH
        # NÀY rỗng, nên `pd.io.common.os.environ` không còn DATABASE_URL.
        #
        # Nó KHÔNG bịt được đường đọc environment của tiến trình CHA: trên
        # Linux, `os.getppid()` + `/proc/<ppid>/environ` lấy lại toàn bộ
        # env của cha, và lời gọi này không chạm tới bộ nhớ của cha. Chặn
        # đường đó cần cô lập ở mức namespace/container (Plan C) — xem
        # docstring đầu module.
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


@contextlib.contextmanager
def _main_module_not_reexecuted():
    """Ngăn child của `spawn` chạy lại toàn bộ file `__main__`.

    `multiprocessing.spawn` dựng lại trạng thái của child bằng cách nạp
    LẠI `sys.modules["__main__"]` qua `runpy`, để nó phân giải được những
    tên có thể đã được pickle theo tham chiếu tới `__main__`. Với một
    script có `if __name__ == "__main__":` bọc phần thân, việc đó gần như
    miễn phí. Với `streamlit run app.py` thì KHÔNG: Streamlit biến
    `app.py` thành `__main__`, và app.py không có lớp bọc đó — nên MỖI
    lời gọi sandbox lại import streamlit + langgraph + matplotlib + dựng
    lại cả graph trong child trước khi chạy một dòng code nào của model.
    Đo được ~12,6s so với ~3,9s ở một entry point có bọc — quá đủ để vượt
    trần timeout và báo "Python execution timed out" cho code hoàn toàn
    đúng.

    Cách chữa tiêu chuẩn cho đúng tình huống này: tạm bỏ
    `__main__.__file__` trong lúc `proc.start()` chạy. Bước main-fixup của
    `spawn` khi đó không thấy file nào để chạy lại. Không có gì trong
    sandbox này pickle theo tham chiếu tới `__main__` (code là chuỗi, df
    là DataFrame, preset là chuỗi), nên không mất mát gì.

    Khôi phục trong `finally`, và chỉ khôi phục nếu thuộc tính vốn CÓ:
    dưới pytest/REPL `__main__` có thể không có `__file__`, và đặt thêm
    một `__file__` giả vào đó sẽ làm hỏng thứ khác. Không khôi phục được
    thì lần gọi sandbox sau, hoặc cơ chế reload của Streamlit, sẽ hỏng.
    """
    main = sys.modules.get("__main__")
    had_file = main is not None and hasattr(main, "__file__")
    original = getattr(main, "__file__", None) if main is not None else None
    if had_file:
        try:
            del main.__file__
        except (AttributeError, TypeError):
            had_file = False
    try:
        yield
    finally:
        if had_file:
            main.__file__ = original


def _run_sandboxed(
    code: str,
    df: pd.DataFrame,
    preset: str,
    timeout_seconds: float,
) -> Any:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_sandbox_worker, args=(code, df, preset, result_queue))
    with _main_module_not_reexecuted():
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
    timeout_seconds: float = CHART_EXEC_TIMEOUT_SECONDS,
) -> dict:
    """Chạy code vẽ biểu đồ, trả dict do code đặt vào biến `result`.

    Thay cho `run_dict_safe(code, local_vars)`: bản cũ nhận namespace tuỳ ý
    từ bên gọi, mà bên gọi duy nhất truyền vào bốn MODULE — thứ không pickle
    được nên không qua nổi ranh giới `spawn`. Giờ child tự dựng namespace
    từ preset.
    """
    return _run_sandboxed(code, df.copy(), PRESET_CHART, timeout_seconds)
