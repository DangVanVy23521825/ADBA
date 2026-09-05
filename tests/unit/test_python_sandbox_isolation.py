"""Sandbox: ranh giới là tiến trình.

Test rò rỉ env dùng `pd.io.common.os` — module `os` THẬT, lấy qua pandas
mà không đi qua `__import__`, nên whitelist import không nhìn thấy nó.
Đây chính là điểm yếu 1 trong spec mục 4.2, và nó là đường rò có thật
chứ không phải mẹo escape mong manh: đã kiểm trên pandas 2.3.3, truy được
34 biến môi trường của tiến trình cha trước bản sửa này.
"""

from __future__ import annotations

import os
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


# ── entry point: child KHÔNG được chạy lại __main__ ─────────────────────────

_ENTRY_SCRIPT = '''\
"""Giả lập entry point kiểu `streamlit run app.py`.

Điểm mấu chốt: phần thân module KHÔNG nằm trong `if __name__ ==
"__main__":`. Streamlit biến app.py thành `__main__`, và app.py có
import/khởi tạo ở cấp module — đúng hình dạng này. Chỉ LỜI GỌI sandbox
được bọc, vì nếu không thì child sẽ tự gọi lại sandbox và multiprocessing
ném lỗi bootstrap thay vì thể hiện đúng cái giá phải trả.

Mỗi lần thân module chạy, ghi một dòng vào file đếm. Cha chạy nó đúng một
lần. Nếu `multiprocessing.spawn` bắt child chạy lại `__main__`, file sẽ có
hai dòng — đó chính là lỗi, đo được mà không phụ thuộc đồng hồ.
"""
import os
import pathlib
import sys

with pathlib.Path(os.environ["ADBA_MAIN_EXEC_COUNTER"]).open("a") as fh:
    fh.write(__name__ + "\\n")

sys.path.insert(0, os.environ["ADBA_REPO_ROOT"])

import pandas as pd

from graph.tools.python_tool import run_pandas_safe

if __name__ == "__main__":
    out = run_pandas_safe(
        "df = df.copy()\\ndf['b'] = df['a'] * 2\\ndf",
        pd.DataFrame({"a": [1, 2, 3]}),
    )
    print("ROWS", len(out))
'''


def test_the_child_does_not_re_execute_the_main_module(tmp_path):
    """Hồi quy cho C3.

    Dưới `spawn`, mỗi child dựng lại trạng thái bằng cách chạy lại
    `sys.modules["__main__"]` qua runpy. Với `streamlit run app.py`, thân
    app.py nằm ngoài mọi lớp bọc, nên MỖI lời gọi sandbox lại import
    streamlit + langgraph + matplotlib và dựng lại graph trong child trước
    khi chạy một dòng code nào của model — vài giây mỗi lần, đủ để sandbox
    báo "Python execution timed out" cho code hoàn toàn đúng.

    Bộ test chạy dưới pytest KHÔNG bắt được lỗi này: ở đó `__main__` là
    entry point của pytest, và `__main__.__spec__.name` kết thúc bằng
    `.__main__` nên spawn bỏ qua bước fixup. Vì vậy test này dựng một entry
    point riêng và chạy nó như một tiến trình thật.
    """
    import subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    script = tmp_path / "fake_streamlit_app.py"
    script.write_text(_ENTRY_SCRIPT, encoding="utf-8")
    counter = tmp_path / "main_execs.txt"

    env = {
        **os.environ,
        "ADBA_MAIN_EXEC_COUNTER": str(counter),
        "ADBA_REPO_ROOT": str(repo_root),
    }
    proc = subprocess.run(
        [_sys.executable, str(script)],
        capture_output=True, text=True, timeout=180, env=env,
    )

    assert proc.returncode == 0, f"entry point chết:\n{proc.stdout}\n{proc.stderr}"
    assert "ROWS 3" in proc.stdout, f"sandbox không trả kết quả:\n{proc.stdout}"

    runs = counter.read_text(encoding="utf-8").split()
    assert runs == ["__main__"], (
        "thân module __main__ chạy nhiều hơn một lần — child của spawn đang "
        f"nạp lại toàn bộ entry point: {runs}"
    )
