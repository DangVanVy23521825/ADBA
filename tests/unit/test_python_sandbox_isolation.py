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
