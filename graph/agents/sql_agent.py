"""
SQL Agent — generate and execute PostgreSQL queries with self-repair retry loop.
"""

from __future__ import annotations

import logging
import re
import time

import sqlparse
from pathlib import Path

from graph.budget import calls_remaining, clamped_tool_timeout_s, node_may_run
from graph.errors import BUDGET_EXCEEDED, MISSING_STEP, SQL_EXECUTION
from graph.state import MultiAgentState
from graph.tools.sql_tool import (
    SQL_CONNECT_TIMEOUT_S,
    SQL_TIMEOUT_MS,
    TableNotPermittedError,
    execute_sql,
    explain_query_plan,
)
from graph.utils import append_trace, df_to_state, with_routing
from model.model_client import BudgetExceededError, ModelClient
from perception.schema_context import SchemaContext
from perception.sql_identifiers import requote_for_tables

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "text_to_sql.txt"

SQL_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")

# Một lần sinh + một lần sửa theo error context. Lần thứ ba trước đây chỉ
# lặp lại cùng một lớp lỗi với giá một lời gọi model (spec 5.4).
MAX_RETRIES = 2


# Ranh giới từ ở CẢ HAI đầu: không có `\b` thì `WITH` khớp cả trong
# "GROUP BY WITHIN", và `SELECT` khớp trong "SELECTED".
_SQL_START = re.compile(r"\b(?:SELECT|WITH)\b", re.IGNORECASE)
# Hình dạng bắt buộc của một mệnh đề WITH thật: `WITH [RECURSIVE] <tên>
# [(cột,...)] AS (`. Văn xuôi "with a CTE:" không khớp.
# Từ khoá có thể mở đầu phần TIẾP THEO của một truy vấn sau dòng trống.
# Thấy một trong số này thì dòng trống chỉ là định dạng, không phải ranh
# giới giữa SQL và lời giải thích.
_SQL_CONTINUES = re.compile(
    r"\b(?:SELECT|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT|OFFSET|FETCH|JOIN|"
    r"LEFT|RIGHT|INNER|OUTER|FULL|CROSS|UNION|INTERSECT|EXCEPT|AND|OR|ON|"
    r"USING|WITH|WINDOW|QUALIFY)\b",
    re.IGNORECASE,
)
_CTE_HEAD = re.compile(
    r'WITH\s+(?:RECURSIVE\s+)?[\w"]+\s*(?:\([^)]*\))?\s*AS\s*\(',
    re.IGNORECASE,
)


def _extract_sql(text: str) -> str:
    """Bóc câu SQL ra khỏi đầu ra của model.

    Bản cũ tìm chuỗi con `"with"` bằng `str.find`, nên nó khớp chữ **with**
    trong văn xuôi tiếng Anh. Model viết "To find the school with the
    highest average score... you can use:" rồi mới tới SQL, và bộ bóc cắt
    ngay từ chữ "with" giữa câu văn, trả về một mớ không phải SQL.

    Đo trên 50 câu BIRD: 8 trong 18 ca SQL-sinh-lỗi (44%) là do đúng lỗi
    này, KHÔNG phải do model viết SQL sai. Prompt cấm văn xuôi (quy tắc 1
    và 3) nhưng model 7B không phải lúc nào cũng nghe.

    Ba lớp, theo thứ tự tin cậy giảm dần:

    1. Nội dung trong hàng rào ```sql — tín hiệu mạnh nhất, model đánh dấu
       sẵn đâu là code.
    2. Ứng viên bắt đầu ở RANH GIỚI TỪ, kiểm bằng `sqlparse.get_type()`.
       Nó phân biệt sạch: `WITH x AS (...) SELECT` ra "SELECT", còn
       "with the highest average" ra "UNKNOWN".
    3. Không có ứng viên nào hợp lệ thì fail closed ngay ở tầng
       sinh SQL, thay vì chuyển văn xuôi xuống tool và gán nhãn lỗi sai.

    Còn sót một ca: văn xuôi BẮT ĐẦU bằng "select" ("select the data you
    need") vẫn parse ra "SELECT". Hiếm hơn hẳn ca "with" vì nó đòi câu văn
    mở đầu bằng đúng từ đó, và chưa gặp lần nào trên 50 câu đã đo.
    """
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    body = fenced.group(1) if fenced else re.sub(r"```(?:sql)?\s*|\s*```", "", text)
    body = body.strip()

    for match in _SQL_START.finditer(body):
        candidate = body[match.start():].strip()
        if candidate[:4].upper() == "WITH" and not _CTE_HEAD.match(candidate):
            # `get_type()` một mình KHÔNG đủ ở đây: với "with a CTE:\nWITH
            # cte AS (...) SELECT ..." nó vẫn trả "SELECT", vì nó quét tới
            # khi gặp SELECT ở phía sau và không quan tâm phần đầu là văn
            # xuôi. Một mệnh đề WITH thật luôn có hình dạng
            # `WITH <tên> AS (`, nên kiểm hình dạng đó mới loại được.
            continue
        parsed = sqlparse.parse(candidate)
        if parsed and parsed[0].get_type() == "SELECT":
            return _cut_trailing_prose(candidate)
    raise ValueError(f"Không trích được SQL từ output của model: {body[:200]!r}")


def _cut_trailing_prose(sql: str) -> str:
    """Bỏ phần giải thích model viết SAU câu SQL.

    Sửa chỗ bắt đầu thôi chưa đủ: model hay viết xong SQL rồi giải thích
    tiếp, và lấy tới hết chuỗi thì nuốt cả đoạn văn. Đo trên 50 câu BIRD,
    8 ca lỗi cú pháp đều là dạng này — thông điệp đổi từ `near "highest"`
    (văn xuôi ĐẦU) sang `near "This"` (văn xuôi ĐUÔI) sau khi sửa nửa
    trước.

    Hai đường cắt, khớp đúng hai hình dạng đã quan sát được:

    1. Có dấu `;` — 7/8 ca. `sqlparse.split` cắt sạch ở câu lệnh đầu.
    2. Không có `;` — 1/8 ca. Cắt ở DÒNG TRỐNG, nhưng chỉ khi phần sau
       không mở đầu bằng từ khoá SQL. Không có điều kiện đó thì một truy
       vấn có dòng trống giữa các mệnh đề sẽ bị cắt cụt, và đó là hỏng
       nặng hơn hẳn thứ đang sửa.
    """
    parts = sqlparse.split(sql)
    if len(parts) > 1:
        return parts[0].strip()

    chunks = re.split(r"\n\s*\n", sql, maxsplit=1)
    if len(chunks) == 2 and not _SQL_CONTINUES.match(chunks[1].lstrip()):
        return chunks[0].strip()
    return sql.strip()


def build_system_prompt(schema_context: SchemaContext) -> str:
    """Render template với lát schema của lượt này.

    Few-shot để rỗng trong pha 1; plan 3 (onboarding) sẽ đổ vào.

    The template's trailing "Task: {task}" line (prompts/text_to_sql.txt)
    is dropped rather than substituted: sql_agent_node always sends the
    real task text as the user turn (`user_prompt = f"Task: {task}"` in the
    retry loop below), so leaving the placeholder unsubstituted would send
    a literal "{task}" to the model, and substituting it here would send
    the task twice (system prompt and user turn). Dropping it means the
    task text reaches the model exactly once, via the user turn — where it
    already lived before this function's {schema}/{few_shots} wiring was
    added.
    """
    few = "\n\n".join(
        f"Task: {fs['question']}\nOutput:\n{fs['sql']}"
        for fs in schema_context.few_shots
    )
    rendered = (SQL_SYSTEM_PROMPT
                .replace("{few_shots}", few)
                .replace("{schema}", schema_context.rendered_text))
    return rendered.replace("Task: {task}\n\n", "")


def _public_error_message(exc: Exception) -> str:
    """The safe-to-surface text for an exception raised during SQL execution.

    TableNotPermittedError (graph/tools/sql_tool.py) is structured: `.public`
    is the safe segment, `.forbidden` carries the exact forbidden table names
    — useful for logs, never for anything that ends up in state (trace
    observations, last_error, agent_outputs), since those can reach the UI.
    Reading `.public` on the exception object, rather than string-splitting
    on a "(nội bộ:" marker, means a future change to that marker (or to how
    the message is formatted) can't silently stop the redaction from working
    — the two are no longer coupled by string content. Exceptions without a
    `.public` attribute (i.e. not a TableNotPermittedError) fall back to
    str(exc) unchanged, so this is safe for every exception type this loop
    can see.
    """
    if isinstance(exc, TableNotPermittedError):
        return exc.public
    return str(exc)


def _find_sql_step(state: MultiAgentState) -> dict | None:
    """Find the sql step in the execution plan."""
    for step in state.get("execution_plan", []):
        if step.get("agent") == "sql":
            return step
    return None


def sql_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Generate SQL, execute, retry on failure (max 3).

    Returns state with sql_result, shared_dataframe, and updated trace.
    On fatal failure sets status='failed' and populates last_error.
    """
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

    step = _find_sql_step(state)
    if step is None:
        logger.error("SQL Agent: no sql step in execution plan")
        return {
            **state,
            "status": "failed",
            "last_error": {"agent": "sql", "error_type": MISSING_STEP},
        }

    task: str = step.get("task", "")
    schema_context = state["schema_context"]
    system_prompt = build_system_prompt(schema_context)

    trace = append_trace(state, "sql", "generate_sql",
                         f"Task: {task}", "started")
    state = {**state, "action_trace": trace}

    client = ModelClient(
        agent_type="sql",
        deadline_ts=state.get("deadline_ts"),
        call_budget_remaining=calls_remaining(state.get("llm_calls_used", 0)),
    )
    error_context: str = ""

    # Nhãn cho đường thất bại. Mặc định là lỗi thực thi SQL; chỉ đổi khi
    # nguyên nhân thật sự khác — xem nhánh BudgetExceededError bên dưới.
    error_label: str = SQL_EXECUTION

    for attempt in range(1, MAX_RETRIES + 1):
        user_prompt = f"Task: {task}"

        # Check for reflector-provided corrected context
        reflector_diag = state.get("shared_metadata", {}).get("reflector_diagnosis", {})
        reflector_hint = reflector_diag.get("corrected_context", "")
        if reflector_hint:
            user_prompt += f"\n\n[HINT from error diagnosis] {reflector_hint}"

        if error_context:
            user_prompt += f"\n\nPrevious attempt failed with error:\n{error_context}\n\nFix the error and rewrite the SQL."

        try:
            raw = client.invoke(system_prompt=system_prompt, user_prompt=user_prompt)
        except BudgetExceededError as exc:
            # PHẢI đứng trước `except Exception`: BudgetExceededError kế
            # thừa RuntimeError, nên nhánh chung nuốt nó và lượt chạy hết
            # giờ bị dán nhãn `sql_execution` — đọc log thì tưởng SQL hỏng,
            # trong khi thứ hỏng là ngân sách. Đó là lý do nhãn
            # `budget_exceeded` tồn tại; trước bản này không nơi nào phát ra
            # nó cả.
            error_label = BUDGET_EXCEEDED
            error_context = str(exc)
            logger.warning("SQL Agent attempt %d hết ngân sách: %s", attempt, error_context)
            continue
        except Exception as exc:
            error_context = f"ModelClient error: {exc}"
            logger.warning("SQL Agent attempt %d: %s", attempt, error_context)
            continue

        try:
            # Bọc nháy định danh TRƯỚC khi log và trước khi chạy: log phải là
            # đúng câu SQL được thực thi. No-op trên schema toàn chữ thường.
            sql = requote_for_tables(_extract_sql(raw), schema_context.tables)
        except ValueError as exc:
            error_context = str(exc)
            logger.warning("SQL Agent attempt %d: %s", attempt, error_context)
            trace = append_trace(state, "sql", "generate_sql",
                                 f"Attempt {attempt}: {error_context}", "error")
            # Ghi lại vào state: `append_trace` dựng danh sách mới từ
            # `state["action_trace"]`, nên vòng lặp sau mà vẫn đọc `state`
            # cũ sẽ dựng lại từ danh sách GỐC và ghi đè mất mục vừa thêm.
            # Không có dòng này, chỉ mục cuối cùng sống sót và mọi chẩn
            # đoán của các lần thử trước biến mất khỏi UI lẫn log JSONL.
            state = {**state, "action_trace": trace}
            continue
        logger.info("SQL Agent attempt %d:\n%s", attempt, sql)

        # ── Ngân sách còn lại SAU model, TRƯỚC khi chạm DB ─────────
        #
        # `node_may_run` ở đầu hàm chỉ gác cửa cho MỘT lời gọi model — nó
        # không thấy phần chi tiêu sau đó. Một lượt được cho qua vì còn đủ
        # ~15s cho model vẫn có thể tốn tới `SQL_TIMEOUT_MS` (10s) ở
        # statement_timeout ngay sau, và cả hai số đó độc lập với deadline
        # cho tới đây — cộng lại có thể vượt ngân sách toàn cục dù mọi cửa
        # gác đều nói "còn giờ". Tính lại thời gian còn lại NGAY TRƯỚC khi
        # chạm DB đóng lỗ đó.
        clock = state.get("_clock", time.time)
        deadline_ts = state.get("deadline_ts")
        sql_timeout_s = clamped_tool_timeout_s(deadline_ts, SQL_TIMEOUT_MS / 1000.0, clock=clock)
        if sql_timeout_s <= 0:
            error_label = BUDGET_EXCEEDED
            error_context = "Hết ngân sách thời gian sau khi model trả lời — không còn chỗ để chạy SQL."
            logger.warning("SQL Agent attempt %d: %s", attempt, error_context)
            continue
        # Sàn 1 giây: `connect_timeout`/`statement_timeout` bằng 0 với
        # psycopg2/Postgres nghĩa là TẮT trần, ngược hẳn ý định — không bao
        # giờ được để một clamp hợp lệ (còn dương) làm tròn xuống 0.
        connect_timeout_s = max(
            1, round(clamped_tool_timeout_s(deadline_ts, SQL_CONNECT_TIMEOUT_S, clock=clock))
        )

        # ── Execution ────────────────────────────────────────────
        try:
            meta = state.get("shared_metadata", {})
            df = execute_sql(
                sql, profile=meta["profile"], user=meta["user"],
                timeout_ms=max(1, int(sql_timeout_s * 1000)),
                connect_timeout_s=connect_timeout_s,
            )
        except Exception as exc:
            # Log the full exception (may include internal detail, e.g. the
            # forbidden-table list on TableNotPermittedError) server-side only.
            logger.warning("SQL Agent execution error (attempt %d): %s", attempt, exc)
            # Anything stored in state below can reach the UI via action_trace
            # or last_error — sanitize before it enters state.
            error_context = _public_error_message(exc)
            trace = append_trace(state, "sql", "execute_sql",
                                 f"Attempt {attempt} error: {error_context}", "error")
            # Ghi lại vào state: `append_trace` dựng danh sách mới từ
            # `state["action_trace"]`, nên vòng lặp sau mà vẫn đọc `state`
            # cũ sẽ dựng lại từ danh sách GỐC và ghi đè mất mục vừa thêm.
            # Không có dòng này, chỉ mục cuối cùng sống sót và mọi chẩn
            # đoán của các lần thử trước biến mất khỏi UI lẫn log JSONL.
            state = {**state, "action_trace": trace}
            continue

        # ── Success ───────────────────────────────────────────────
        # `explain_query_plan` chỉ để trang trí trace ("cost ~X") — không
        # đáng đánh đổi lấy thời gian ngân sách. Tính lại thời gian còn lại
        # (đã trôi thêm kể từ lúc chạy execute_sql ở trên) và BỎ QUA hẳn
        # nếu không còn chỗ, thay vì truyền timeout=0 xuống Postgres (nghĩa
        # là TẮT trần, ngược ý định) hay cứ gọi và hy vọng try/except cứu
        # kịp — connect() có thể treo trước khi có ngoại lệ nào để bắt.
        plan_timeout_s = clamped_tool_timeout_s(deadline_ts, SQL_TIMEOUT_MS / 1000.0, clock=clock)
        if plan_timeout_s <= 0:
            plan, cost = {}, "?"
        else:
            plan_connect_timeout_s = max(
                1, round(clamped_tool_timeout_s(deadline_ts, SQL_CONNECT_TIMEOUT_S, clock=clock))
            )
            try:
                plan = explain_query_plan(
                    sql, profile=meta["profile"], user=meta["user"],
                    timeout_ms=max(1, int(plan_timeout_s * 1000)),
                    connect_timeout_s=plan_connect_timeout_s,
                )
                cost = plan.get("total_cost_estimate", "?")
            except Exception:
                plan = {}
                cost = "?"

        truncated = bool(df.attrs.get("truncated"))
        truncated_note = " (BỊ CẮT ở trần dòng)" if truncated else ""
        trace = append_trace(state, "sql", "execute_sql",
                             f"OK — {len(df)} rows{truncated_note}, cost ~{cost}", "ok")
        logger.info("SQL Agent: %d rows returned", len(df))

        # Kết quả bị cắt ở trần dòng (df.attrs["truncated"], graph/tools/
        # sql_tool.py) trước bản này chỉ tới được `action_trace` — một
        # dòng chữ chôn trong danh sách trace, không phải
        # `degradation_reason`. `finalize_node` phân loại "success" khi
        # KHÔNG có lý do suy giảm nào (đúng bản sửa cho lỗi C1 review toàn
        # nhánh: một lượt bị cắt không được gọi là success) — nhưng nó chỉ
        # thấy được những gì NẰM TRONG `degradation_reason`. Một câu SELECT
        # trả về nhiều hơn `SQL_MAX_ROWS` (mặc định 50.000) dòng thì
        # "thành công" này là giả: người dùng nhận một bảng THIẾU dữ liệu
        # mà không có banner nào cảnh báo (app.py chỉ hiện banner cho
        # partial/failed).
        row_cap = df.attrs.get("row_cap")
        reasons = state.get("degradation_reason", [])
        if truncated:
            reasons = reasons + [
                f"Kết quả SQL bị cắt ở {row_cap} dòng — câu truy vấn trả về nhiều hơn thế."
            ]

        return with_routing({
            **state,
            "current_agent": "sql",
            "completed_agents": state.get("completed_agents", []) + ["sql"],
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "sql": {"status": "ok", "sql": sql, "row_count": len(df), "truncated": truncated},
            },
            "shared_dataframe": df_to_state(df),
            "sql_result": {"sql": sql, "row_count": len(df), "truncated": truncated, "row_cap": row_cap},
            "shared_metadata": {
                **state.get("shared_metadata", {}),
                "sql_row_count": len(df),
                "sql_query": sql,
                "sql_result": {"sql": sql, "row_count": len(df), "truncated": truncated, "row_cap": row_cap},
            },
            "degradation_reason": reasons,
            "action_trace": trace,
            "status": "running",
            # `client.calls_made`, không phải `attempt`: `attempt` chỉ đếm
            # số vòng lặp NGOÀI của node này, còn ModelClient có thể tự
            # retry hoặc rơi về OpenAI fallback bên trong MỘT vòng —
            # xem ghi chú ở model/model_client.py::ModelClient.__init__.
            "llm_calls_used": state.get("llm_calls_used", 0) + client.calls_made,
        })

    # ── Exhausted retries ─────────────────────────────────────
    error_summary = f"SQL Agent failed after {MAX_RETRIES} attempts. Last error: {error_context}"
    trace = append_trace(state, "sql", "execute_sql", error_summary, "error")

    return {
        **state,
        "current_agent": "sql",
        "agent_outputs": {
            **state.get("agent_outputs", {}),
            "sql": {"status": "error", "error": error_summary},
        },
        "error_count": state.get("error_count", 0) + 1,
        "agent_error_counts": {
            **state.get("agent_error_counts", {}),
            "sql": state.get("agent_error_counts", {}).get("sql", 0) + 1,
        },
        "last_error": {
            "agent": "sql",
            "error_type": error_label,
            "traceback": error_summary,
        },
        "action_trace": trace,
        "status": "running",
        "llm_calls_used": state.get("llm_calls_used", 0) + client.calls_made,
    }
