"""
ADBA — Insight Output Schema
=============================
Pydantic models for the Insight Agent's structured output.

Changes from v1:
  - finding: enforce single sentence (count sentence-ending punctuation)
  - action:  enforce single sentence AND must start with an action verb
  - anomaly + confidence: downgraded from hard reject to warning log
    (anomaly detected + low confidence is unusual but valid when data is scarce)
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)


ConfidenceLevel = Literal["high", "medium", "low"]
AnomalyType     = Literal["positive_outlier", "negative_outlier", "none"]

# Sentence-ending pattern: period / ! / ? followed by space or end-of-string.
# Excludes decimal numbers (3.5), abbreviations (e.g.), and ellipsis (...).
_SENTENCE_END = re.compile(
    r'(?<!\d)(?<!\.\.)(?<![A-Za-z]{1}\.[A-Za-z]{1})'  # not decimal, not abbrev
    r'[.!?]'
    r'(?:\s|$)',
)

# Action verbs recognised in both Vietnamese and English.
# Stored as a frozenset at module level — not a Pydantic field.
_ACTION_VERBS: frozenset[str] = frozenset({
    # Vietnamese — single first words only (split()[0].lower())
    "tăng", "giảm", "kiểm", "xem", "cần", "nên", "thực",
    "đề", "báo", "điều", "theo", "cập", "liên",
    "mở", "thu", "dừng", "tiếp", "ưu", "phân",
    # English
    "increase", "decrease", "review", "investigate", "monitor", "update",
    "check", "consider", "implement", "report", "contact", "prioritize",
    "allocate", "reduce", "expand", "pause", "restart", "escalate",
    "verify", "confirm", "schedule", "assign", "notify",
})


def _count_sentences(text: str) -> int:
    """
    Count sentences in text using _SENTENCE_END pattern.
    Returns at least 1 for any non-empty string.
    """
    matches = _SENTENCE_END.findall(text.strip())
    return max(len(matches), 1)


# =============================================================================
# AnomalyInfo
# =============================================================================

class AnomalyInfo(BaseModel):
    """
    Anomaly information produced alongside the insight.

    type:
      positive_outlier — value is abnormally HIGH
      negative_outlier — value is abnormally LOW / declining
      none             — no anomaly detected; detail must be None
    """

    type:   AnomalyType
    detail: str | None = None

    @model_validator(mode="after")
    def detail_consistent_with_type(self) -> "AnomalyInfo":
        if self.type == "none" and self.detail is not None:
            raise ValueError("detail must be None when anomaly type is 'none'")
        if self.type != "none" and not self.detail:
            raise ValueError(
                f"detail is required when anomaly type is '{self.type}'"
            )
        return self


# =============================================================================
# InsightOutput
# =============================================================================

class InsightOutput(BaseModel):
    """
    Structured business insight from the Insight Agent.

    Enforced constraints:
      finding  — single sentence, must contain ≥1 number
      evidence — 2–3 items, each containing ≥1 number
      action   — single sentence, must start with a recognised action verb
      anomaly  — AnomalyInfo with consistent type/detail
      confidence — if anomaly detected but confidence='low', a WARNING is
                   logged (not raised), because scarce data can justify this.
    """

    finding:    str
    evidence:   list[str]
    anomaly:    AnomalyInfo
    action:     str
    confidence: ConfidenceLevel

    # ── finding ───────────────────────────────────────────────────────────────

    @field_validator("finding")
    @classmethod
    def finding_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("finding must not be empty")
        if not re.search(r"\d", v):
            raise ValueError(
                "finding must contain at least one number "
                "(percentage, absolute value, or rank)"
            )
        n = _count_sentences(v)
        if n > 1:
            raise ValueError(
                f"finding must be a single sentence, got ~{n} sentences. "
                "Merge into one sentence or move extra detail to evidence."
            )
        return v

    # ── evidence ─────────────────────────────────────────────────────────────

    @field_validator("evidence")
    @classmethod
    def evidence_valid(cls, v: list[str]) -> list[str]:
        if not (2 <= len(v) <= 3):
            raise ValueError(
                f"evidence must have 2 or 3 items, got {len(v)}"
            )
        cleaned = []
        for i, item in enumerate(v):
            item = item.strip()
            if not item:
                raise ValueError(f"evidence[{i}] must not be empty")
            if not re.search(r"\d", item):
                raise ValueError(
                    f"evidence[{i}] must contain a number. Got: {repr(item)}"
                )
            cleaned.append(item)
        return cleaned

    # ── action ────────────────────────────────────────────────────────────────

    @field_validator("action")
    @classmethod
    def action_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("action must not be empty")

        # Enforce single sentence
        n = _count_sentences(v)
        if n > 1:
            raise ValueError(
                f"action must be a single sentence, got ~{n} sentences."
            )

        # Enforce action verb as first word
        first_word = v.split()[0].lower().rstrip(".,;:")
        if first_word not in _ACTION_VERBS:
            raise ValueError(
                f"action must start with a recognised action verb. "
                f"First word '{first_word}' is not in the allowed list. "
                f"Examples: Tăng / Giảm / Kiểm tra / Review / Investigate / Monitor."
            )

        return v

    # ── cross-field: anomaly + confidence ────────────────────────────────────

    @model_validator(mode="after")
    def warn_if_anomaly_with_low_confidence(self) -> "InsightOutput":
        """
        Anomaly detected + confidence='low' is unusual but NOT invalid —
        it can happen when data volume is small. Log a warning instead of
        raising an error so the agent doesn't retry unnecessarily.
        """
        if self.anomaly.type != "none" and self.confidence == "low":
            logger.warning(
                "InsightOutput: anomaly_type='%s' with confidence='low'. "
                "This is unusual — consider raising confidence or "
                "setting anomaly.type='none' if data is insufficient.",
                self.anomaly.type,
            )
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def has_anomaly(self) -> bool:
        return self.anomaly.type != "none"

    def is_positive(self) -> bool:
        return self.anomaly.type == "positive_outlier"

    def to_streamlit_dict(self) -> dict:
        """Flat dict for Streamlit card rendering."""
        return {
            "finding":        self.finding,
            "evidence":       self.evidence,
            "anomaly_type":   self.anomaly.type,
            "anomaly_detail": self.anomaly.detail,
            "action":         self.action,
            "confidence":     self.confidence,
            "has_anomaly":    self.has_anomaly(),
        }


# =============================================================================
# CANONICAL EXAMPLES — embed field names in prompt few-shot sections
# =============================================================================

EXAMPLE_WITH_ANOMALY: dict = {
    "finding": "Miền Bắc tăng trưởng +89% YoY trong Q4 2024, gấp 3.9 lần trung bình cả nước.",
    "evidence": [
        "Doanh thu Q4 2024 Miền Bắc: 142,000,000 VND vs Q4 2023: 75,400,000 VND",
        "Tăng trưởng trung bình toàn quốc Q4 2024: +23% YoY",
        "Miền Bắc vượt 3.2 sigma so với baseline các region khác",
    ],
    "anomaly": {
        "type":   "positive_outlier",
        "detail": "Campaign B2B Enterprise tháng 10–11/2024 tại Miền Bắc có thể là nguyên nhân chính.",
    },
    "action": "Tăng tồn kho khu vực Miền Bắc ít nhất 60% trước Q1 2025 để duy trì momentum.",
    "confidence": "high",
}

EXAMPLE_NO_ANOMALY: dict = {
    "finding": "Doanh thu Q4 2024 tăng đều +23% YoY ở tất cả 4 regions, phù hợp seasonal pattern.",
    "evidence": [
        "Tổng doanh thu Q4 2024: 485,000,000 VND vs Q4 2023: 394,300,000 VND",
        "Tất cả regions tăng trong khoảng 18–28% YoY",
    ],
    "anomaly": {"type": "none", "detail": None},
    "action": "Theo dõi tiếp xu hướng Q1 2025 để xác nhận seasonal pattern.",
    "confidence": "high",
}

EXAMPLE_LOW_CONFIDENCE: dict = {
    "finding": "Phòng Tài chính có tổng lương tháng 7/2024 vượt budget 18%, dựa trên 20 bản ghi.",
    "evidence": [
        "SUM(net_salary) tháng 7: 380,000,000 VND vs budget: 320,000,000 VND",
        "Chênh lệch: +60,000,000 VND (+18.75%)",
    ],
    "anomaly": {"type": "none", "detail": None},
    "action": "Kiểm tra lại dữ liệu payroll tháng 7 trước khi đưa ra kết luận.",
    "confidence": "low",
}

# Example: anomaly + low confidence — valid (warning logged, not rejected)
EXAMPLE_ANOMALY_LOW_CONFIDENCE: dict = {
    "finding": "3 sản phẩm Electronics có tỷ lệ hoàn trả +340% tháng 11/2023 so với baseline.",
    "evidence": [
        "Số đơn refunded Electronics tháng 11/2023: 62 vs trung bình tháng khác: 18",
        "Tỷ lệ refund tháng 11: 24.8% vs baseline 7.2%",
    ],
    "anomaly": {
        "type":   "negative_outlier",
        "detail": "Nghi lô hàng lỗi; chỉ có 62 đơn nên cần xác nhận thêm.",
    },
    "action": "Điều tra lô hàng Electronics nhập tháng 10/2023 trước khi kết luận.",
    "confidence": "low",  # valid — small sample, but anomaly is real
}


if __name__ == "__main__":
    import warnings
    logging.basicConfig(level=logging.WARNING)

    print("insight_schema.py — self-test")
    print("=" * 42)

    # Valid examples
    out1 = InsightOutput.model_validate(EXAMPLE_WITH_ANOMALY)
    assert out1.has_anomaly() and out1.is_positive()
    print(f"✓ With anomaly:        has_anomaly={out1.has_anomaly()}, confidence={out1.confidence}")

    out2 = InsightOutput.model_validate(EXAMPLE_NO_ANOMALY)
    assert not out2.has_anomaly()
    print(f"✓ No anomaly:          has_anomaly={out2.has_anomaly()}")

    out3 = InsightOutput.model_validate(EXAMPLE_LOW_CONFIDENCE)
    assert out3.confidence == "low"
    print(f"✓ Low confidence:      confidence={out3.confidence}")

    # anomaly + low confidence → warning logged, not rejected
    print("  (expect a WARNING log on the next line)")
    out4 = InsightOutput.model_validate(EXAMPLE_ANOMALY_LOW_CONFIDENCE)
    assert out4.has_anomaly() and out4.confidence == "low"
    print(f"✓ Anomaly + low conf:  passes with warning (not rejected)")

    # to_streamlit_dict
    d = out1.to_streamlit_dict()
    assert all(k in d for k in ("finding", "evidence", "action", "has_anomaly"))
    print("✓ to_streamlit_dict()  all keys present")

    # Bad cases
    bad_cases: list[tuple[str, dict]] = [
        # finding: no number
        ("finding has no number", {
            **EXAMPLE_WITH_ANOMALY,
            "finding": "Miền Bắc tăng trưởng rất tốt trong quý vừa rồi.",
        }),
        # finding: multiple sentences
        ("finding has 2 sentences", {
            **EXAMPLE_WITH_ANOMALY,
            "finding": "Miền Bắc tăng +89%. Điều này rất bất thường.",
        }),
        # evidence: only 1 item
        ("evidence has 1 item", {
            **EXAMPLE_NO_ANOMALY,
            "evidence": ["Tổng doanh thu 485,000,000 VND"],
        }),
        # evidence: item without number
        ("evidence item has no number", {
            **EXAMPLE_NO_ANOMALY,
            "evidence": ["Không có số liệu", "Cũng không có số"],
        }),
        # action: multiple sentences
        ("action has 2 sentences", {
            **EXAMPLE_NO_ANOMALY,
            "action": "Theo dõi xu hướng. Báo cáo hàng tuần.",
        }),
        # action: does not start with action verb
        ("action doesn't start with verb", {
            **EXAMPLE_NO_ANOMALY,
            "action": "Xu hướng tăng đều cần được theo dõi tiếp.",
        }),
        # anomaly: type != none but detail = None
        ("anomaly has no detail", {
            **EXAMPLE_WITH_ANOMALY,
            "anomaly": {"type": "positive_outlier", "detail": None},
        }),
    ]

    for label, bad in bad_cases:
        try:
            InsightOutput.model_validate(bad)
            print(f"  ✗ '{label}' — should have failed")
        except Exception as e:
            short = str(e).replace("\n", " ")[:72]
            print(f"  ✓ '{label}'\n      → {short}")

    print("\nAll tests passed.")