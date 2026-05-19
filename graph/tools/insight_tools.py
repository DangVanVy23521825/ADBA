"""
Insight analysis tools — anomaly detection, period comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_anomaly(
    df: pd.DataFrame,
    col: str,
    *,
    sigma_threshold: float = 2.0,
    method: str = "sigma",
) -> dict:
    """Detect anomalies in a numeric column using sigma and optional IQR bounds.

    Returns:
        JSON-safe summary with anomalies, thresholds, method, and anomaly_count.
    """
    if col not in df.columns:
        raise KeyError(f"column not found: {col}")

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(series) < 2:
        return {"anomalies": [], "thresholds": {}, "method": method, "anomaly_count": 0}

    Q1 = float(series.quantile(0.25))
    Q3 = float(series.quantile(0.75))
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    sigma_lower = mean - sigma_threshold * std
    sigma_upper = mean + sigma_threshold * std

    anomalies = []
    for _, row in df.iterrows():
        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.isna(val):
            continue

        sigma = abs(float(val) - mean) / std if std > 0 else 0.0
        iqr_hit = bool(val < lower or val > upper)
        sigma_hit = bool(std > 0 and sigma >= sigma_threshold)

        if (method == "iqr" and iqr_hit) or (method == "sigma" and sigma_hit) or (method == "hybrid" and (iqr_hit or sigma_hit)):
            direction = "positive_outlier" if float(val) > mean else "negative_outlier"
            anomalies.append({
                "region": row.get("region", row.get("name", str(row.name))),
                "value": float(val),
                "sigma": round(sigma, 2),
                "direction": direction,
                "reasons": {
                    "sigma": sigma_hit,
                    "iqr": iqr_hit,
                },
            })

    return {
        "anomalies": anomalies,
        "thresholds": {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "iqr_lower": float(lower),
            "iqr_upper": float(upper),
            "mean": mean,
            "std": std,
            "sigma_lower": float(sigma_lower),
            "sigma_upper": float(sigma_upper),
            "sigma_threshold": float(sigma_threshold),
            "lower": float(sigma_lower if method == "sigma" else lower),
            "upper": float(sigma_upper if method == "sigma" else upper),
        },
        "method": method,
        "anomaly_count": len(anomalies),
    }


def compare_periods(
    df: pd.DataFrame,
    current_col: str,
    previous_col: str,
    group_col: str = "region",
    period_type: str = "YoY",
    sigma_threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute YoY/QoQ period-over-period change as a percentage.

    Returns DataFrame with added pct_change column and is_anomaly flag.
    """
    df = df.copy()
    if current_col not in df.columns or previous_col not in df.columns:
        raise KeyError("current_col and previous_col must exist in DataFrame")

    normalized_period = period_type.upper()
    if normalized_period not in {"YOY", "QOQ"}:
        raise ValueError("period_type must be 'YoY' or 'QoQ'")

    mask = df[previous_col].notna() & (df[previous_col] > 0)
    df["pct_change"] = np.where(
        mask,
        ((df[current_col] - df[previous_col]) / df[previous_col] * 100).round(2),
        np.nan,
    )
    df["period_comparison"] = normalized_period

    valid_change = df["pct_change"].dropna()
    mean = valid_change.mean()
    std = valid_change.std(ddof=0)

    if len(valid_change) >= 2 and std > 0:
        df["change_sigma"] = ((df["pct_change"] - mean).abs() / std).round(2)
        df["is_anomaly"] = df["change_sigma"] >= sigma_threshold
    else:
        df["change_sigma"] = np.nan
        df["is_anomaly"] = False

    if group_col in df.columns:
        cols = [group_col] + [c for c in df.columns if c != group_col]
        df = df[cols]

    return df
