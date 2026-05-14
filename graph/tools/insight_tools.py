"""
Insight analysis tools — anomaly detection, period comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_anomaly(
    df: pd.DataFrame,
    col: str,
) -> dict:
    """Detect anomalies in a numeric column using the IQR method.

    Returns:
        {"anomalies": list[dict], "thresholds": {"Q1": float, "Q3": float, "IQR": float,
         "lower": float, "upper": float}, "anomaly_count": int}
    """
    series = df[col].dropna()
    if len(series) < 2:
        return {"anomalies": [], "thresholds": {}, "anomaly_count": 0}

    Q1 = float(series.quantile(0.25))
    Q3 = float(series.quantile(0.75))
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    anomalies = []
    for _, row in df.iterrows():
        val = row[col]
        if pd.notna(val) and (val < lower or val > upper):
            sigma = abs(val - series.mean()) / series.std() if series.std() > 0 else 0
            direction = "positive_outlier" if val > upper else "negative_outlier"
            anomalies.append({
                "region": row.get("region", row.get("name", str(row.name))),
                "value": float(val),
                "sigma": round(sigma, 2),
                "direction": direction,
            })

    return {
        "anomalies": anomalies,
        "thresholds": {"Q1": Q1, "Q3": Q3, "IQR": IQR, "lower": lower, "upper": upper},
        "anomaly_count": len(anomalies),
    }


def compare_periods(
    df: pd.DataFrame,
    current_col: str,
    previous_col: str,
    group_col: str = "region",
) -> pd.DataFrame:
    """Compute period-over-period change as a percentage.

    Returns DataFrame with added pct_change column and is_anomaly flag.
    """
    df = df.copy()
    mask = df[previous_col].notna() & (df[previous_col] > 0)
    df["pct_change"] = np.where(
        mask,
        ((df[current_col] - df[previous_col]) / df[previous_col] * 100).round(2),
        np.nan,
    )

    Q1 = df["pct_change"].quantile(0.25)
    Q3 = df["pct_change"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["is_anomaly"] = ((df["pct_change"] < lower) | (df["pct_change"] > upper))

    return df
