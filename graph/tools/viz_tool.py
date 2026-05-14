"""
Visualization tool — matplotlib chart → base64 PNG.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

CHART_SIZE = (10, 6)
DPI = 150
ANOMALY_COLOR = "#E24B4A"
DEFAULT_COLOR = "#4A72B0"


def generate_chart(
    df: pd.DataFrame,
    chart_type: str | None = None,
    title: str = "",
    x_col: str = "",
    y_col: str = "",
    group_col: str = "",
) -> dict[str, Any]:
    """Generate a matplotlib chart from a DataFrame, return base64 PNG + metadata.

    Args:
        df: Input DataFrame.
        chart_type: One of 'bar', 'line', 'scatter', 'hist', 'pie', 'hbar'.
                    Auto-detected if None.
        title: Chart title.
        x_col: Column for x-axis.
        y_col: Column for y-axis.
        group_col: Column for grouping (multi-series).

    Returns:
        {"chart_b64": str, "chart_type": str, "metadata": dict}
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=CHART_SIZE)

    if x_col and y_col:
        if chart_type is None:
            chart_type = _auto_chart_type(df, x_col, y_col)

        if chart_type in ("bar", "hbar"):
            _draw_bar(df, ax, x_col, y_col, group_col, chart_type)
        elif chart_type == "line":
            _draw_line(df, ax, x_col, y_col, group_col)
        elif chart_type == "scatter":
            _draw_scatter(df, ax, x_col, y_col, group_col)
        elif chart_type == "hist":
            _draw_hist(df, ax, y_col)
        elif chart_type == "pie":
            _draw_pie(df, ax, x_col, y_col)
        else:
            _draw_bar(df, ax, x_col, y_col, group_col, "bar")

    ax.set_title(title, fontsize=14, fontweight="bold")
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel())
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel())
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "chart_b64": chart_b64,
        "chart_type": chart_type or "bar",
        "chart_metadata": {
            "chart_type": chart_type,
            "x_col": x_col,
            "y_col": y_col,
            "title": title,
            "dpi": DPI,
        },
    }


def _auto_chart_type(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    """Auto-select chart type based on column characteristics."""
    nunique = df[x_col].nunique()
    if nunique <= 6 and df[y_col].dtype in ("int64", "float64"):
        return "pie"
    if "date" in x_col.lower() or "time" in x_col.lower():
        return "line"
    if nunique <= 10:
        max_len = df[x_col].astype(str).str.len().max()
        return "hbar" if max_len > 15 else "bar"
    return "bar"


def _draw_bar(df, ax, x_col, y_col, group_col, chart_type):
    """Grouped or single bar chart. Highlights is_anomaly in red."""
    if group_col and group_col in df.columns:
        pivoted = df.pivot(index=x_col, columns=group_col, values=y_col)
        pivoted.plot(kind="bar" if chart_type == "bar" else "barh", ax=ax)
    else:
        colors = _anomaly_colors(df)
        kind = "bar" if chart_type == "bar" else "barh"
        bars = ax.bar(df[x_col].astype(str), df[y_col], color=colors)
        ax.bar_label(bars, fmt="{:.0f}" if df[y_col].dtype == "int64" else "{:.1f}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def _draw_line(df, ax, x_col, y_col, group_col):
    if group_col and group_col in df.columns:
        for name, grp in df.groupby(group_col):
            ax.plot(grp[x_col], grp[y_col], marker="o", label=str(name))
        ax.legend()
    else:
        ax.plot(df[x_col], df[y_col], marker="o", color=DEFAULT_COLOR)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def _draw_scatter(df, ax, x_col, y_col, group_col):
    if group_col and group_col in df.columns:
        for name, grp in df.groupby(group_col):
            ax.scatter(grp[x_col], grp[y_col], label=str(name))
        ax.legend()
    else:
        colors = _anomaly_colors(df)
        ax.scatter(df[x_col], df[y_col], c=colors)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def _draw_hist(df, ax, col):
    ax.hist(df[col].dropna(), bins=20, color=DEFAULT_COLOR, edgecolor="white")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")


def _draw_pie(df, ax, x_col, y_col):
    ax.pie(df[y_col], labels=df[x_col].astype(str), autopct="%1.1f%%",
           startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(df))))
    ax.set_ylabel("")


def _anomaly_colors(df):
    """Return list of colors — red for is_anomaly=True, blue otherwise."""
    if "is_anomaly" in df.columns:
        return [ANOMALY_COLOR if v else DEFAULT_COLOR for v in df["is_anomaly"]]
    return [DEFAULT_COLOR] * len(df)
