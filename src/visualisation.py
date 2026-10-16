"""Small plotting helpers shared by the notebooks and the Streamlit app.

Kept deliberately thin: each function takes already-computed numbers or frames
and returns a Matplotlib figure, so the teaching logic stays in ``src`` and the
notebooks stay about the story.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.evaluation.leakage import LeakageResult


def plot_leakage_bars(result: LeakageResult) -> Figure:
    """Four-bar chart: naive vs honest, offline vs true-future."""
    labels = ["naive\noffline", "honest\noffline", "naive\nfuture", "honest\nfuture"]
    values = [
        result.naive_offline_auc,
        result.honest_offline_auc,
        result.naive_future_auc,
        result.honest_future_auc,
    ]
    colors = ["#c0392b", "#27ae60", "#c0392b", "#27ae60"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC")
    ax.set_title("The naive offline number was a mirage")
    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_time_travel(
    history: pd.DataFrame,
    event_timestamp: pd.Timestamp,
    timestamp_col: str,
    feature_col: str,
) -> Figure:
    """A single entity's feature over time, marking the point-in-time cut."""
    ordered = history.sort_values(timestamp_col)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(ordered[timestamp_col], ordered[feature_col], where="post")
    # matplotlib accepts datetimes on a date axis at runtime; the stub says float.
    ax.axvline(event_timestamp, color="#c0392b", linestyle="--", label="event time")  # ty: ignore[invalid-argument-type]
    ax.set_xlabel("time")
    ax.set_ylabel(feature_col)
    ax.set_title("Point-in-time value vs the latest value")
    ax.legend()
    fig.tight_layout()
    return fig
