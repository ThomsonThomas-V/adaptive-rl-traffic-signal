"""Metric aggregation helpers."""

from __future__ import annotations

import pandas as pd


METRIC_COLUMNS = [
    "average_waiting_time",
    "max_queue_length",
    "average_queue_length",
    "throughput",
    "completion_ratio",
    "signal_switches",
    "total_waiting_time",
    "episode_reward",
]


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate episode metrics to mean and sample standard deviation."""
    if results.empty:
        return pd.DataFrame()
    present = [column for column in METRIC_COLUMNS if column in results.columns]
    summary = (
        results.groupby(["controller", "scenario"], sort=False)[present]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary
