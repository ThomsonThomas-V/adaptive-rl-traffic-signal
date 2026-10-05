"""Tests for evaluation aggregation."""

import pandas as pd

from traffic_rl.evaluation.metrics import summarize_results


def test_summary_calculates_mean_and_sample_std() -> None:
    frame = pd.DataFrame(
        {
            "controller": ["Fixed", "Fixed"],
            "scenario": ["low", "low"],
            "average_waiting_time": [2.0, 4.0],
            "throughput": [10, 12],
        }
    )
    summary = summarize_results(frame)
    assert summary.loc[0, "average_waiting_time_mean"] == 3.0
    assert summary.loc[0, "throughput_mean"] == 11.0
