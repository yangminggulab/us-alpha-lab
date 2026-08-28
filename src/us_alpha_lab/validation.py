from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    dates: list[pd.Timestamp] | pd.Series,
    train_size: int = 252,
    test_size: int = 63,
    embargo: int = 5,
    step_size: int | None = None,
) -> list[WalkForwardSplit]:
    """Create rolling walk-forward splits from sorted unique dates."""
    unique_dates = pd.Index(pd.to_datetime(dates)).drop_duplicates().sort_values()
    step_size = step_size or test_size
    splits: list[WalkForwardSplit] = []
    start = 0

    while True:
        train_start_idx = start
        train_end_idx = train_start_idx + train_size - 1
        test_start_idx = train_end_idx + embargo + 1
        test_end_idx = test_start_idx + test_size - 1
        if test_end_idx >= len(unique_dates):
            break

        splits.append(
            WalkForwardSplit(
                train_start=pd.Timestamp(unique_dates[train_start_idx]),
                train_end=pd.Timestamp(unique_dates[train_end_idx]),
                test_start=pd.Timestamp(unique_dates[test_start_idx]),
                test_end=pd.Timestamp(unique_dates[test_end_idx]),
            )
        )
        start += step_size

    return splits
