from __future__ import annotations

import pandas as pd

from us_alpha_lab.validation import walk_forward_splits


def test_walk_forward_splits_with_embargo() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    splits = walk_forward_splits(dates, train_size=10, test_size=5, embargo=2)

    assert splits
    first = splits[0]
    assert first.train_end < first.test_start
    assert (dates.get_loc(first.test_start) - dates.get_loc(first.train_end)) == 3
