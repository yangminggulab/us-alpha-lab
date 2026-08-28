from __future__ import annotations

import pandas as pd


def add_forward_return_label(
    frame: pd.DataFrame,
    horizon: int = 5,
    label_column: str | None = None,
) -> pd.DataFrame:
    """Add a forward close-to-close return label."""
    data = frame.sort_values(["ticker", "date"]).copy()
    label_column = label_column or f"future_return_{horizon}d"

    next_close = data.groupby("ticker")["close"].shift(-horizon)
    data[label_column] = next_close / data["close"] - 1
    return data
