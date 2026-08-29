from __future__ import annotations

import numpy as np
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


def add_cross_sectional_return_label(
    frame: pd.DataFrame,
    horizon: int = 5,
    transform: str = "return",
    quantiles: int = 5,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Add a forward-return label transformed within each daily cross-section."""
    transform = transform.strip().lower().replace("-", "_")
    data = add_forward_return_label(frame, horizon=horizon)
    raw_label = f"future_return_{horizon}d"

    if transform == "return":
        return data, raw_label

    label_column = label_column or f"{raw_label}_{transform}"
    grouped = data.groupby("date")[raw_label]
    if transform == "rank":
        data[label_column] = grouped.rank(pct=True)
    elif transform == "zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        data[label_column] = (data[raw_label] - mean) / std
    elif transform == "quantile":
        data[label_column] = grouped.transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                q=min(quantiles, values.notna().sum()),
                labels=False,
                duplicates="drop",
            )
            if values.notna().sum() >= 2
            else np.nan
        )
    else:
        raise ValueError("label transform must be one of: return, rank, zscore, quantile")

    return data, label_column
