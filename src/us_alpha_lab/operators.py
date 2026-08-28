from __future__ import annotations

import numpy as np
import pandas as pd


def delay(series: pd.Series, periods: int = 1) -> pd.Series:
    """Lag a per-ticker series."""
    return series.groupby(level=0, group_keys=False).shift(periods)


def delta(series: pd.Series, periods: int = 1) -> pd.Series:
    """Difference a per-ticker series."""
    return series.groupby(level=0, group_keys=False).diff(periods)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Rolling per-ticker mean."""
    return series.groupby(level=0, group_keys=False).rolling(window).mean().droplevel(0)


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling per-ticker standard deviation."""
    return series.groupby(level=0, group_keys=False).rolling(window).std().droplevel(0)


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    """Rolling per-ticker maximum."""
    return series.groupby(level=0, group_keys=False).rolling(window).max().droplevel(0)


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling time-series percentile rank of the latest value."""
    return series.groupby(level=0, group_keys=False).rolling(window).rank(pct=True).droplevel(0)


def rolling_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """Rolling per-ticker correlation."""
    pieces = []
    names = left.index.names
    for ticker, left_group in left.groupby(level=0, sort=False):
        right_group = right.xs(ticker, level=0)
        corr = left_group.droplevel(0).rolling(window).corr(right_group)
        corr.index = pd.MultiIndex.from_product([[ticker], corr.index], names=names)
        pieces.append(corr)

    if not pieces:
        return pd.Series(dtype=float, index=left.index)

    return pd.concat(pieces).reindex(left.index)


def cs_rank(series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank by date."""
    return series.groupby(level=1, group_keys=False).rank(pct=True)


def signed_power(series: pd.Series, exponent: float) -> pd.Series:
    """Power transform that keeps the original sign."""
    return np.sign(series) * np.abs(series).pow(exponent)


def decay_linear(series: pd.Series, window: int) -> pd.Series:
    """Linearly weighted rolling mean, oldest observation has the smallest weight."""
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()

    return (
        series.groupby(level=0, group_keys=False)
        .rolling(window)
        .apply(lambda values: float(np.dot(values, weights)), raw=True)
        .droplevel(0)
    )
