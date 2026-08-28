from __future__ import annotations

import numpy as np
import pandas as pd


def sanitize_factor_values(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace non-finite factor values with missing values."""
    data = frame.copy()
    existing = [column for column in columns if column in data.columns]
    if existing:
        data[existing] = data[existing].replace([np.inf, -np.inf], np.nan)
    return data


def cross_sectional_zscore(series: pd.Series, dates: pd.Series) -> pd.Series:
    """Same-day z-score with zero-variance groups left missing."""
    grouped = series.groupby(dates)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (series - mean) / std


def winsorize_by_date(
    frame: pd.DataFrame,
    columns: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Clip factor tails within each date to reduce outlier impact."""
    data = frame.copy()
    for column in columns:
        if column not in data.columns:
            continue
        lower_bound = data.groupby("date")[column].transform(lambda values: values.quantile(lower))
        upper_bound = data.groupby("date")[column].transform(lambda values: values.quantile(upper))
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data


def add_cross_sectional_features(
    frame: pd.DataFrame,
    columns: list[str],
    include_ranks: bool = True,
    include_zscores: bool = True,
) -> pd.DataFrame:
    """Add cross-sectional rank and z-score feature variants for factor columns."""
    data = sanitize_factor_values(frame, columns)
    for column in columns:
        if column not in data.columns:
            continue
        if include_ranks:
            data[f"{column}_rank"] = data.groupby("date")[column].rank(pct=True)
        if include_zscores:
            data[f"{column}_zscore"] = cross_sectional_zscore(data[column], data["date"])
    return data
