from __future__ import annotations

import numpy as np
import pandas as pd

from us_alpha_lab.alpha_registry import enabled_alpha_names
from us_alpha_lab.feature_engineering import add_cross_sectional_features, sanitize_factor_values
from us_alpha_lab.formula_factors import add_formula_alpha_factors

BASE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume", "vwap"]
FACTOR_COLUMNS = enabled_alpha_names()


def add_alpha_factors(bars: pd.DataFrame) -> pd.DataFrame:
    """Build a compact set of price-volume alpha factors."""
    missing = sorted(set(BASE_COLUMNS) - set(bars.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = bars.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = data.groupby("ticker", group_keys=False)
    data["ret_1d"] = grouped["close"].pct_change()
    data["reversal_1d"] = -data["ret_1d"]
    data["momentum_5d"] = grouped["close"].pct_change(5)
    data["momentum_21d"] = grouped["close"].pct_change(21)
    data["momentum_63d"] = grouped["close"].pct_change(63)
    data["volatility_21d"] = grouped["ret_1d"].rolling(21).std().reset_index(level=0, drop=True)

    volume_mean = grouped["volume"].rolling(21).mean().reset_index(level=0, drop=True)
    volume_std = grouped["volume"].rolling(21).std().reset_index(level=0, drop=True)
    data["volume_z_21d"] = (data["volume"] - volume_mean) / volume_std.replace(0, np.nan)

    data["dollar_volume"] = data["close"] * data["volume"]
    day_range = (data["high"] - data["low"]).replace(0, np.nan)
    data["close_to_high"] = (data["high"] - data["close"]) / day_range
    data["close_to_low"] = (data["close"] - data["low"]) / day_range
    data["vwap_gap"] = data["close"] / data["vwap"] - 1

    data = add_formula_alpha_factors(data.set_index(["ticker", "date"], drop=False)).reset_index(drop=True)
    data = sanitize_factor_values(data, FACTOR_COLUMNS)

    return data


def add_cross_sectional_ranks(
    frame: pd.DataFrame,
    columns: list[str] = FACTOR_COLUMNS,
    include_zscores: bool = True,
) -> pd.DataFrame:
    """Add same-day percentile ranks and z-scores for each factor."""
    return add_cross_sectional_features(
        frame,
        columns=columns,
        include_ranks=True,
        include_zscores=include_zscores,
    )
