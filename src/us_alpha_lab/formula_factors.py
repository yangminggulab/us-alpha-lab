from __future__ import annotations

import numpy as np
import pandas as pd

from us_alpha_lab.operators import (
    cs_rank,
    decay_linear,
    delay,
    delta,
    rolling_corr,
    rolling_max,
    rolling_mean,
    ts_rank,
)


def add_formula_alpha_factors(indexed: pd.DataFrame) -> pd.DataFrame:
    """Add formula-style alpha candidates to a ticker/date indexed frame."""
    data = indexed.copy()

    open_rank = cs_rank(data["open"])
    close_rank = cs_rank(data["close"])
    volume_rank = cs_rank(data["volume"])
    daily_range = (data["high"] - data["low"]).replace(0, np.nan)
    intraday_return = (data["close"] - data["open"]) / data["open"].replace(0, np.nan)
    overnight_gap = data["open"] / delay(data["close"], 1).replace(0, np.nan) - 1
    amihud = data["ret_1d"].abs() / data["dollar_volume"].replace(0, np.nan)

    data["alpha_open_volume_corr_10"] = -rolling_corr(open_rank, volume_rank, 10)
    data["alpha_vwap_reversion_10"] = -cs_rank(data["open"] - decay_linear(data["vwap"], 10))
    data["alpha_range_ts_rank_9"] = -ts_rank(delta(data["high"] - data["low"], 1), 9)
    data["alpha_overnight_gap_reversal_5"] = -ts_rank(overnight_gap, 5)
    data["alpha_intraday_strength_5"] = ts_rank(intraday_return, 5)
    data["alpha_volume_price_divergence_21"] = -rolling_corr(close_rank, volume_rank, 21)
    data["alpha_low_volatility_21d"] = -data["volatility_21d"]
    data["alpha_price_to_21d_high"] = data["close"] / rolling_max(data["high"], 21).replace(0, np.nan) - 1
    data["alpha_ma_gap_21d"] = data["close"] / rolling_mean(data["close"], 21).replace(0, np.nan) - 1
    data["alpha_liquidity_quality_21d"] = -rolling_mean(amihud, 21)
    data["alpha_close_position_5"] = ts_rank((data["close"] - data["low"]) / daily_range, 5)

    return data
