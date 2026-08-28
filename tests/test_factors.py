from __future__ import annotations

import pandas as pd

from us_alpha_lab.factors import add_alpha_factors, add_cross_sectional_ranks
from us_alpha_lab.labels import add_forward_return_label


def test_factor_and_label_pipeline() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    rows = []
    for ticker, offset in [("AAA", 0), ("BBB", 10)]:
        for index, date in enumerate(dates):
            close = 100 + offset + index
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000 + index * 1000,
                    "vwap": close - 0.1,
                }
            )

    bars = pd.DataFrame(rows)
    factors = add_cross_sectional_ranks(add_alpha_factors(bars))
    labeled = add_forward_return_label(factors, horizon=5)

    assert "momentum_21d" in labeled.columns
    assert "momentum_21d_rank" in labeled.columns
    assert "momentum_21d_zscore" in labeled.columns
    assert "alpha_ma_gap_21d" in labeled.columns
    assert "alpha_liquidity_quality_21d_rank" in labeled.columns
    assert "future_return_5d" in labeled.columns
    assert labeled["future_return_5d"].notna().sum() > 0
