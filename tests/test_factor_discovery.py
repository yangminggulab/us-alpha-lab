from __future__ import annotations

import pandas as pd

from us_alpha_lab.factor_discovery import build_factor_discovery_report
from us_alpha_lab.factors import add_alpha_factors, add_cross_sectional_ranks


def test_build_factor_discovery_report_returns_candidates() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    rows = []
    for ticker_index, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        for date_index, date in enumerate(dates):
            close = 100 + ticker_index * 5 + date_index
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000 + ticker_index * 10_000 + date_index * 1000,
                    "vwap": close - 0.1,
                }
            )

    factors = add_cross_sectional_ranks(add_alpha_factors(pd.DataFrame(rows)))
    report = build_factor_discovery_report(factors, horizon=5)

    assert "alpha_ma_gap_21d" in set(report["factor"])
    assert "discovery_score" in report.columns
    assert report["factor"].is_unique
