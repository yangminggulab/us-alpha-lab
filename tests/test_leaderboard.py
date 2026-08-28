from __future__ import annotations

import pandas as pd

from us_alpha_lab.leaderboard import build_factor_leaderboard


def test_build_factor_leaderboard() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100 + date_index + ticker_index,
                    "dollar_volume": ticker_index + 1,
                    "volatility_21d": 5 - ticker_index,
                }
            )
    ic_report = pd.DataFrame(
        [
            {"factor": "dollar_volume", "mean_ic": 0.1, "ic_ir": 0.2, "positive_ic_rate": 0.6},
            {"factor": "volatility_21d", "mean_ic": -0.1, "ic_ir": -0.2, "positive_ic_rate": 0.4},
        ]
    )

    leaderboard, results = build_factor_leaderboard(pd.DataFrame(rows), ic_report=ic_report)

    assert not leaderboard.empty
    assert "score" in leaderboard.columns
    assert set(results) == {"dollar_volume", "volatility_21d"}
