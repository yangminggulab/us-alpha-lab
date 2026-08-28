from __future__ import annotations

import pandas as pd

from us_alpha_lab.visualization import create_factor_charts


def test_create_factor_charts(tmp_path) -> None:
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
                    "dollar_volume": ticker_index + date_index * 0.1,
                    "volatility_21d": ticker_index * -1 + date_index * 0.05,
                }
            )

    paths = create_factor_charts(pd.DataFrame(rows), output_dir=tmp_path, horizon=2)

    assert len(paths) == 4
    assert all(path.exists() for path in paths)
