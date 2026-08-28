from __future__ import annotations

import pandas as pd

from us_alpha_lab.analysis import factor_ic_report


def test_factor_ic_report_returns_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(["AAA", "BBB", "CCC"]):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100 + date_index + ticker_index,
                    "reversal_1d": ticker_index,
                    "momentum_5d": date_index + ticker_index,
                }
            )

    report = factor_ic_report(pd.DataFrame(rows), horizon=2)

    assert set(report["factor"]) == {"reversal_1d", "momentum_5d"}
    assert report["days"].min() > 0
