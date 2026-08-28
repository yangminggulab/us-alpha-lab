from __future__ import annotations

import pandas as pd

from us_alpha_lab.backtest import (
    BacktestConfig,
    build_quantile_weights,
    quantile_backtest,
    run_quantile_backtest,
)


def test_quantile_backtest_returns_metrics() -> None:
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
                    "test_factor": ticker_index,
                }
            )

    result = quantile_backtest(pd.DataFrame(rows), factor="test_factor", horizon=1)

    assert not result.daily_returns.empty
    assert "annualized_return" in result.metrics
    assert "cumulative_spread_return" in result.daily_returns.columns


def test_run_quantile_backtest_includes_benchmark_and_weights() -> None:
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
                    "test_factor": ticker_index,
                }
            )

    frame = pd.DataFrame(rows)
    weights = build_quantile_weights(frame, factor="test_factor")
    result = run_quantile_backtest(frame, BacktestConfig(factor="test_factor"))

    assert not weights.empty
    assert {"long_return", "spread_return", "benchmark_return", "long_excess_return"}.issubset(
        result.daily_returns.columns
    )
    assert "benchmark_total_return" in result.metrics
    assert not result.weights.empty
