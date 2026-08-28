from __future__ import annotations

import pandas as pd

from us_alpha_lab.risk import beta_to_benchmark, correlation_to_benchmark, return_metrics


def equal_weight_benchmark(factors: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Equal-weight universe forward return by date."""
    data = factors.sort_values(["ticker", "date"]).copy()
    data["date"] = pd.to_datetime(data["date"])
    data["future_return"] = data.groupby("ticker")["close"].shift(-horizon) / data["close"] - 1
    return data.groupby("date")["future_return"].mean().dropna()


def benchmark_diagnostics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    horizon: int = 1,
) -> dict[str, float]:
    """Compare a strategy return stream against a benchmark."""
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {
            "benchmark_beta": 0.0,
            "benchmark_correlation": 0.0,
            "excess_total_return": 0.0,
            "excess_information_ratio": 0.0,
            "excess_max_drawdown": 0.0,
        }

    excess = aligned["strategy"] - aligned["benchmark"]
    excess_metrics = return_metrics(excess, horizon=horizon, prefix="excess")
    return {
        "benchmark_beta": beta_to_benchmark(aligned["strategy"], aligned["benchmark"]),
        "benchmark_correlation": correlation_to_benchmark(aligned["strategy"], aligned["benchmark"]),
        "excess_total_return": excess_metrics["excess_total_return"],
        "excess_information_ratio": excess_metrics["excess_information_ratio"],
        "excess_max_drawdown": excess_metrics["excess_max_drawdown"],
    }
