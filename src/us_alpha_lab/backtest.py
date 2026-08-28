from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_alpha_lab.benchmark import benchmark_diagnostics, equal_weight_benchmark
from us_alpha_lab.risk import return_metrics


@dataclass(frozen=True)
class BacktestConfig:
    factor: str
    horizon: int = 1
    quantiles: int = 5
    cost_bps: float = 5.0
    long_short: bool = True
    include_benchmark: bool = True


@dataclass(frozen=True)
class BacktestResult:
    daily_returns: pd.DataFrame
    metrics: dict[str, float]
    weights: pd.DataFrame

def _membership_turnover(current: set[str], previous: set[str]) -> float:
    if not previous:
        return 1.0 if current else 0.0
    return len(current.symmetric_difference(previous)) / max(len(current | previous), 1)

def _future_returns(frame: pd.DataFrame, horizon: int) -> pd.Series:
    data = frame.sort_values(["ticker", "date"])
    return data.groupby("ticker")["close"].shift(-horizon) / data["close"] - 1


def build_quantile_weights(
    factors: pd.DataFrame,
    factor: str,
    horizon: int = 1,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Build top/bottom quantile weights for each rebalance date."""
    if factor not in factors.columns:
        raise ValueError(f"Unknown factor: {factor}")

    data = factors.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["future_return"] = _future_returns(data, horizon=horizon)
    rows = []

    for date, group in data.groupby("date"):
        clean = group[["ticker", factor, "future_return"]].dropna()
        if clean[factor].nunique() < 2 or len(clean) < quantiles:
            continue

        clean = clean.copy()
        clean["quantile"] = pd.qcut(
            clean[factor].rank(method="first"),
            q=quantiles,
            labels=False,
            duplicates="drop",
        )
        clean["quantile"] = clean["quantile"].astype(int) + 1

        top = clean[clean["quantile"] == quantiles]
        bottom = clean[clean["quantile"] == 1]
        if top.empty or bottom.empty:
            continue

        for _, row in top.iterrows():
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "ticker": row["ticker"],
                    "side": "long",
                    "quantile": quantiles,
                    "weight": 1 / len(top),
                    "future_return": row["future_return"],
                }
            )
        for _, row in bottom.iterrows():
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "ticker": row["ticker"],
                    "side": "short",
                    "quantile": 1,
                    "weight": -1 / len(bottom),
                    "future_return": row["future_return"],
                }
            )

    if not rows:
        raise RuntimeError("Backtest has no valid periods. Try fewer quantiles or a larger universe.")

    return pd.DataFrame(rows).sort_values(["date", "side", "ticker"]).reset_index(drop=True)


def run_quantile_backtest(
    factors: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Run a quantile portfolio backtest with long, short, spread, and benchmark returns."""
    weights = build_quantile_weights(
        factors,
        factor=config.factor,
        horizon=config.horizon,
        quantiles=config.quantiles,
    )
    rows = []
    previous_top: set[str] = set()
    previous_bottom: set[str] = set()

    data = factors.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["future_return"] = _future_returns(data, horizon=config.horizon)
    benchmark = equal_weight_benchmark(data, horizon=config.horizon)

    for date, group in weights.groupby("date"):
        longs = group[group["side"] == "long"]
        shorts = group[group["side"] == "short"]
        top_names = set(longs["ticker"])
        bottom_names = set(shorts["ticker"])
        turnover = (
            _membership_turnover(top_names, previous_top)
            + _membership_turnover(bottom_names, previous_bottom)
        ) / 2
        cost = turnover * config.cost_bps / 10_000

        long_return = float((longs["weight"].abs() * longs["future_return"]).sum())
        short_leg_return = float((shorts["weight"].abs() * shorts["future_return"]).sum())
        spread_return = long_return - short_leg_return - cost
        benchmark_return = float(benchmark.get(date, np.nan))

        rows.append(
            {
                "date": pd.Timestamp(date),
                "long_return": long_return,
                "short_leg_return": short_leg_return,
                "spread_return": spread_return,
                "benchmark_return": benchmark_return,
                "long_excess_return": long_return - benchmark_return,
                "turnover": turnover,
                "cost": cost,
                "top_count": len(longs),
                "bottom_count": len(shorts),
            }
        )
        previous_top = top_names
        previous_bottom = bottom_names

    returns = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if returns.empty:
        raise RuntimeError("Backtest has no valid periods. Try fewer quantiles or a larger universe.")

    returns["cumulative_long_return"] = (1 + returns["long_return"]).cumprod() - 1
    returns["cumulative_spread_return"] = (1 + returns["spread_return"]).cumprod() - 1
    returns["cumulative_benchmark_return"] = (1 + returns["benchmark_return"]).cumprod() - 1
    returns["cumulative_long_excess_return"] = (1 + returns["long_excess_return"]).cumprod() - 1

    spread_metrics = return_metrics(returns["spread_return"], config.horizon)
    long_metrics = return_metrics(returns["long_return"], config.horizon, prefix="long")
    benchmark_metrics = return_metrics(returns["benchmark_return"], config.horizon, prefix="benchmark")
    long_excess_metrics = return_metrics(
        returns["long_excess_return"],
        config.horizon,
        prefix="long_excess",
    )
    metrics = {
        **spread_metrics,
        "long_total_return": long_metrics["long_total_return"],
        "benchmark_total_return": benchmark_metrics["benchmark_total_return"],
        "long_excess_total_return": long_excess_metrics["long_excess_total_return"],
        **benchmark_diagnostics(
            returns["long_return"],
            returns["benchmark_return"],
            horizon=config.horizon,
        ),
        "average_turnover": float(returns["turnover"].mean()),
        "average_cost": float(returns["cost"].mean()),
    }
    return BacktestResult(daily_returns=returns, metrics=metrics, weights=weights)


def quantile_backtest(
    factors: pd.DataFrame,
    factor: str,
    horizon: int = 1,
    quantiles: int = 5,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Daily quantile long-short backtest using forward close-to-close returns."""
    return run_quantile_backtest(
        factors,
        BacktestConfig(
            factor=factor,
            horizon=horizon,
            quantiles=quantiles,
            cost_bps=cost_bps,
        ),
    )
