from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(cumulative_return: pd.Series) -> float:
    equity = 1 + cumulative_return
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def return_metrics(returns: pd.Series, horizon: int = 1, prefix: str = "") -> dict[str, float]:
    clean = returns.dropna()
    key = f"{prefix}_" if prefix else ""
    if clean.empty:
        return {
            f"{key}periods": 0.0,
            f"{key}total_return": 0.0,
            f"{key}annualized_return": 0.0,
            f"{key}annualized_volatility": 0.0,
            f"{key}information_ratio": 0.0,
            f"{key}max_drawdown": 0.0,
            f"{key}hit_rate": 0.0,
            f"{key}downside_volatility": 0.0,
            f"{key}calmar_ratio": 0.0,
        }

    periods_per_year = 252 / horizon
    cumulative = (1 + clean).cumprod() - 1
    annualized_return = float((1 + cumulative.iloc[-1]) ** (periods_per_year / len(clean)) - 1)
    annualized_volatility = float(clean.std() * np.sqrt(periods_per_year))
    information_ratio = annualized_return / annualized_volatility if annualized_volatility else 0.0
    drawdown = max_drawdown(cumulative)
    downside = clean[clean < 0]
    downside_volatility = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    calmar_ratio = annualized_return / abs(drawdown) if drawdown < 0 else 0.0
    return {
        f"{key}periods": float(len(clean)),
        f"{key}total_return": float(cumulative.iloc[-1]),
        f"{key}annualized_return": annualized_return,
        f"{key}annualized_volatility": annualized_volatility,
        f"{key}information_ratio": float(information_ratio),
        f"{key}max_drawdown": drawdown,
        f"{key}hit_rate": float((clean > 0).mean()),
        f"{key}downside_volatility": downside_volatility,
        f"{key}calmar_ratio": float(calmar_ratio),
    }


def beta_to_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    strategy = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]
    variance = benchmark.var()
    if variance == 0:
        return 0.0
    return float(strategy.cov(benchmark) / variance)


def correlation_to_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
