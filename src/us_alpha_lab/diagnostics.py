from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_alpha_lab.alpha_registry import get_alpha_spec
from us_alpha_lab.analysis import daily_factor_ic
from us_alpha_lab.backtest import BacktestResult, quantile_backtest
from us_alpha_lab.labels import add_forward_return_label


@dataclass(frozen=True)
class AlphaDiagnostic:
    factor: str
    status: str
    error_codes: list[str]
    metrics: dict[str, float]
    backtest: BacktestResult | None = None


def _status_from_errors(error_codes: list[str]) -> str:
    fail_codes = {"MISSING_FACTOR", "LOW_COVERAGE", "LOW_ACTIVE_DAYS", "CONSTANT_CROSS_SECTION"}
    if any(code in fail_codes for code in error_codes):
        return "fail"
    if error_codes:
        return "review"
    return "pass"


def _uses_active_window(factor: str) -> bool:
    try:
        return get_alpha_spec(factor).category == "machine_learning"
    except KeyError:
        return False


def diagnose_factor(
    factors: pd.DataFrame,
    factor: str,
    ic_horizon: int = 5,
    backtest_horizon: int = 1,
    quantiles: int = 5,
    cost_bps: float = 5.0,
    min_coverage: float = 0.6,
    min_active_days: int = 100,
) -> AlphaDiagnostic:
    """Run data, signal, and portfolio diagnostics for one factor."""
    if factor not in factors.columns:
        return AlphaDiagnostic(
            factor=factor,
            status="fail",
            error_codes=["MISSING_FACTOR"],
            metrics={},
        )

    values = pd.to_numeric(factors[factor], errors="coerce")
    finite = values.replace([np.inf, -np.inf], np.nan)
    raw_coverage = float(finite.notna().mean())
    daily_unique_all = factors.assign(_factor_value=finite).groupby("date")["_factor_value"].nunique()

    if _uses_active_window(factor):
        active_index = daily_unique_all[daily_unique_all > 0].index
        scoped = factors["date"].isin(active_index)
        coverage = float(finite[scoped].notna().mean()) if scoped.any() else 0.0
        daily_unique = daily_unique_all.loc[active_index]
    else:
        coverage = raw_coverage
        daily_unique = daily_unique_all

    active_days = int((daily_unique >= quantiles).sum())
    median_unique = float(daily_unique.median()) if not daily_unique.empty else 0.0
    error_codes: list[str] = []

    if coverage < min_coverage:
        error_codes.append("LOW_COVERAGE")
    if active_days < min_active_days:
        error_codes.append("LOW_ACTIVE_DAYS")
    if median_unique < quantiles:
        error_codes.append("CONSTANT_CROSS_SECTION")

    labeled = add_forward_return_label(factors, horizon=ic_horizon)
    label = f"future_return_{ic_horizon}d"
    daily_ic = daily_factor_ic(labeled, factor=factor, label=label).dropna()
    mean_ic = float(daily_ic.mean()) if not daily_ic.empty else 0.0
    ic_std = float(daily_ic.std()) if not daily_ic.empty else 0.0
    ic_ir = mean_ic / ic_std if ic_std else 0.0
    positive_ic_rate = float((daily_ic > 0).mean()) if not daily_ic.empty else 0.0

    if abs(mean_ic) < 0.01 and abs(ic_ir) < 0.05:
        error_codes.append("WEAK_SIGNAL")

    backtest_result: BacktestResult | None = None
    portfolio_metrics: dict[str, float] = {}
    try:
        backtest_result = quantile_backtest(
            factors,
            factor=factor,
            horizon=backtest_horizon,
            quantiles=quantiles,
            cost_bps=cost_bps,
        )
        portfolio_metrics = backtest_result.metrics
    except RuntimeError:
        error_codes.append("BACKTEST_NO_VALID_PERIODS")

    information_ratio = float(portfolio_metrics.get("information_ratio", 0.0))
    total_return = float(portfolio_metrics.get("total_return", 0.0))
    max_drawdown = float(portfolio_metrics.get("max_drawdown", 0.0))
    average_turnover = float(portfolio_metrics.get("average_turnover", 0.0))
    benchmark_beta = float(portfolio_metrics.get("benchmark_beta", 0.0))
    benchmark_correlation = float(portfolio_metrics.get("benchmark_correlation", 0.0))
    excess_total_return = float(portfolio_metrics.get("excess_total_return", 0.0))

    if backtest_result is not None:
        if information_ratio < 0 or total_return < 0:
            error_codes.append("NEGATIVE_PORTFOLIO")
        if excess_total_return < 0:
            error_codes.append("POOR_EXCESS_RETURN")
        if benchmark_beta > 1.5:
            error_codes.append("HIGH_BENCHMARK_BETA")
        if benchmark_correlation > 0.9:
            error_codes.append("HIGH_BENCHMARK_CORRELATION")
        if max_drawdown < -0.35:
            error_codes.append("HIGH_DRAWDOWN")
        if average_turnover > 0.8:
            error_codes.append("HIGH_TURNOVER")
        if (mean_ic > 0.01 and information_ratio < 0) or (mean_ic < -0.01 and information_ratio > 0):
            error_codes.append("IC_BACKTEST_MISMATCH")

    metrics = {
        "coverage": coverage,
        "raw_coverage": raw_coverage,
        "diagnostic_window_days": float(len(daily_unique)),
        "active_days": float(active_days),
        "median_daily_unique": median_unique,
        "mean_ic": mean_ic,
        "ic_ir": float(ic_ir),
        "positive_ic_rate": positive_ic_rate,
        **portfolio_metrics,
    }
    return AlphaDiagnostic(
        factor=factor,
        status=_status_from_errors(error_codes),
        error_codes=error_codes,
        metrics=metrics,
        backtest=backtest_result,
    )
