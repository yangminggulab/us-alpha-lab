from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_alpha_lab.alpha_registry import enabled_alpha_names
from us_alpha_lab.analysis import daily_factor_ic
from us_alpha_lab.backtest import BacktestConfig, run_quantile_backtest
from us_alpha_lab.labels import add_forward_return_label
from us_alpha_lab.verdict import (
    MAX_DRAWDOWN,
    MONOTONIC_SPEARMAN,
    NET_SHARPE,
    SIGNAL_ICIR,
    SIGNAL_T,
    STABILITY_AGREEMENT,
    newey_west_t_stat,
    orthogonalized_ic_series,
    quantile_monotonicity,
    yearly_ic_sign_agreement,
)

BASE_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume", "vwap"}
DERIVED_SUFFIXES = ("_rank", "_zscore")


@dataclass(frozen=True)
class ExperimentValidationConfig:
    horizon: int = 5
    backtest_horizon: int = 1
    quantiles: int = 5
    cost_bps: float = 5.0
    min_coverage: float = 0.2
    factor_prefixes: tuple[str, ...] = ("alpha_", "ml_prediction")


def infer_experiment_factor_columns(
    experiment: pd.DataFrame,
    prefixes: Iterable[str] = ("alpha_", "ml_prediction"),
) -> list[str]:
    """Infer candidate factor columns from an arbitrary experiment output frame."""
    prefixes_tuple = tuple(prefixes)
    columns = []
    for column in experiment.columns:
        if column in BASE_COLUMNS or column.endswith(DERIVED_SUFFIXES):
            continue
        if not column.startswith(prefixes_tuple):
            continue
        if pd.api.types.is_numeric_dtype(experiment[column]):
            columns.append(column)
    return columns


def default_public_factor_pool(base_factors: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Use enabled non-ML alpha factors from the main factor table as the public pool."""
    excluded = set(exclude)
    return [
        factor
        for factor in enabled_alpha_names()
        if factor in base_factors.columns
        and factor not in excluded
        and not factor.startswith("ml_prediction")
    ]


def merge_experiment_with_base(
    experiment: pd.DataFrame,
    base_factors: pd.DataFrame,
    factor_columns: list[str],
    pool_columns: list[str],
) -> pd.DataFrame:
    """Build a validation frame with experiment factors, close, and public pool columns."""
    required = {"ticker", "date", *factor_columns}
    missing = sorted(required - set(experiment.columns))
    if missing:
        raise ValueError(f"Experiment frame is missing required columns: {missing}")

    experiment_subset = experiment[["ticker", "date", *factor_columns]].copy()
    experiment_subset["date"] = pd.to_datetime(experiment_subset["date"])

    base_required = {"ticker", "date", "close", *pool_columns}
    base_missing = sorted(base_required - set(base_factors.columns))
    if base_missing:
        raise ValueError(f"Base factor frame is missing required columns: {base_missing}")

    base_subset = base_factors[["ticker", "date", "close", *pool_columns]].copy()
    base_subset["date"] = pd.to_datetime(base_subset["date"])
    return experiment_subset.merge(base_subset, on=["ticker", "date"], how="left")


def _factor_profile(frame: pd.DataFrame, factor: str) -> dict[str, float]:
    values = pd.to_numeric(frame[factor], errors="coerce").replace([np.inf, -np.inf], np.nan)
    daily_unique = frame.assign(_factor_value=values).groupby("date")["_factor_value"].nunique()
    return {
        "coverage": float(values.notna().mean()),
        "active_days": float((daily_unique >= 5).sum()),
        "median_daily_unique": float(daily_unique.median()) if not daily_unique.empty else 0.0,
    }


def _ic_stats(daily_ic: pd.Series) -> dict[str, float | int]:
    if daily_ic.empty:
        return {
            "mean_ic": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "positive_ic_rate": np.nan,
            "days": 0,
        }
    ic_std = float(daily_ic.std())
    return {
        "mean_ic": float(daily_ic.mean()),
        "ic_std": ic_std,
        "ic_ir": float(daily_ic.mean() / ic_std) if ic_std else 0.0,
        "positive_ic_rate": float((daily_ic > 0).mean()),
        "days": len(daily_ic),
    }


def _directional_rate(mean_ic: float, positive_ic_rate: float) -> float:
    if pd.isna(mean_ic) or pd.isna(positive_ic_rate):
        return np.nan
    return float(positive_ic_rate if mean_ic >= 0 else 1.0 - positive_ic_rate)


def _gate_verdict(passed: int, evaluated: int) -> str:
    if evaluated < 3:
        return "数据不足"
    if passed == evaluated == 5:
        return "可盈利候选"
    if passed >= 4:
        return "接近达标"
    if passed >= 3:
        return "部分达标"
    return "未达标"


def build_experiment_validation_report(
    experiment: pd.DataFrame,
    base_factors: pd.DataFrame,
    factor_columns: list[str] | None = None,
    orthogonal_pool: list[str] | None = None,
    config: ExperimentValidationConfig | None = None,
) -> pd.DataFrame:
    """Validate arbitrary experiment factors through the shared IC/backtest/verdict gates."""
    cfg = config or ExperimentValidationConfig()
    factor_columns = factor_columns or infer_experiment_factor_columns(
        experiment,
        prefixes=cfg.factor_prefixes,
    )
    if not factor_columns:
        return pd.DataFrame()

    orthogonal_pool = (
        orthogonal_pool
        if orthogonal_pool is not None
        else default_public_factor_pool(base_factors, exclude=factor_columns)
    )
    frame = merge_experiment_with_base(experiment, base_factors, factor_columns, orthogonal_pool)
    data = add_forward_return_label(frame, horizon=cfg.horizon)
    label = f"future_return_{cfg.horizon}d"

    rows = []
    for factor in factor_columns:
        profile = _factor_profile(data, factor)
        if profile["coverage"] < cfg.min_coverage:
            continue

        daily_ic = daily_factor_ic(data, factor=factor, label=label).dropna()
        ic = _ic_stats(daily_ic)
        if int(ic["days"]) == 0:
            continue

        orth_series = (
            orthogonalized_ic_series(data, factor, orthogonal_pool, label).dropna()
            if orthogonal_pool
            else pd.Series(dtype=float)
        )
        orth = _ic_stats(orth_series)
        monotonicity = quantile_monotonicity(data, factor, horizon=cfg.horizon, quantiles=cfg.quantiles)
        agreement, n_years = yearly_ic_sign_agreement(daily_ic)
        ic_t = newey_west_t_stat(daily_ic, lag=cfg.horizon)

        net_sharpe: float | None = None
        max_drawdown: float | None = None
        average_turnover: float | None = None
        try:
            backtest = run_quantile_backtest(
                data,
                BacktestConfig(
                    factor=factor,
                    horizon=cfg.backtest_horizon,
                    quantiles=cfg.quantiles,
                    cost_bps=cfg.cost_bps,
                ),
            )
            net_sharpe = float(backtest.metrics["information_ratio"])
            max_drawdown = float(backtest.metrics["max_drawdown"])
            average_turnover = float(backtest.metrics["average_turnover"])
        except RuntimeError:
            pass

        signal_ok = bool(float(ic["ic_ir"]) >= SIGNAL_ICIR and abs(ic_t) >= SIGNAL_T)
        monotonic_ok = bool(abs(monotonicity) >= MONOTONIC_SPEARMAN)
        net_return_ok: bool | None = (
            net_sharpe >= NET_SHARPE and max_drawdown > MAX_DRAWDOWN
            if net_sharpe is not None and max_drawdown is not None
            else None
        )
        stability_ok: bool | None = (
            agreement >= STABILITY_AGREEMENT
            if n_years >= 2
            else None
        )
        incremental_ok: bool | None = (
            abs(float(orth["mean_ic"])) >= 0.01 and abs(float(orth["ic_ir"])) >= 0.3
            if int(orth["days"]) > 0
            and pd.notna(orth["mean_ic"])
            and pd.notna(orth["ic_ir"])
            else None
        )
        gates = (signal_ok, monotonic_ok, net_return_ok, stability_ok, incremental_ok)
        passed = sum(1 for gate in gates if gate is True)
        evaluated = sum(1 for gate in gates if gate is not None)

        row = {
            "factor": factor,
            **profile,
            **ic,
            "ic_t": ic_t,
            "monotonicity": monotonicity,
            "yearly_agreement": agreement,
            "n_years": n_years,
            "ic_orth": orth["mean_ic"],
            "ic_orth_ir": orth["ic_ir"],
            "orth_positive_ic_rate": orth["positive_ic_rate"],
            "orth_days": orth["days"],
            "net_sharpe": net_sharpe,
            "max_drawdown": max_drawdown,
            "average_turnover": average_turnover,
            "directional_ic_rate": _directional_rate(
                float(ic["mean_ic"]),
                float(ic["positive_ic_rate"]),
            ),
            "signal_ok": signal_ok,
            "monotonic_ok": monotonic_ok,
            "net_return_ok": net_return_ok,
            "stability_ok": stability_ok,
            "incremental_ok": incremental_ok,
            "passed": passed,
            "evaluated": evaluated,
            "verdict": _gate_verdict(passed, evaluated),
        }
        row["validation_score"] = (
            abs(float(row["ic_orth"] if pd.notna(row["ic_orth"]) else 0.0))
            + abs(float(row["ic_orth_ir"] if pd.notna(row["ic_orth_ir"]) else 0.0)) * 0.05
            + passed * 0.01
            - float(row["average_turnover"] if row["average_turnover"] is not None else 0.0) * 0.005
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["validation_score", "passed"], ascending=[False, False])
        .reset_index(drop=True)
    )
