from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_alpha_lab.alpha_registry import enabled_alpha_names
from us_alpha_lab.backtest import BacktestResult, quantile_backtest


def build_factor_leaderboard(
    factors: pd.DataFrame,
    ic_report: pd.DataFrame,
    horizon: int = 1,
    quantiles: int = 5,
    cost_bps: float = 5.0,
    factor_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Batch backtest factors and merge signal and portfolio metrics."""
    factor_columns = factor_columns or [factor for factor in enabled_alpha_names() if factor in factors.columns]
    ic_by_factor = ic_report.set_index("factor")
    rows = []
    results: dict[str, BacktestResult] = {}

    for factor in factor_columns:
        try:
            result = quantile_backtest(
                factors,
                factor=factor,
                horizon=horizon,
                quantiles=quantiles,
                cost_bps=cost_bps,
            )
        except RuntimeError:
            continue

        results[factor] = result
        ic_metrics = ic_by_factor.loc[factor].to_dict() if factor in ic_by_factor.index else {}
        rows.append(
            {
                "factor": factor,
                "mean_ic": ic_metrics.get("mean_ic"),
                "ic_ir": ic_metrics.get("ic_ir"),
                "positive_ic_rate": ic_metrics.get("positive_ic_rate"),
                **result.metrics,
            }
        )

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard, results

    leaderboard["score"] = (
        leaderboard["information_ratio"].fillna(0)
        + leaderboard["ic_ir"].fillna(0)
        + (leaderboard["hit_rate"].fillna(0) - 0.5)
        - leaderboard["average_turnover"].fillna(0) * 0.1
    )
    leaderboard = leaderboard.sort_values("score", ascending=False).reset_index(drop=True)
    return leaderboard, results


def save_backtest_results(results: dict[str, BacktestResult], output_dir: str | Path) -> list[Path]:
    paths = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for factor, result in results.items():
        path = output_path / f"{factor}_backtest.parquet"
        result.daily_returns.to_parquet(path, index=False)
        paths.append(path)

    return paths
