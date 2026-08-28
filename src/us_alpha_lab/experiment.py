from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from us_alpha_lab.alpha_cluster import diagnostics_to_frame, run_alpha_cluster, save_cluster_outputs
from us_alpha_lab.alpha_registry import alpha_registry_frame
from us_alpha_lab.analysis import factor_ic_report
from us_alpha_lab.backtest import BacktestConfig, run_quantile_backtest
from us_alpha_lab.config import ResearchConfig
from us_alpha_lab.leaderboard import build_factor_leaderboard, save_backtest_results
from us_alpha_lab.visualization import (
    create_factor_charts,
    plot_backtest_drawdown,
    plot_backtest_equity,
    plot_leaderboard,
    plot_metric_scatter,
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "free_alpha"
    ic_horizon: int = 5
    backtest_horizon: int = 1
    quantiles: int = 5
    cost_bps: float = 5.0
    workers: int = 1
    best_factor: str | None = None


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: Path
    leaderboard_path: Path
    cluster_report_path: Path
    metrics_path: Path
    best_factor: str


def make_run_dir(name: str, root: str | Path = "runs") -> Path:
    timestamp = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    run_dir = Path(root) / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _snapshot_config(config_path: Path, run_dir: Path) -> Path:
    target = run_dir / "config.yaml"
    shutil.copy2(config_path, target)
    return target


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def run_experiment(
    factors: pd.DataFrame,
    research_config: ResearchConfig,
    research_config_path: str | Path,
    experiment: ExperimentConfig,
    run_root: str | Path = "runs",
) -> ExperimentResult:
    """Create a reproducible experiment run from an existing factor table."""
    run_dir = make_run_dir(experiment.name, root=run_root)
    _snapshot_config(Path(research_config_path), run_dir)

    registry_path = run_dir / "alpha_registry.csv"
    alpha_registry_frame().to_csv(registry_path, index=False)

    ic_report = factor_ic_report(factors, horizon=experiment.ic_horizon)
    ic_report_path = run_dir / "ic_report.csv"
    ic_report.to_csv(ic_report_path, index=False)

    leaderboard_frame, leaderboard_results = build_factor_leaderboard(
        factors,
        ic_report=ic_report,
        horizon=experiment.backtest_horizon,
        quantiles=experiment.quantiles,
        cost_bps=experiment.cost_bps,
    )
    leaderboard_path = run_dir / "factor_leaderboard.csv"
    leaderboard_frame.to_csv(leaderboard_path, index=False)
    save_backtest_results(leaderboard_results, run_dir / "leaderboard_backtests")

    diagnostics = run_alpha_cluster(
        factors,
        ic_horizon=experiment.ic_horizon,
        backtest_horizon=experiment.backtest_horizon,
        quantiles=experiment.quantiles,
        cost_bps=experiment.cost_bps,
        workers=experiment.workers,
    )
    cluster_report_path, _ = save_cluster_outputs(diagnostics, run_dir / "alpha_cluster")
    cluster_frame = diagnostics_to_frame(diagnostics)

    if experiment.best_factor:
        best_factor = experiment.best_factor
    elif not leaderboard_frame.empty:
        best_factor = str(leaderboard_frame.iloc[0]["factor"])
    else:
        best_factor = str(ic_report.iloc[0]["factor"])

    backtest_result = run_quantile_backtest(
        factors,
        BacktestConfig(
            factor=best_factor,
            horizon=experiment.backtest_horizon,
            quantiles=experiment.quantiles,
            cost_bps=experiment.cost_bps,
        ),
    )
    backtest_dir = run_dir / "best_backtest"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    backtest_returns_path = backtest_dir / "returns.parquet"
    backtest_weights_path = backtest_dir / "weights.parquet"
    backtest_result.daily_returns.to_parquet(backtest_returns_path, index=False)
    backtest_result.weights.to_parquet(backtest_weights_path, index=False)

    chart_dir = run_dir / "charts"
    create_factor_charts(
        factors,
        output_dir=chart_dir,
        horizon=experiment.ic_horizon,
        factor=best_factor,
    )
    plot_leaderboard(leaderboard_frame, chart_dir)
    plot_metric_scatter(leaderboard_frame, chart_dir)
    plot_backtest_equity(backtest_result.daily_returns, best_factor, chart_dir)
    plot_backtest_drawdown(backtest_result.daily_returns, best_factor, chart_dir)

    metrics = {
        "experiment": asdict(experiment),
        "research_config": {
            **asdict(research_config),
            "raw_path": str(research_config.raw_path),
            "factors_path": str(research_config.factors_path),
            "model_path": str(research_config.model_path),
        },
        "rows": len(factors),
        "tickers": int(factors["ticker"].nunique()),
        "start": str(pd.to_datetime(factors["date"]).min().date()),
        "end": str(pd.to_datetime(factors["date"]).max().date()),
        "best_factor": best_factor,
        "best_backtest": backtest_result.metrics,
        "leaderboard_rows": len(leaderboard_frame),
        "cluster_status_counts": cluster_frame["status"].value_counts().to_dict()
        if not cluster_frame.empty
        else {},
        "artifacts": {
            "alpha_registry": str(registry_path),
            "ic_report": str(ic_report_path),
            "leaderboard": str(leaderboard_path),
            "cluster_report": str(cluster_report_path),
            "backtest_returns": str(backtest_returns_path),
            "backtest_weights": str(backtest_weights_path),
            "charts": str(chart_dir),
        },
    }
    metrics_path = run_dir / "metrics.json"
    _write_json(metrics_path, metrics)

    return ExperimentResult(
        run_dir=run_dir,
        leaderboard_path=leaderboard_path,
        cluster_report_path=cluster_report_path,
        metrics_path=metrics_path,
        best_factor=best_factor,
    )
