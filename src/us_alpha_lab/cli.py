from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from us_alpha_lab.alpha_cluster import diagnostics_to_frame, run_alpha_cluster, save_cluster_outputs
from us_alpha_lab.alpha_registry import alpha_registry_frame
from us_alpha_lab.analysis import factor_ic_report
from us_alpha_lab.backtest import BacktestConfig, run_quantile_backtest
from us_alpha_lab.config import load_config
from us_alpha_lab.experiment import ExperimentConfig, run_experiment
from us_alpha_lab.factor_discovery import build_factor_discovery_report
from us_alpha_lab.factors import add_alpha_factors, add_cross_sectional_ranks
from us_alpha_lab.leaderboard import build_factor_leaderboard, save_backtest_results
from us_alpha_lab.massive_data import fetch_daily_bars
from us_alpha_lab.modeling import generate_ml_predictions, train_model
from us_alpha_lab.visualization import (
    create_factor_charts,
    plot_backtest_drawdown,
    plot_backtest_equity,
    plot_leaderboard,
    plot_metric_scatter,
)

app = typer.Typer(help="US stock alpha research tools.")


def _parse_tickers(tickers: str | None, fallback: list[str]) -> list[str]:
    if not tickers:
        return fallback
    return [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]


@app.command()
def fetch(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    tickers: str | None = typer.Option(None, help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA."),
    start: str | None = typer.Option(None, help="Start date, YYYY-MM-DD."),
    end: str | None = typer.Option(None, help="End date, YYYY-MM-DD."),
) -> None:
    cfg = load_config(config)
    frame = fetch_daily_bars(
        tickers=_parse_tickers(tickers, cfg.tickers),
        start=start or cfg.start,
        end=end or cfg.end,
        output_path=cfg.raw_path,
        multiplier=cfg.multiplier,
        timespan=cfg.timespan,
        request_pause_seconds=cfg.request_pause_seconds,
    )
    typer.echo(f"Saved {len(frame):,} rows to {cfg.raw_path}")


@app.command()
def factors(config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path.")) -> None:
    cfg = load_config(config)
    bars = pd.read_parquet(cfg.raw_path)
    frame = add_cross_sectional_ranks(add_alpha_factors(bars))
    cfg.factors_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cfg.factors_path, index=False)
    typer.echo(f"Saved {len(frame):,} rows to {cfg.factors_path}")


@app.command("discover-factors")
def discover_factors(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon for IC analysis."),
    output_path: Path = typer.Option(
        Path("reports/factor_discovery/factor_candidates.csv"),
        help="Output candidate report path.",
    ),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    report_frame = build_factor_discovery_report(factor_frame, horizon=horizon)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(output_path, index=False)
    typer.echo(f"Saved factor discovery report to {output_path}")
    typer.echo(report_frame.head(12).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


@app.command()
def train(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon in trading days."),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    metrics = train_model(factor_frame, model_path=cfg.model_path, horizon=horizon)
    typer.echo(f"Saved model to {cfg.model_path}")
    for name, value in metrics.items():
        typer.echo(f"{name}: {value:.6f}")


@app.command("ml-alpha")
def ml_alpha(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon in trading days."),
    train_size: int = typer.Option(252, help="Walk-forward train window in trading days."),
    test_size: int = typer.Option(63, help="Walk-forward test window in trading days."),
    embargo: int = typer.Option(5, help="Embargo gap between train and test windows."),
    output_path: Path | None = typer.Option(None, help="Output factor table path."),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    enriched, metrics = generate_ml_predictions(
        factor_frame,
        horizon=horizon,
        train_size=train_size,
        test_size=test_size,
        embargo=embargo,
    )
    target_path = output_path or cfg.factors_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(target_path, index=False)
    typer.echo(f"Saved ML alpha factors to {target_path}")
    for name, value in metrics.items():
        typer.echo(f"{name}: {value:.6f}")


@app.command()
def report(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon in trading days."),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    report_frame = factor_ic_report(factor_frame, horizon=horizon)
    typer.echo(report_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


@app.command()
def charts(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon in trading days."),
    factor: str | None = typer.Option(None, help="Factor to chart. Defaults to best mean IC factor."),
    output_dir: Path = typer.Option(Path("reports"), help="Where chart PNG files are saved."),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    paths = create_factor_charts(
        factor_frame,
        output_dir=output_dir,
        horizon=horizon,
        factor=factor,
    )
    for path in paths:
        typer.echo(f"Saved {path}")


@app.command()
def backtest(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    factor: str = typer.Option("dollar_volume", help="Factor to rank for top-minus-bottom portfolio."),
    horizon: int = typer.Option(1, help="Forward return horizon in trading days."),
    quantiles: int = typer.Option(5, help="Number of daily factor buckets."),
    cost_bps: float = typer.Option(5.0, help="Turnover cost in basis points."),
    output_path: Path = typer.Option(Path("reports/backtest.parquet"), help="Backtest return path."),
    weights_path: Path | None = typer.Option(None, help="Optional portfolio weights path."),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    result = run_quantile_backtest(
        factor_frame,
        BacktestConfig(
            factor=factor,
            horizon=horizon,
            quantiles=quantiles,
            cost_bps=cost_bps,
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.daily_returns.to_parquet(output_path, index=False)
    typer.echo(f"Saved returns to {output_path}")
    if weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        result.weights.to_parquet(weights_path, index=False)
        typer.echo(f"Saved weights to {weights_path}")
    for name, value in result.metrics.items():
        typer.echo(f"{name}: {value:.6f}")


@app.command()
def leaderboard(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    ic_horizon: int = typer.Option(5, help="Forward return horizon for IC analysis."),
    backtest_horizon: int = typer.Option(1, help="Forward return horizon for daily backtest."),
    quantiles: int = typer.Option(5, help="Number of daily factor buckets."),
    cost_bps: float = typer.Option(5.0, help="Turnover cost in basis points."),
    output_dir: Path = typer.Option(Path("reports/leaderboard"), help="Output directory."),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    ic_report = factor_ic_report(factor_frame, horizon=ic_horizon)
    leaderboard_frame, results = build_factor_leaderboard(
        factor_frame,
        ic_report=ic_report,
        horizon=backtest_horizon,
        quantiles=quantiles,
        cost_bps=cost_bps,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "factor_leaderboard.csv"
    leaderboard_frame.to_csv(leaderboard_path, index=False)
    save_backtest_results(results, output_dir / "backtests")

    typer.echo(f"Saved leaderboard to {leaderboard_path}")
    if leaderboard_frame.empty:
        typer.echo("No valid factors were backtested.")
        return

    best_factor = str(leaderboard_frame.iloc[0]["factor"])
    plot_leaderboard(leaderboard_frame, output_dir)
    plot_metric_scatter(leaderboard_frame, output_dir)
    plot_backtest_equity(results[best_factor].daily_returns, best_factor, output_dir)
    plot_backtest_drawdown(results[best_factor].daily_returns, best_factor, output_dir)
    typer.echo(leaderboard_frame.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


@app.command()
def alpha_cluster(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    ic_horizon: int = typer.Option(5, help="Forward return horizon for IC diagnostics."),
    backtest_horizon: int = typer.Option(1, help="Forward return horizon for portfolio diagnostics."),
    quantiles: int = typer.Option(5, help="Number of daily factor buckets."),
    cost_bps: float = typer.Option(5.0, help="Turnover cost in basis points."),
    workers: int = typer.Option(1, help="Local worker count for factor diagnostics."),
    output_dir: Path = typer.Option(Path("reports/alpha_cluster"), help="Output directory."),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    diagnostics = run_alpha_cluster(
        factor_frame,
        ic_horizon=ic_horizon,
        backtest_horizon=backtest_horizon,
        quantiles=quantiles,
        cost_bps=cost_bps,
        workers=workers,
    )
    report_path, _ = save_cluster_outputs(diagnostics, output_dir)
    report_frame = diagnostics_to_frame(diagnostics)
    typer.echo(f"Saved alpha cluster report to {report_path}")
    typer.echo(report_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


@app.command()
def registry(
    output_path: Path | None = typer.Option(None, help="Optional CSV export path."),
    include_disabled: bool = typer.Option(False, help="Include disabled alpha specs."),
) -> None:
    frame = alpha_registry_frame(enabled_only=not include_disabled)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        typer.echo(f"Saved alpha registry to {output_path}")
    typer.echo(frame.to_string(index=False))


@app.command("run-experiment")
def run_experiment_cmd(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    name: str = typer.Option("free_alpha", help="Experiment name."),
    run_root: Path = typer.Option(Path("runs"), help="Run output root directory."),
    ic_horizon: int = typer.Option(5, help="Forward return horizon for IC analysis."),
    backtest_horizon: int = typer.Option(1, help="Forward return horizon for backtest."),
    quantiles: int = typer.Option(5, help="Number of daily factor buckets."),
    cost_bps: float = typer.Option(5.0, help="Turnover cost in basis points."),
    workers: int = typer.Option(1, help="Local worker count for alpha diagnostics."),
    best_factor: str | None = typer.Option(None, help="Optional factor to use for best-backtest charts."),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    result = run_experiment(
        factor_frame,
        research_config=cfg,
        research_config_path=config,
        experiment=ExperimentConfig(
            name=name,
            ic_horizon=ic_horizon,
            backtest_horizon=backtest_horizon,
            quantiles=quantiles,
            cost_bps=cost_bps,
            workers=workers,
            best_factor=best_factor,
        ),
        run_root=run_root,
    )
    typer.echo(f"Saved run to {result.run_dir}")
    typer.echo(f"Best factor: {result.best_factor}")
    typer.echo(f"Leaderboard: {result.leaderboard_path}")
    typer.echo(f"Alpha cluster: {result.cluster_report_path}")
    typer.echo(f"Metrics: {result.metrics_path}")


@app.command()
def pipeline(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon in trading days."),
) -> None:
    fetch(config=config)
    factors(config=config)
    train(config=config, horizon=horizon)


if __name__ == "__main__":
    app()
