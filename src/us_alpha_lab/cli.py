from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from us_alpha_lab.a_share_l3 import (
    add_minute_forward_returns,
    available_l2_l3_days,
    build_l3_coverage_report,
    build_minute_features,
    build_order_lifecycle,
    build_quality_gate_report,
    cluster_share_report,
    lifecycle_quality_report,
    read_stock_day,
    run_hmm_force_regimes,
    run_static_force_clustering,
    state_signal_screen,
)
from us_alpha_lab.alpha_cluster import diagnostics_to_frame, run_alpha_cluster, save_cluster_outputs
from us_alpha_lab.alpha_registry import alpha_registry_frame
from us_alpha_lab.analysis import factor_ic_report
from us_alpha_lab.backtest import BacktestConfig, run_quantile_backtest
from us_alpha_lab.config import load_config
from us_alpha_lab.experiment import ExperimentConfig, run_experiment
from us_alpha_lab.experiment_validation import (
    ExperimentValidationConfig,
    build_experiment_validation_report,
    infer_experiment_factor_columns,
)
from us_alpha_lab.factor_discovery import build_factor_discovery_report
from us_alpha_lab.factors import add_alpha_factors, add_cross_sectional_ranks
from us_alpha_lab.feature_engineering import add_cross_sectional_features
from us_alpha_lab.kline_discovery import build_kline_pattern_discovery
from us_alpha_lab.kline_tokens import add_kline_sequence_factors, kline_factor_columns
from us_alpha_lab.latent_participant import (
    add_latent_participant_states,
    latent_factor_columns,
)
from us_alpha_lab.leaderboard import build_factor_leaderboard, save_backtest_results
from us_alpha_lab.massive_data import fetch_daily_bars
from us_alpha_lab.methodology import build_methodology_html
from us_alpha_lab.modeling import generate_ml_predictions, train_model, tune_lightgbm
from us_alpha_lab.report import build_html_report, build_l2_l3_html_report
from us_alpha_lab.visualization import (
    create_factor_charts,
    plot_backtest_drawdown,
    plot_backtest_equity,
    plot_leaderboard,
    plot_metric_scatter,
)

app = typer.Typer(help="US stock alpha research tools.")


def _echo_metrics(metrics: dict[str, float | str]) -> None:
    for name, value in metrics.items():
        if isinstance(value, str):
            typer.echo(f"{name}: {value}")
        else:
            typer.echo(f"{name}: {value:.6f}")


def _parse_tickers(tickers: str | None, fallback: list[str]) -> list[str]:
    if not tickers:
        return fallback
    return [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]


def _parse_csv_option(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


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


@app.command("kline-factors")
def kline_factors(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    windows: str = typer.Option("20,60", help="Comma-separated rolling windows, e.g. 20,60."),
    output_path: Path = typer.Option(
        Path("data/processed/kline_sequence_factors.parquet"),
        help="Output path for standalone K-line sequence factors.",
    ),
    include_cross_sectional: bool = typer.Option(
        True,
        help="Add same-day ranks and z-scores for K-line alpha columns.",
    ),
) -> None:
    cfg = load_config(config)
    parsed_windows = [int(value.strip()) for value in windows.split(",") if value.strip()]
    bars = pd.read_parquet(cfg.raw_path)
    frame = add_kline_sequence_factors(bars, windows=parsed_windows)
    if include_cross_sectional:
        frame = add_cross_sectional_features(frame, kline_factor_columns(parsed_windows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    typer.echo(f"Saved {len(frame):,} rows to {output_path}")


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


@app.command("discover-kline-patterns")
def discover_kline_patterns(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon for IC analysis."),
    windows: str = typer.Option("20,60", help="Comma-separated rolling windows, e.g. 20,60."),
    kline_factors_path: Path | None = typer.Option(
        Path("data/processed/kline_sequence_factors.parquet"),
        help="Optional K-line factor parquet path; falls back to raw bars when missing.",
    ),
    output_path: Path = typer.Option(
        Path("reports/kline_discovery/kline_pattern_candidates.csv"),
        help="Output path for K-line pattern discovery candidates.",
    ),
    max_token_patterns: int = typer.Option(
        24,
        help="Number of observed full K-line tokens to include as candidate events.",
    ),
    min_coverage: float = typer.Option(0.2, help="Minimum non-null candidate coverage."),
    top_n: int = typer.Option(20, help="Number of top candidates to print."),
) -> None:
    cfg = load_config(config)
    parsed_windows = [int(value.strip()) for value in windows.split(",") if value.strip()]
    source_path = (
        kline_factors_path
        if kline_factors_path is not None and kline_factors_path.exists()
        else cfg.raw_path
    )
    source = pd.read_parquet(source_path)
    base_factors = pd.read_parquet(cfg.factors_path)
    report_frame = build_kline_pattern_discovery(
        source,
        base_factors,
        horizon=horizon,
        windows=parsed_windows,
        max_token_patterns=max_token_patterns,
        min_coverage=min_coverage,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(output_path, index=False)
    typer.echo(f"Saved K-line pattern discovery report to {output_path}")
    typer.echo(report_frame.head(top_n).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


@app.command("validate-experiment")
def validate_experiment(
    experiment_path: Path = typer.Argument(..., help="Experiment parquet/CSV with ticker/date + alpha columns."),
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon for IC analysis."),
    backtest_horizon: int = typer.Option(1, help="Forward return horizon for quantile backtest."),
    factor_columns: str | None = typer.Option(
        None,
        help="Comma-separated experiment factor columns. Defaults to inferred alpha_* columns.",
    ),
    factor_prefixes: str = typer.Option(
        "alpha_,ml_prediction",
        help="Comma-separated prefixes used when inferring factor columns.",
    ),
    orthogonal_pool: str | None = typer.Option(
        None,
        help="Comma-separated public factor pool. Defaults to enabled non-ML main factors.",
    ),
    output_path: Path = typer.Option(
        Path("reports/experiment_validation/latest_validation.csv"),
        help="Output validation CSV path.",
    ),
    min_coverage: float = typer.Option(0.2, help="Minimum non-null factor coverage."),
    quantiles: int = typer.Option(5, help="Number of daily factor buckets."),
    cost_bps: float = typer.Option(5.0, help="Turnover cost in basis points."),
    top_n: int = typer.Option(20, help="Number of top rows to print."),
) -> None:
    cfg = load_config(config)
    experiment = (
        pd.read_csv(experiment_path)
        if experiment_path.suffix.lower() == ".csv"
        else pd.read_parquet(experiment_path)
    )
    base_factors = pd.read_parquet(cfg.factors_path)
    prefixes = tuple(_parse_csv_option(factor_prefixes) or ["alpha_", "ml_prediction"])
    parsed_factor_columns = _parse_csv_option(factor_columns)
    inferred = parsed_factor_columns or infer_experiment_factor_columns(experiment, prefixes=prefixes)
    report_frame = build_experiment_validation_report(
        experiment,
        base_factors,
        factor_columns=inferred,
        orthogonal_pool=_parse_csv_option(orthogonal_pool),
        config=ExperimentValidationConfig(
            horizon=horizon,
            backtest_horizon=backtest_horizon,
            quantiles=quantiles,
            cost_bps=cost_bps,
            min_coverage=min_coverage,
            factor_prefixes=prefixes,
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(output_path, index=False)
    typer.echo(f"Saved experiment validation report to {output_path}")
    if report_frame.empty:
        typer.echo("No valid experiment factors were found.")
    else:
        typer.echo(report_frame.head(top_n).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


@app.command("latent-participants")
def latent_participants(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    lookback: int = typer.Option(21, help="Rolling lookback for low-frequency observation z-scores."),
    persistence: float = typer.Option(0.86, help="HMM-style state persistence in [0, 1)."),
    emission_scale: float = typer.Option(1.25, help="Gaussian emission distance scale."),
    output_path: Path = typer.Option(
        Path("data/processed/latent_participant_states.parquet"),
        help="Output path for standalone latent participant states.",
    ),
    include_cross_sectional: bool = typer.Option(
        True,
        help="Add same-day ranks and z-scores for latent participant alpha columns.",
    ),
) -> None:
    cfg = load_config(config)
    bars = pd.read_parquet(cfg.raw_path)
    frame = add_latent_participant_states(
        bars,
        lookback=lookback,
        persistence=persistence,
        emission_scale=emission_scale,
    )
    if include_cross_sectional:
        frame = add_cross_sectional_features(frame, latent_factor_columns())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    typer.echo(f"Saved {len(frame):,} rows to {output_path}")


@app.command("a-share-l3-lifecycle")
def a_share_l3_lifecycle(
    date: str = typer.Option(..., help="Trading day, e.g. 20170123."),
    wind_code: str = typer.Option(..., help="A-share Wind code, e.g. 000725.SZ."),
    raw_dir: Path = typer.Option(
        Path("data/raw/a_share_l2_hf"),
        help="Raw A-share L2/L3 archive directory.",
    ),
    output_path: Path | None = typer.Option(
        None,
        help="Output lifecycle parquet path. Defaults to reports/l2_l3/lifecycle_<date>_<ticker>.parquet.",
    ),
    quality_path: Path | None = typer.Option(
        None,
        help="Optional one-row CSV quality report path.",
    ),
) -> None:
    stock_day = read_stock_day(raw_dir, date, wind_code, include_quotes=True)
    lifecycle = build_order_lifecycle(stock_day.orders, stock_day.trades)
    ticker_slug = wind_code.replace(".", "_")
    target = output_path or Path("reports/l2_l3") / f"lifecycle_{date}_{ticker_slug}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    lifecycle.to_parquet(target, index=False)

    quality = lifecycle_quality_report(
        stock_day.orders,
        stock_day.trades,
        lifecycle,
        quotes=stock_day.quotes,
    )
    if quality_path is not None:
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([quality]).to_csv(quality_path, index=False)
    typer.echo(f"Saved {len(lifecycle):,} lifecycle rows to {target}")
    typer.echo(pd.DataFrame([quality]).to_string(index=False))


@app.command("a-share-l3-coverage")
def a_share_l3_coverage(
    raw_dir: Path = typer.Option(
        Path("data/raw/a_share_l2_hf"),
        help="Raw A-share L2/L3 archive directory.",
    ),
    dates: str | None = typer.Option(
        None,
        help="Comma-separated trading days. Defaults to all complete local days.",
    ),
    output_path: Path = typer.Option(
        Path("reports/l2_l3/daily_coverage.csv"),
        help="Output daily coverage CSV path.",
    ),
) -> None:
    parsed_dates = _parse_csv_option(dates)
    selected_dates = parsed_dates
    if selected_dates is None:
        selected_dates = available_l2_l3_days(raw_dir)
    rows = []
    for date in selected_dates:
        typer.echo(f"Scanning {date}...")
        rows.append(build_l3_coverage_report(raw_dir, dates=[date]))
    report_frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(output_path, index=False)
    typer.echo(f"Saved {len(report_frame):,} daily coverage rows to {output_path}")
    typer.echo(report_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


@app.command("a-share-l3-minute-features")
def a_share_l3_minute_features(
    date: str = typer.Option(..., help="Trading day, e.g. 20170123."),
    wind_code: str = typer.Option(..., help="A-share Wind code, e.g. 000725.SZ."),
    raw_dir: Path = typer.Option(
        Path("data/raw/a_share_l2_hf"),
        help="Raw A-share L2/L3 archive directory.",
    ),
    output_path: Path | None = typer.Option(
        None,
        help="Output minute feature parquet path. Defaults to reports/l2_l3/minute_features_<date>_<ticker>.parquet.",
    ),
) -> None:
    stock_day = read_stock_day(raw_dir, date, wind_code, include_quotes=True)
    lifecycle = build_order_lifecycle(stock_day.orders, stock_day.trades)
    minute_features = build_minute_features(
        lifecycle,
        stock_day.trades,
        quotes=stock_day.quotes,
    )
    ticker_slug = wind_code.replace(".", "_")
    target = output_path or Path("reports/l2_l3") / f"minute_features_{date}_{ticker_slug}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    minute_features.to_parquet(target, index=False)
    typer.echo(f"Saved {len(minute_features):,} minute feature rows to {target}")
    typer.echo(minute_features.head(12).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


@app.command("a-share-l3-quality-gates")
def a_share_l3_quality_gates(
    dates: str = typer.Option(..., help="Comma-separated trading days, e.g. 20170123,20170124."),
    wind_codes: str = typer.Option(..., help="Comma-separated SZ Wind codes, e.g. 000725.SZ,000001.SZ."),
    raw_dir: Path = typer.Option(
        Path("data/raw/a_share_l2_hf"),
        help="Raw A-share L2/L3 archive directory.",
    ),
    output_path: Path = typer.Option(
        Path("reports/l2_l3/minute_quality_gates.csv"),
        help="Output quality gate CSV path.",
    ),
) -> None:
    parsed_dates = _parse_csv_option(dates) or []
    parsed_wind_codes = _parse_csv_option(wind_codes) or []
    rows = []
    for date in parsed_dates:
        for wind_code in parsed_wind_codes:
            typer.echo(f"Checking {date} {wind_code}...")
            rows.append(build_quality_gate_report(raw_dir, dates=[date], wind_codes=[wind_code]))
    report_frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(output_path, index=False)
    typer.echo(f"Saved {len(report_frame):,} quality gate rows to {output_path}")
    typer.echo(report_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


@app.command("a-share-l3-static-clusters")
def a_share_l3_static_clusters(
    dates: str = typer.Option(..., help="Comma-separated trading days, e.g. 20170123,20170124."),
    wind_codes: str = typer.Option(..., help="Comma-separated SZ Wind codes, e.g. 000725.SZ,000001.SZ."),
    raw_dir: Path = typer.Option(
        Path("data/raw/a_share_l2_hf"),
        help="Raw A-share L2/L3 archive directory.",
    ),
    output_dir: Path = typer.Option(
        Path("reports/l2_l3/static_clusters"),
        help="Directory for cluster labels, diagnostics, and profile outputs.",
    ),
    k_values: str = typer.Option("2,3,4,5,6", help="Comma-separated candidate cluster counts."),
    sessions: str = typer.Option(
        "continuous",
        help="Comma-separated sessions to include, or 'all'.",
    ),
    feature_columns: str | None = typer.Option(
        None,
        help="Optional comma-separated minute feature columns. Defaults to the L3 static feature set.",
    ),
    clip_quantile: float = typer.Option(0.01, help="Two-sided winsorization quantile."),
    random_state: int = typer.Option(0, help="KMeans random seed."),
) -> None:
    parsed_dates = _parse_csv_option(dates) or []
    parsed_wind_codes = _parse_csv_option(wind_codes) or []
    minute_frames = []
    for date in parsed_dates:
        for wind_code in parsed_wind_codes:
            typer.echo(f"Building minute features for {date} {wind_code}...")
            stock_day = read_stock_day(raw_dir, date, wind_code, include_quotes=True)
            lifecycle = build_order_lifecycle(stock_day.orders, stock_day.trades)
            minute_frames.append(
                build_minute_features(
                    lifecycle,
                    stock_day.trades,
                    quotes=stock_day.quotes,
                )
            )
    minute_features = pd.concat(minute_frames, ignore_index=True) if minute_frames else pd.DataFrame()
    parsed_k_values = [int(value) for value in (_parse_csv_option(k_values) or [])]
    parsed_sessions = None if sessions.strip().lower() == "all" else _parse_csv_option(sessions)
    result = run_static_force_clustering(
        minute_features,
        feature_columns=_parse_csv_option(feature_columns),
        k_values=parsed_k_values,
        sessions=parsed_sessions,
        clip_quantile=clip_quantile,
        random_state=random_state,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.parquet"
    diagnostics_path = output_dir / "diagnostics.csv"
    profile_path = output_dir / "profile.csv"
    shares_path = output_dir / "shares.csv"
    result.labels.to_parquet(labels_path, index=False)
    result.diagnostics.to_csv(diagnostics_path, index=False)
    result.profile.to_csv(profile_path, index=False)
    shares = cluster_share_report(result.labels)
    shares.to_csv(shares_path, index=False)
    typer.echo(f"Selected K={result.selected_k} using features: {', '.join(result.feature_columns)}")
    typer.echo(f"Saved labels to {labels_path}")
    typer.echo(f"Saved diagnostics to {diagnostics_path}")
    typer.echo(f"Saved profile to {profile_path}")
    typer.echo(f"Saved shares to {shares_path}")
    typer.echo(result.diagnostics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    typer.echo(result.profile.head(12).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


@app.command("a-share-l3-hmm-states")
def a_share_l3_hmm_states(
    dates: str = typer.Option(..., help="Comma-separated trading days, e.g. 20170123,20170124."),
    wind_codes: str = typer.Option(..., help="Comma-separated SZ Wind codes, e.g. 000725.SZ,000001.SZ."),
    raw_dir: Path = typer.Option(
        Path("data/raw/a_share_l2_hf"),
        help="Raw A-share L2/L3 archive directory.",
    ),
    output_dir: Path = typer.Option(
        Path("reports/l2_l3/hmm_states"),
        help="Directory for HMM labels, diagnostics, profile, and transition outputs.",
    ),
    k_values: str = typer.Option("2,3,4,5,6", help="Comma-separated candidate HMM state counts."),
    sessions: str = typer.Option(
        "continuous",
        help="Comma-separated sessions to include, or 'all'.",
    ),
    feature_columns: str | None = typer.Option(
        None,
        help="Optional comma-separated minute feature columns. Defaults to the L3 HMM feature set.",
    ),
    clip_quantile: float = typer.Option(0.01, help="Two-sided winsorization quantile."),
    max_iter: int = typer.Option(50, help="Maximum EM iterations per K."),
    tol: float = typer.Option(1e-4, help="EM log-likelihood convergence tolerance."),
    random_state: int = typer.Option(0, help="KMeans initialization random seed."),
) -> None:
    parsed_dates = _parse_csv_option(dates) or []
    parsed_wind_codes = _parse_csv_option(wind_codes) or []
    minute_frames = []
    for date in parsed_dates:
        for wind_code in parsed_wind_codes:
            typer.echo(f"Building minute features for {date} {wind_code}...")
            stock_day = read_stock_day(raw_dir, date, wind_code, include_quotes=True)
            lifecycle = build_order_lifecycle(stock_day.orders, stock_day.trades)
            minute_frames.append(
                build_minute_features(
                    lifecycle,
                    stock_day.trades,
                    quotes=stock_day.quotes,
                )
            )
    minute_features = pd.concat(minute_frames, ignore_index=True) if minute_frames else pd.DataFrame()
    parsed_k_values = [int(value) for value in (_parse_csv_option(k_values) or [])]
    parsed_sessions = None if sessions.strip().lower() == "all" else _parse_csv_option(sessions)
    result = run_hmm_force_regimes(
        minute_features,
        feature_columns=_parse_csv_option(feature_columns),
        k_values=parsed_k_values,
        sessions=parsed_sessions,
        clip_quantile=clip_quantile,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.parquet"
    diagnostics_path = output_dir / "diagnostics.csv"
    profile_path = output_dir / "profile.csv"
    transitions_path = output_dir / "transitions.csv"
    result.labels.to_parquet(labels_path, index=False)
    result.diagnostics.to_csv(diagnostics_path, index=False)
    result.profile.to_csv(profile_path, index=False)
    result.transitions.to_csv(transitions_path, index=False)
    typer.echo(f"Selected HMM K={result.selected_k} using features: {', '.join(result.feature_columns)}")
    typer.echo(f"Saved labels to {labels_path}")
    typer.echo(f"Saved diagnostics to {diagnostics_path}")
    typer.echo(f"Saved profile to {profile_path}")
    typer.echo(f"Saved transitions to {transitions_path}")
    typer.echo(result.diagnostics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    typer.echo(result.profile.head(12).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


@app.command("a-share-l3-hmm-signal-screen")
def a_share_l3_hmm_signal_screen(
    labels_path: Path = typer.Option(
        Path("reports/l2_l3/hmm_states_sample/labels.parquet"),
        help="Label parquet from a-share-l3-hmm-states (must carry quote_last_price_scaled).",
    ),
    state_col: str = typer.Option("hmm_state", help="Column holding the regime label."),
    horizons: str = typer.Option("1,3,5", help="Comma-separated forward return horizons in minutes."),
    output_dir: Path = typer.Option(
        Path("reports/l2_l3/hmm_state_screen"),
        help="Directory for the forward labels and screen tables.",
    ),
) -> None:
    labels = pd.read_parquet(labels_path)
    parsed_horizons = _parse_csv_option(horizons)
    if not parsed_horizons:
        raise typer.BadParameter("horizons must be a non-empty comma-separated list")
    horizon_values = [int(value) for value in parsed_horizons]
    labels = add_minute_forward_returns(labels, horizons=horizon_values)
    result = state_signal_screen(labels, state_col=state_col, horizons=horizon_values)

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path_out = output_dir / "forward_labels.parquet"
    labels.to_parquet(labels_path_out, index=False)
    result.summary.to_csv(output_dir / "screen_by_state.csv", index=False)
    result.cell_level.to_csv(output_dir / "cell_level.csv", index=False)
    result.spread.to_csv(output_dir / "spread.csv", index=False)
    typer.echo(f"Saved forward labels to {labels_path_out}")
    typer.echo(f"Saved summary / cell_level / spread CSVs to {output_dir}")
    typer.echo("")
    typer.echo("=== 状态条件前向收益 (pooled, 描述性) ===")
    typer.echo(result.summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    typer.echo("")
    typer.echo("=== 状态顶底 spread (bps) ===")
    typer.echo(result.spread.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    typer.echo(
        "\nCaveat: HMM labels are fit on the full day (Viterbi decoding sees the future), so this "
        "screen is descriptive feasibility, NOT an estimate of a tradable edge. Evidence rests on "
        "only 6 stock-day cells; overlap/serial correlation inflates the pooled t-stats."
    )


@app.command()
def train(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon in trading days."),
    model: str = typer.Option(
        "random_forest",
        help="Model backend: random_forest, lightgbm, lightgbm_ranker, or rank_xendcg.",
    ),
    label_transform: str = typer.Option(
        "return",
        help="Training label transform: return, rank, zscore, quantile, or top_bottom.",
    ),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    metrics = train_model(
        factor_frame,
        model_path=cfg.model_path,
        horizon=horizon,
        model_name=model,
        label_transform=label_transform,
    )
    typer.echo(f"Saved model to {cfg.model_path}")
    _echo_metrics(metrics)


@app.command("ml-alpha")
def ml_alpha(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon in trading days."),
    train_size: int = typer.Option(252, help="Walk-forward train window in trading days."),
    test_size: int = typer.Option(63, help="Walk-forward test window in trading days."),
    embargo: int = typer.Option(5, help="Embargo gap between train and test windows."),
    model: str = typer.Option(
        "random_forest",
        help="Model backend: random_forest, lightgbm, lightgbm_ranker, or rank_xendcg.",
    ),
    label_transform: str = typer.Option(
        "return",
        help="Training label transform: return, rank, zscore, quantile, or top_bottom.",
    ),
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
        model_name=model,
        label_transform=label_transform,
    )
    target_path = output_path or cfg.factors_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(target_path, index=False)
    typer.echo(f"Saved ML alpha factors to {target_path}")
    _echo_metrics(metrics)


@app.command("tune-lightgbm")
def tune_lightgbm_cmd(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon in trading days."),
    train_size: int = typer.Option(252, help="Walk-forward train window in trading days."),
    test_size: int = typer.Option(63, help="Walk-forward test window in trading days."),
    embargo: int = typer.Option(5, help="Embargo gap between train and test windows."),
    max_trials: int = typer.Option(24, help="Maximum parameter combinations to evaluate."),
    output_path: Path = typer.Option(
        Path("reports/lightgbm_tuning.csv"),
        help="CSV leaderboard for LightGBM tuning results.",
    ),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    results = tune_lightgbm(
        factor_frame,
        horizon=horizon,
        train_size=train_size,
        test_size=test_size,
        embargo=embargo,
        max_trials=max_trials,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    typer.echo(f"Saved LightGBM tuning results to {output_path}")
    typer.echo(results.head(10).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


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


@app.command("html-report")
def html_report_cmd(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    horizon: int = typer.Option(5, help="Forward return horizon in trading days."),
    top_n: int = typer.Option(5, help="Number of top factors to chart in the report."),
    factor_top_n: int | None = typer.Option(
        None,
        help="Optional IC-ranked limit for backtests and expensive report sections.",
    ),
    verdict_top_n: int | None = typer.Option(
        None,
        help="Optional limit for the costly five-gate verdict table, ranked by IC.",
    ),
    backtest_horizon: int = typer.Option(1, help="Forward return horizon for the quantile backtest."),
    output_path: Path = typer.Option(Path("research_report.html"), help="Output HTML report path."),
    kline_factors_path: Path | None = typer.Option(
        Path("data/processed/kline_sequence_factors.parquet"),
        help="Optional standalone K-line factor parquet path; ignored when missing.",
    ),
    kline_discovery_path: Path | None = typer.Option(
        Path("reports/kline_discovery/kline_pattern_candidates.csv"),
        help="Optional K-line discovery CSV path; ignored when missing.",
    ),
    latent_states_path: Path | None = typer.Option(
        Path("data/processed/latent_participant_states.parquet"),
        help="Optional latent participant state parquet path; ignored when missing.",
    ),
    experiment_validation_paths: str | None = typer.Option(
        "reports/experiment_validation/kline_sequence_validation.csv,"
        "reports/experiment_validation/latent_participant_validation.csv",
        help="Comma-separated experiment validation CSV paths; missing files are ignored.",
    ),
    l2_l3_report_dir: Path | None = typer.Option(
        Path("reports/l2_l3"),
        help="Optional A-share L2/L3 report directory; ignored when disabled.",
    ),
    l2_l3_report_link: str | None = typer.Option(
        "l2_l3_report.html",
        help="Link to the standalone A-share L2/L3 report (None to disable).",
    ),
) -> None:
    cfg = load_config(config)
    factor_frame = pd.read_parquet(cfg.factors_path)
    kline_frame = (
        pd.read_parquet(kline_factors_path)
        if kline_factors_path is not None and kline_factors_path.exists()
        else None
    )
    kline_discovery_frame = (
        pd.read_csv(kline_discovery_path)
        if kline_discovery_path is not None and kline_discovery_path.exists()
        else None
    )
    latent_states_frame = (
        pd.read_parquet(latent_states_path)
        if latent_states_path is not None and latent_states_path.exists()
        else None
    )
    experiment_validations = {}
    for value in _parse_csv_option(experiment_validation_paths) or []:
        validation_path = Path(value)
        if validation_path.exists():
            experiment_validations[validation_path.stem] = pd.read_csv(validation_path)
    path = build_html_report(
        factor_frame,
        output_path=output_path,
        horizon=horizon,
        top_n=top_n,
        factor_top_n=factor_top_n,
        verdict_top_n=verdict_top_n,
        backtest_horizon=backtest_horizon,
        kline_factors=kline_frame,
        kline_discovery=kline_discovery_frame,
        latent_states=latent_states_frame,
        experiment_validations=experiment_validations,
        l2_l3_report_dir=l2_l3_report_dir,
        l2_l3_report_link=l2_l3_report_link,
    )
    typer.echo(f"Saved HTML report to {path}")


@app.command("l2-l3-report")
def l2_l3_report_cmd(
    report_dir: Path = typer.Option(
        Path("reports/l2_l3"),
        help="A-share L2/L3 report directory.",
    ),
    output_path: Path = typer.Option(Path("l2_l3_report.html"), help="Output L2/L3 HTML path."),
    main_report_link: str | None = typer.Option(
        "research_report.html",
        help="Link back to the main research report (None to disable).",
    ),
    methodology_link: str | None = typer.Option(
        "methodology.html",
        help="Link to methodology HTML (None to disable).",
    ),
) -> None:
    path = build_l2_l3_html_report(
        report_dir=report_dir,
        output_path=output_path,
        main_report_link=main_report_link,
        methodology_link=methodology_link,
    )
    typer.echo(f"Saved L2/L3 HTML report to {path}")


@app.command()
def methodology(
    config: Path = typer.Option(Path("configs/universe.yaml"), help="Research config path."),
    output_path: Path = typer.Option(Path("methodology.html"), help="Output methodology HTML path."),
    report_link: str | None = typer.Option("research_report.html", help="Link back to research report (None to disable)."),
    l2_l3_report_link: str | None = typer.Option(
        "l2_l3_report.html",
        help="Link to the standalone A-share L2/L3 report (None to disable).",
    ),
) -> None:
    cfg = load_config(config)
    path = build_methodology_html(
        cfg,
        output_path=output_path,
        report_link=report_link,
        l2_l3_report_link=l2_l3_report_link,
    )
    typer.echo(f"Saved methodology HTML to {path}")


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
