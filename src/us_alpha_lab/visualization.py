from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("reports/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from us_alpha_lab.analysis import daily_factor_ic, factor_ic_report, quantile_return_report
from us_alpha_lab.labels import add_forward_return_label


def _save_current(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_ic_summary(factors: pd.DataFrame, output_dir: Path, horizon: int = 5, top_n: int = 10) -> Path:
    report = factor_ic_report(factors, horizon=horizon).head(top_n)
    path = output_dir / "ic_summary.png"

    plt.figure(figsize=(10, 5))
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in report["mean_ic"]]
    plt.barh(report["factor"], report["mean_ic"], color=colors)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title(f"Mean Spearman IC by Factor ({horizon}D Forward Return)")
    plt.xlabel("Mean IC")
    plt.gca().invert_yaxis()
    return _save_current(path)


def plot_ic_timeseries(
    factors: pd.DataFrame,
    output_dir: Path,
    horizon: int = 5,
    factor: str | None = None,
) -> Path:
    report = factor_ic_report(factors, horizon=horizon)
    factor = factor or str(report.iloc[0]["factor"])
    data = add_forward_return_label(factors, horizon=horizon)
    label = f"future_return_{horizon}d"
    daily_ic = daily_factor_ic(data, factor=factor, label=label)
    rolling_ic = daily_ic.rolling(21, min_periods=5).mean()
    path = output_dir / f"{factor}_ic_timeseries.png"

    plt.figure(figsize=(11, 5))
    plt.plot(daily_ic.index, daily_ic.values, color="#8c8c8c", linewidth=0.8, alpha=0.45, label="Daily IC")
    plt.plot(rolling_ic.index, rolling_ic.values, color="#1f77b4", linewidth=2, label="21D Average")
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title(f"{factor}: Daily IC and 21D Average")
    plt.ylabel("Spearman IC")
    plt.legend()
    return _save_current(path)


def plot_quantile_returns(
    factors: pd.DataFrame,
    output_dir: Path,
    horizon: int = 5,
    factor: str | None = None,
    quantiles: int = 5,
) -> Path:
    report = factor_ic_report(factors, horizon=horizon)
    factor = factor or str(report.iloc[0]["factor"])
    quantile_returns = quantile_return_report(
        factors,
        factor=factor,
        horizon=horizon,
        quantiles=quantiles,
    )
    mean_returns = quantile_returns.groupby("quantile")["mean_return"].mean()
    path = output_dir / f"{factor}_quantile_returns.png"

    plt.figure(figsize=(9, 5))
    plt.bar(mean_returns.index.astype(str), mean_returns.values, color="#2ca02c")
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title(f"{factor}: Average {horizon}D Forward Return by Quantile")
    plt.xlabel("Factor Quantile, Low to High")
    plt.ylabel("Average Forward Return")
    return _save_current(path)


def plot_cumulative_spread(
    factors: pd.DataFrame,
    output_dir: Path,
    horizon: int = 5,
    factor: str | None = None,
    quantiles: int = 5,
) -> Path:
    report = factor_ic_report(factors, horizon=horizon)
    factor = factor or str(report.iloc[0]["factor"])
    quantile_returns = quantile_return_report(
        factors,
        factor=factor,
        horizon=horizon,
        quantiles=quantiles,
    )
    pivot = quantile_returns.pivot(index="date", columns="quantile", values="mean_return").sort_index()
    spread = (pivot[quantiles] - pivot[1]).dropna()
    cumulative = (1 + spread).cumprod() - 1
    path = output_dir / f"{factor}_cumulative_spread.png"

    plt.figure(figsize=(11, 5))
    plt.plot(cumulative.index, cumulative.values, color="#9467bd", linewidth=2)
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title(f"{factor}: Cumulative Top-minus-Bottom Quantile Return")
    plt.ylabel("Cumulative Return")
    return _save_current(path)


def create_factor_charts(
    factors: pd.DataFrame,
    output_dir: str | Path = "reports",
    horizon: int = 5,
    factor: str | None = None,
) -> list[Path]:
    output_path = Path(output_dir)
    return [
        plot_ic_summary(factors, output_path, horizon=horizon),
        plot_ic_timeseries(factors, output_path, horizon=horizon, factor=factor),
        plot_quantile_returns(factors, output_path, horizon=horizon, factor=factor),
        plot_cumulative_spread(factors, output_path, horizon=horizon, factor=factor),
    ]


def plot_leaderboard(leaderboard: pd.DataFrame, output_dir: str | Path, top_n: int = 10) -> Path:
    output_path = Path(output_dir)
    path = output_path / "factor_leaderboard.png"
    data = leaderboard.head(top_n).sort_values("score")

    plt.figure(figsize=(10, 5))
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in data["score"]]
    plt.barh(data["factor"], data["score"], color=colors)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title("Factor Leaderboard Score")
    plt.xlabel("Score")
    return _save_current(path)


def plot_backtest_equity(backtest_returns: pd.DataFrame, factor: str, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    path = output_path / f"{factor}_backtest_equity.png"

    plt.figure(figsize=(11, 5))
    plt.plot(
        pd.to_datetime(backtest_returns["date"]),
        backtest_returns["cumulative_spread_return"],
        color="#1f77b4",
        linewidth=2,
    )
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title(f"{factor}: Backtest Cumulative Top-minus-Bottom Return")
    plt.ylabel("Cumulative Return")
    return _save_current(path)


def plot_backtest_drawdown(backtest_returns: pd.DataFrame, factor: str, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    path = output_path / f"{factor}_backtest_drawdown.png"
    equity = 1 + backtest_returns["cumulative_spread_return"]
    drawdown = equity / equity.cummax() - 1

    plt.figure(figsize=(11, 4))
    plt.fill_between(pd.to_datetime(backtest_returns["date"]), drawdown, 0, color="#d62728", alpha=0.35)
    plt.plot(pd.to_datetime(backtest_returns["date"]), drawdown, color="#d62728", linewidth=1)
    plt.title(f"{factor}: Backtest Drawdown")
    plt.ylabel("Drawdown")
    return _save_current(path)


def plot_metric_scatter(leaderboard: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    path = output_path / "ic_vs_information_ratio.png"

    plt.figure(figsize=(8, 6))
    sizes = np.clip(leaderboard["hit_rate"].fillna(0.5).to_numpy() * 160, 40, 140)
    plt.scatter(
        leaderboard["mean_ic"],
        leaderboard["information_ratio"],
        s=sizes,
        c=leaderboard["score"],
        cmap="coolwarm",
        alpha=0.8,
        edgecolor="#333333",
        linewidth=0.5,
    )
    for _, row in leaderboard.head(6).iterrows():
        plt.annotate(row["factor"], (row["mean_ic"], row["information_ratio"]), fontsize=8)
    plt.axhline(0, color="#333333", linewidth=1)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title("Mean IC vs Information Ratio")
    plt.xlabel("Mean IC")
    plt.ylabel("Information Ratio")
    plt.colorbar(label="Score")
    return _save_current(path)
