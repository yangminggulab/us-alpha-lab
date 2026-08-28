from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from us_alpha_lab.alpha_registry import enabled_alpha_names
from us_alpha_lab.diagnostics import AlphaDiagnostic, diagnose_factor


def run_alpha_cluster(
    factors: pd.DataFrame,
    factor_columns: list[str] | None = None,
    ic_horizon: int = 5,
    backtest_horizon: int = 1,
    quantiles: int = 5,
    cost_bps: float = 5.0,
    workers: int = 1,
) -> list[AlphaDiagnostic]:
    """Run alpha diagnostics for every candidate factor."""
    candidates = factor_columns or [factor for factor in enabled_alpha_names() if factor in factors.columns]
    workers = max(1, workers)

    if workers == 1:
        return [
            diagnose_factor(
                factors,
                factor=factor,
                ic_horizon=ic_horizon,
                backtest_horizon=backtest_horizon,
                quantiles=quantiles,
                cost_bps=cost_bps,
            )
            for factor in candidates
        ]

    diagnostics: list[AlphaDiagnostic] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                diagnose_factor,
                factors,
                factor,
                ic_horizon,
                backtest_horizon,
                quantiles,
                cost_bps,
            ): factor
            for factor in candidates
        }
        for future in as_completed(futures):
            diagnostics.append(future.result())

    order = {factor: index for index, factor in enumerate(candidates)}
    return sorted(diagnostics, key=lambda item: order[item.factor])


def diagnostics_to_frame(diagnostics: list[AlphaDiagnostic]) -> pd.DataFrame:
    rows = []
    for diagnostic in diagnostics:
        rows.append(
            {
                "factor": diagnostic.factor,
                "status": diagnostic.status,
                "error_codes": ",".join(diagnostic.error_codes),
                **diagnostic.metrics,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    status_order = {"pass": 0, "review": 1, "fail": 2}
    frame["_status_order"] = frame["status"].map(status_order).fillna(3)
    frame = frame.sort_values(
        ["_status_order", "information_ratio", "mean_ic"],
        ascending=[True, False, False],
    ).drop(columns=["_status_order"])
    return frame.reset_index(drop=True)


def save_cluster_outputs(
    diagnostics: list[AlphaDiagnostic],
    output_dir: str | Path,
) -> tuple[Path, list[Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = diagnostics_to_frame(diagnostics)
    report_path = output_path / "alpha_cluster_report.csv"
    report.to_csv(report_path, index=False)

    backtest_dir = output_path / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    backtest_paths = []
    for diagnostic in diagnostics:
        if diagnostic.backtest is None:
            continue
        path = backtest_dir / f"{diagnostic.factor}_backtest.parquet"
        diagnostic.backtest.daily_returns.to_parquet(path, index=False)
        backtest_paths.append(path)

    return report_path, backtest_paths
