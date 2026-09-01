from __future__ import annotations

import pandas as pd

from us_alpha_lab.experiment_validation import (
    ExperimentValidationConfig,
    build_experiment_validation_report,
    infer_experiment_factor_columns,
)


def _base_and_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=260, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    base_rows = []
    experiment_rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            close = 100 + ticker_index * date_index * 0.02 + date_index * 0.1
            public = float(ticker_index % 3)
            independent = float(ticker_index) + (date_index % 5) * 0.01
            base_rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": close,
                    "reversal_1d": public,
                    "momentum_5d": public * -0.5,
                }
            )
            experiment_rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "alpha_experiment_signal": independent,
                    "alpha_experiment_signal_rank": ticker_index / len(tickers),
                    "debug_value": independent * 2,
                }
            )
    return pd.DataFrame(base_rows), pd.DataFrame(experiment_rows)


def test_infer_experiment_factor_columns_ignores_derived_and_debug_columns() -> None:
    _, experiment = _base_and_experiment()
    assert infer_experiment_factor_columns(experiment) == ["alpha_experiment_signal"]


def test_build_experiment_validation_report_returns_shared_verdict_schema() -> None:
    base, experiment = _base_and_experiment()
    report = build_experiment_validation_report(
        experiment,
        base,
        orthogonal_pool=["reversal_1d", "momentum_5d"],
        config=ExperimentValidationConfig(horizon=1, backtest_horizon=1, min_coverage=0.1),
    )

    assert not report.empty
    row = report.iloc[0]
    assert row["factor"] == "alpha_experiment_signal"
    for column in (
        "mean_ic",
        "ic_t",
        "ic_orth",
        "ic_orth_ir",
        "net_sharpe",
        "monotonicity",
        "incremental_ok",
        "verdict",
        "validation_score",
    ):
        assert column in report.columns
    assert row["coverage"] > 0.9
    assert row["orth_days"] > 0
