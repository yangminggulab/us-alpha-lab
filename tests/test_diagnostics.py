from __future__ import annotations

import pandas as pd

from us_alpha_lab.alpha_cluster import diagnostics_to_frame, run_alpha_cluster
from us_alpha_lab.diagnostics import diagnose_factor


def _sample_factors() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100 + date_index + ticker_index,
                    "dollar_volume": ticker_index + 1,
                    "constant_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_diagnose_factor_flags_constant_cross_section() -> None:
    diagnostic = diagnose_factor(_sample_factors(), factor="constant_factor", min_active_days=1)

    assert diagnostic.status == "fail"
    assert "CONSTANT_CROSS_SECTION" in diagnostic.error_codes


def test_diagnose_factor_uses_active_window_for_ml_alpha() -> None:
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100 + date_index + ticker_index,
                    "ml_prediction_5d": ticker_index + 1 if date_index >= 80 else None,
                }
            )

    diagnostic = diagnose_factor(
        pd.DataFrame(rows),
        factor="ml_prediction_5d",
        min_active_days=10,
    )

    assert "LOW_COVERAGE" not in diagnostic.error_codes
    assert "CONSTANT_CROSS_SECTION" not in diagnostic.error_codes
    assert diagnostic.metrics["raw_coverage"] < diagnostic.metrics["coverage"]


def test_run_alpha_cluster_returns_frame() -> None:
    diagnostics = run_alpha_cluster(
        _sample_factors(),
        factor_columns=["dollar_volume", "constant_factor"],
    )
    frame = diagnostics_to_frame(diagnostics)

    assert set(frame["factor"]) == {"dollar_volume", "constant_factor"}
    assert "error_codes" in frame.columns
