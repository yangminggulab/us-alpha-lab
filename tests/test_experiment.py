from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_alpha_lab.config import ResearchConfig
from us_alpha_lab.experiment import ExperimentConfig, run_experiment


def test_run_experiment_creates_artifacts(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            close = 100 + date_index + ticker_index
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": close,
                    "dollar_volume": ticker_index + 1,
                    "volatility_21d": 5 - ticker_index + date_index * 0.01,
                }
            )

    config_path = tmp_path / "config.yaml"
    config_path.write_text("tickers: [AAA]\nstart: '2024-01-01'\nend: '2024-02-01'\n")
    result = run_experiment(
        pd.DataFrame(rows),
        research_config=ResearchConfig(
            tickers=["AAA"],
            start="2024-01-01",
            end="2024-02-01",
            raw_path=Path("raw.parquet"),
            factors_path=Path("factors.parquet"),
            model_path=Path("model.joblib"),
        ),
        research_config_path=config_path,
        experiment=ExperimentConfig(name="test_run", workers=1),
        run_root=tmp_path / "runs",
    )

    assert result.run_dir.exists()
    assert result.leaderboard_path.exists()
    assert result.cluster_report_path.exists()
    assert result.metrics_path.exists()
    assert (result.run_dir / "alpha_registry.csv").exists()
    assert (result.run_dir / "best_backtest" / "weights.parquet").exists()
