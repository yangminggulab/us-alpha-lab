from __future__ import annotations

import numpy as np
import pandas as pd

from us_alpha_lab.modeling import generate_ml_predictions, make_training_frame, normalize_model_name


def test_generate_ml_predictions_adds_alpha_column() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
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
                    "dollar_volume": 1_000_000 + ticker_index * 1000 + date_index,
                    "dollar_volume_rank": (ticker_index + 1) / len(tickers),
                    "volatility_21d": 0.01 * (ticker_index + 1),
                    "volatility_21d_rank": (ticker_index + 1) / len(tickers),
                }
            )

    enriched, metrics = generate_ml_predictions(
        pd.DataFrame(rows),
        horizon=5,
        train_size=30,
        test_size=10,
        embargo=2,
    )

    assert "ml_prediction_5d" in enriched.columns
    assert enriched["ml_prediction_5d"].notna().sum() > 0
    assert metrics["folds"] > 0


def test_generate_ml_predictions_handles_infinite_features() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100 + date_index + ticker_index,
                    "dollar_volume": 1_000_000 + ticker_index * 1000 + date_index,
                    "dollar_volume_rank": (ticker_index + 1) / len(tickers),
                    "volatility_21d": np.inf if date_index == 15 and ticker == "AAA" else 0.01,
                    "volatility_21d_rank": (ticker_index + 1) / len(tickers),
                }
            )

    enriched, metrics = generate_ml_predictions(
        pd.DataFrame(rows),
        horizon=5,
        train_size=30,
        test_size=10,
        embargo=2,
    )

    assert enriched["ml_prediction_5d"].notna().sum() > 0
    assert metrics["prediction_rows"] > 0


def test_make_training_frame_can_use_cross_sectional_zscore_label() -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100 + date_index * (ticker_index + 1),
                    "dollar_volume": 1_000_000 + ticker_index,
                    "dollar_volume_rank": (ticker_index + 1) / len(tickers),
                }
            )

    training, _, label_column = make_training_frame(
        pd.DataFrame(rows),
        horizon=2,
        label_transform="zscore",
    )

    assert label_column == "future_return_2d_zscore"
    assert training[label_column].notna().sum() > 0


def test_make_training_frame_can_use_top_bottom_label() -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100 + date_index * (ticker_index + 1),
                    "dollar_volume": 1_000_000 + ticker_index,
                    "dollar_volume_rank": (ticker_index + 1) / len(tickers),
                }
            )

    training, _, label_column = make_training_frame(
        pd.DataFrame(rows),
        horizon=2,
        label_transform="top_bottom",
    )

    assert label_column == "future_return_2d_top_bottom"
    assert set(training[label_column].dropna().unique()) <= {0.0, 4.0}
    assert training.groupby("date")[label_column].count().max() <= 2


def test_normalize_model_name_accepts_lightgbm_ranker_aliases() -> None:
    assert normalize_model_name("lightboost") == "lightgbm"
    assert normalize_model_name("lambdarank") == "lightgbm_ranker"
    assert normalize_model_name("rank_xendcg") == "lightgbm_xendcg_ranker"
