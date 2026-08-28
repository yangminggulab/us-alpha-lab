from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from us_alpha_lab.alpha_registry import enabled_alpha_names
from us_alpha_lab.labels import add_forward_return_label
from us_alpha_lab.validation import walk_forward_splits


def _feature_columns(data: pd.DataFrame) -> list[str]:
    alpha_names = [name for name in enabled_alpha_names() if not name.startswith("ml_prediction")]
    factor_columns = [column for column in alpha_names if column in data.columns]
    rank_columns = [f"{column}_rank" for column in alpha_names if f"{column}_rank" in data.columns]
    zscore_columns = [f"{column}_zscore" for column in alpha_names if f"{column}_zscore" in data.columns]
    return factor_columns + rank_columns + zscore_columns


def make_model_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=20,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_training_frame(
    factors: pd.DataFrame,
    horizon: int = 5,
    feature_lag: int = 1,
) -> tuple[pd.DataFrame, list[str], str]:
    """Create a supervised learning table.

    feature_lag=1 means today's prediction only uses yesterday's factor values.
    """
    data = add_forward_return_label(factors, horizon=horizon)
    feature_columns = _feature_columns(data)

    for column in feature_columns:
        data[column] = data.groupby("ticker")[column].shift(feature_lag)

    data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    label_column = f"future_return_{horizon}d"
    data = data.dropna(subset=[label_column]).reset_index(drop=True)
    return data, feature_columns, label_column


def train_model(
    factors: pd.DataFrame,
    model_path: str | Path,
    horizon: int = 5,
    test_size: float = 0.25,
) -> dict[str, float]:
    training, feature_columns, label_column = make_training_frame(factors, horizon=horizon)
    training = training.dropna(how="all", subset=feature_columns)
    if training.empty:
        raise RuntimeError("Training data is empty. Download a longer date range or more tickers.")

    dates = np.array(sorted(training["date"].unique()))
    split_index = max(1, int(len(dates) * (1 - test_size)))
    split_date = dates[split_index - 1]

    train = training[training["date"] <= split_date]
    test = training[training["date"] > split_date]
    if train.empty or test.empty:
        raise RuntimeError("Train/test split is empty. Use a longer date range.")

    pipeline = make_model_pipeline()
    pipeline.fit(train[feature_columns], train[label_column])
    predictions = pipeline.predict(test[feature_columns])

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "label_column": label_column,
            "split_date": str(pd.Timestamp(split_date).date()),
        },
        model_path,
    )

    return {
        "train_rows": float(len(train)),
        "test_rows": float(len(test)),
        "mae": float(mean_absolute_error(test[label_column], predictions)),
        "r2": float(r2_score(test[label_column], predictions)),
        "directional_accuracy": float(
            (np.sign(test[label_column].to_numpy()) == np.sign(predictions)).mean()
        ),
    }


def generate_ml_predictions(
    factors: pd.DataFrame,
    horizon: int = 5,
    train_size: int = 252,
    test_size: int = 63,
    embargo: int = 5,
    feature_lag: int = 1,
    prediction_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate out-of-sample walk-forward ML predictions and append as an alpha column."""
    prediction_column = prediction_column or f"ml_prediction_{horizon}d"
    training, feature_columns, label_column = make_training_frame(
        factors,
        horizon=horizon,
        feature_lag=feature_lag,
    )
    training = training.dropna(how="all", subset=feature_columns).copy()
    if training.empty:
        raise RuntimeError("Training data is empty. Download a longer date range or more tickers.")

    splits = walk_forward_splits(
        list(training["date"].unique()),
        train_size=train_size,
        test_size=test_size,
        embargo=embargo,
    )
    if not splits:
        raise RuntimeError("No walk-forward splits. Reduce train_size/test_size or use more history.")

    predictions = []
    fold_metrics = []
    for fold_index, split in enumerate(splits):
        train = training[(training["date"] >= split.train_start) & (training["date"] <= split.train_end)]
        test = training[(training["date"] >= split.test_start) & (training["date"] <= split.test_end)]
        if train.empty or test.empty:
            continue

        pipeline = make_model_pipeline(random_state=42 + fold_index)
        pipeline.fit(train[feature_columns], train[label_column])
        pred = pipeline.predict(test[feature_columns])
        fold = test[["ticker", "date", label_column]].copy()
        fold[prediction_column] = pred
        fold["fold"] = fold_index
        predictions.append(fold)
        fold_metrics.append(
            {
                "fold": fold_index,
                "train_rows": float(len(train)),
                "test_rows": float(len(test)),
                "mae": float(mean_absolute_error(test[label_column], pred)),
                "r2": float(r2_score(test[label_column], pred)),
                "directional_accuracy": float(
                    (np.sign(test[label_column].to_numpy()) == np.sign(pred)).mean()
                ),
            }
        )

    if not predictions:
        raise RuntimeError("No valid walk-forward predictions were produced.")

    prediction_frame = pd.concat(predictions, ignore_index=True)
    result = factors.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.merge(
        prediction_frame[["ticker", "date", prediction_column]],
        on=["ticker", "date"],
        how="left",
    )

    metrics_frame = pd.DataFrame(fold_metrics)
    metrics = {
        "folds": float(len(metrics_frame)),
        "prediction_rows": float(prediction_frame[prediction_column].notna().sum()),
        "mae": float(metrics_frame["mae"].mean()),
        "r2": float(metrics_frame["r2"].mean()),
        "directional_accuracy": float(metrics_frame["directional_accuracy"].mean()),
        "train_size": float(train_size),
        "test_size": float(test_size),
        "embargo": float(embargo),
    }
    return result, metrics
