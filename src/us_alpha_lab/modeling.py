from __future__ import annotations

from itertools import product
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
from us_alpha_lab.labels import add_cross_sectional_return_label
from us_alpha_lab.validation import walk_forward_splits

ModelName = str


def _feature_columns(data: pd.DataFrame) -> list[str]:
    alpha_names = [name for name in enabled_alpha_names() if not name.startswith("ml_prediction")]
    factor_columns = [column for column in alpha_names if column in data.columns]
    rank_columns = [f"{column}_rank" for column in alpha_names if f"{column}_rank" in data.columns]
    zscore_columns = [f"{column}_zscore" for column in alpha_names if f"{column}_zscore" in data.columns]
    return factor_columns + rank_columns + zscore_columns


def normalize_model_name(model_name: str) -> ModelName:
    normalized = model_name.strip().lower().replace("-", "_")
    aliases = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest": "random_forest",
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "lightboost": "lightgbm",
        "lightgbm": "lightgbm",
        "lambdarank": "lightgbm_ranker",
        "lgbm_ranker": "lightgbm_ranker",
        "lightgbm_ranker": "lightgbm_ranker",
        "ranker": "lightgbm_ranker",
    }
    if normalized not in aliases:
        allowed = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown model '{model_name}'. Expected one of: {allowed}")
    return aliases[normalized]


def make_model_pipeline(random_state: int = 42, model_name: str = "random_forest") -> Pipeline:
    model_name = normalize_model_name(model_name)
    if model_name == "lightgbm_ranker":
        raise ValueError("Use fit_predict_model for lightgbm_ranker because it requires date groups.")
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is not installed. Install it with `.venv/bin/pip install lightgbm`."
            ) from exc
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        objective="regression",
                        n_estimators=1000,
                        learning_rate=0.03,
                        num_leaves=31,
                        min_child_samples=50,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )

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
    label_transform: str = "return",
    label_quantiles: int = 5,
) -> tuple[pd.DataFrame, list[str], str]:
    """Create a supervised learning table.

    feature_lag=1 means today's prediction only uses yesterday's factor values.
    """
    data, label_column = add_cross_sectional_return_label(
        factors,
        horizon=horizon,
        transform=label_transform,
        quantiles=label_quantiles,
    )
    feature_columns = _feature_columns(data)

    for column in feature_columns:
        data[column] = data.groupby("ticker")[column].shift(feature_lag)

    data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[label_column]).reset_index(drop=True)
    return data, feature_columns, label_column


def _date_group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("date", sort=False).size().astype(int).tolist()


def _split_fit_validation(frame: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(frame["date"].unique()))
    if len(dates) < 5:
        return frame, frame.iloc[0:0]
    split_index = max(1, int(len(dates) * (1 - validation_fraction)))
    if split_index >= len(dates):
        return frame, frame.iloc[0:0]
    split_date = dates[split_index - 1]
    fit = frame[frame["date"] <= split_date]
    valid = frame[frame["date"] > split_date]
    return fit, valid


def _lightgbm_params(
    model_name: str,
    random_state: int,
    overrides: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str]:
    common: dict[str, float | int | str] = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if model_name == "lightgbm_ranker":
        params = {**common, "objective": "lambdarank", "metric": "ndcg"}
    else:
        params = {**common, "objective": "regression", "metric": "l2"}
    if overrides:
        params.update(overrides)
    return params


def fit_predict_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str = "random_forest",
    random_state: int = 42,
    model_params: dict[str, float | int | str] | None = None,
) -> tuple[object, np.ndarray]:
    """Fit one model fold and predict the test frame."""
    model_name = normalize_model_name(model_name)
    if model_name == "random_forest":
        pipeline = make_model_pipeline(random_state=random_state, model_name=model_name)
        pipeline.fit(train[feature_columns], train[label_column])
        return pipeline, pipeline.predict(test[feature_columns])

    try:
        from lightgbm import LGBMRanker, LGBMRegressor, early_stopping, log_evaluation
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is not installed. Install it with `.venv/bin/pip install lightgbm`."
        ) from exc

    fit, valid = _split_fit_validation(train)
    fit = fit.sort_values(["date", "ticker"])
    valid = valid.sort_values(["date", "ticker"])
    test = test.sort_values(["date", "ticker"])

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(fit[feature_columns])
    x_test = imputer.transform(test[feature_columns])
    callbacks = []
    fit_kwargs: dict[str, object] = {}

    if model_name == "lightgbm_ranker":
        model = LGBMRanker(**_lightgbm_params(model_name, random_state, model_params))
        fit_kwargs["group"] = _date_group_sizes(fit)
        fit_y = fit[label_column].astype(int)
        if not valid.empty:
            x_valid = imputer.transform(valid[feature_columns])
            fit_kwargs.update(
                {
                    "eval_X": x_valid,
                    "eval_y": valid[label_column].astype(int),
                    "eval_group": [_date_group_sizes(valid)],
                    "eval_at": [5, 10],
                }
            )
            callbacks = [early_stopping(50, verbose=False), log_evaluation(0)]
    else:
        model = LGBMRegressor(**_lightgbm_params(model_name, random_state, model_params))
        fit_y = fit[label_column]
        if not valid.empty:
            x_valid = imputer.transform(valid[feature_columns])
            fit_kwargs.update(
                {
                    "eval_X": x_valid,
                    "eval_y": valid[label_column],
                    "eval_metric": "l2",
                }
            )
            callbacks = [early_stopping(50, verbose=False), log_evaluation(0)]

    if callbacks:
        fit_kwargs["callbacks"] = callbacks
    model.fit(x_fit, fit_y, **fit_kwargs)
    predictions = model.predict(x_test)
    return {"imputer": imputer, "model": model, "model_name": model_name}, np.asarray(predictions)


def _prediction_metrics(
    frame: pd.DataFrame,
    label_column: str,
    predictions: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    raw_label = f"future_return_{horizon}d"
    metrics = {
        "mae": float(mean_absolute_error(frame[label_column], predictions)),
        "r2": float(r2_score(frame[label_column], predictions)),
        "directional_accuracy": float(
            (np.sign(frame[raw_label].to_numpy()) == np.sign(predictions)).mean()
        )
        if raw_label in frame.columns
        else float("nan"),
    }
    if raw_label not in frame.columns:
        return metrics

    scored = frame[["date", raw_label]].copy()
    scored["prediction"] = predictions
    daily_ic = []
    for _, group in scored.groupby("date"):
        clean = group[["prediction", raw_label]].dropna()
        if len(clean) >= 2 and clean["prediction"].nunique() > 1 and clean[raw_label].nunique() > 1:
            daily_ic.append(clean["prediction"].corr(clean[raw_label], method="spearman"))
    if daily_ic:
        ic = pd.Series(daily_ic, dtype=float).dropna()
        metrics.update(
            {
                "prediction_mean_ic": float(ic.mean()),
                "prediction_ic_ir": float(ic.mean() / ic.std()) if ic.std() else 0.0,
                "prediction_positive_ic_rate": float((ic > 0).mean()),
            }
        )
    return metrics


def train_model(
    factors: pd.DataFrame,
    model_path: str | Path,
    horizon: int = 5,
    test_size: float = 0.25,
    model_name: str = "random_forest",
    label_transform: str = "return",
    model_params: dict[str, float | int | str] | None = None,
) -> dict[str, float | str]:
    model_name = normalize_model_name(model_name)
    if model_name == "lightgbm_ranker" and label_transform == "return":
        label_transform = "quantile"
    training, feature_columns, label_column = make_training_frame(
        factors,
        horizon=horizon,
        label_transform=label_transform,
    )
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

    pipeline, predictions = fit_predict_model(
        train,
        test,
        feature_columns,
        label_column,
        model_name=model_name,
        model_params=model_params,
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "label_column": label_column,
            "split_date": str(pd.Timestamp(split_date).date()),
            "model_name": model_name,
            "label_transform": label_transform,
        },
        model_path,
    )

    return {
        "model": model_name,
        "label_transform": label_transform,
        "train_rows": float(len(train)),
        "test_rows": float(len(test)),
        **_prediction_metrics(test, label_column, predictions, horizon=horizon),
    }


def generate_ml_predictions(
    factors: pd.DataFrame,
    horizon: int = 5,
    train_size: int = 252,
    test_size: int = 63,
    embargo: int = 5,
    feature_lag: int = 1,
    prediction_column: str | None = None,
    model_name: str = "random_forest",
    label_transform: str = "return",
    model_params: dict[str, float | int | str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Generate out-of-sample walk-forward ML predictions and append as an alpha column."""
    model_name = normalize_model_name(model_name)
    if model_name == "lightgbm_ranker" and label_transform == "return":
        label_transform = "quantile"
    prediction_column = prediction_column or f"ml_prediction_{horizon}d"
    training, feature_columns, label_column = make_training_frame(
        factors,
        horizon=horizon,
        feature_lag=feature_lag,
        label_transform=label_transform,
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

        _, pred = fit_predict_model(
            train,
            test,
            feature_columns,
            label_column,
            model_name=model_name,
            random_state=42 + fold_index,
            model_params=model_params,
        )
        fold = test[["ticker", "date", label_column]].copy()
        raw_label = f"future_return_{horizon}d"
        if raw_label in test.columns and raw_label != label_column:
            fold[raw_label] = test[raw_label].to_numpy()
        fold[prediction_column] = pred
        fold["fold"] = fold_index
        predictions.append(fold)
        fold_metrics.append(
            {
                "fold": fold_index,
                "train_rows": float(len(train)),
                "test_rows": float(len(test)),
                **_prediction_metrics(test, label_column, pred, horizon=horizon),
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
        "model": model_name,
        "label_transform": label_transform,
        "folds": float(len(metrics_frame)),
        "prediction_rows": float(prediction_frame[prediction_column].notna().sum()),
        "mae": float(metrics_frame["mae"].mean()),
        "r2": float(metrics_frame["r2"].mean()),
        "directional_accuracy": float(metrics_frame["directional_accuracy"].mean()),
        "prediction_mean_ic": float(metrics_frame["prediction_mean_ic"].mean())
        if "prediction_mean_ic" in metrics_frame
        else float("nan"),
        "prediction_ic_ir": float(metrics_frame["prediction_ic_ir"].mean())
        if "prediction_ic_ir" in metrics_frame
        else float("nan"),
        "prediction_positive_ic_rate": float(metrics_frame["prediction_positive_ic_rate"].mean())
        if "prediction_positive_ic_rate" in metrics_frame
        else float("nan"),
        "train_size": float(train_size),
        "test_size": float(test_size),
        "embargo": float(embargo),
    }
    return result, metrics


def tune_lightgbm(
    factors: pd.DataFrame,
    horizon: int = 5,
    train_size: int = 252,
    test_size: int = 63,
    embargo: int = 5,
    max_trials: int = 24,
) -> pd.DataFrame:
    """Run a compact LightGBM search scored by out-of-sample cross-sectional IC."""
    grid = []
    for num_leaves, min_child_samples, learning_rate, fraction in product(
        (15, 31, 63),
        (100, 300),
        (0.01, 0.03),
        (0.7, 0.9),
    ):
        grid.extend(
            [
                ("lightgbm", "zscore", num_leaves, min_child_samples, learning_rate, fraction),
                ("lightgbm", "quantile", num_leaves, min_child_samples, learning_rate, fraction),
                (
                    "lightgbm_ranker",
                    "quantile",
                    num_leaves,
                    min_child_samples,
                    learning_rate,
                    fraction,
                ),
            ]
        )
    rows = []
    for trial, (model_name, label_transform, num_leaves, min_child_samples, learning_rate, fraction) in enumerate(
        grid[:max_trials],
        start=1,
    ):
        if model_name == "lightgbm_ranker" and label_transform != "quantile":
            continue
        params = {
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "learning_rate": learning_rate,
            "subsample": fraction,
            "colsample_bytree": fraction,
        }
        _, metrics = generate_ml_predictions(
            factors,
            horizon=horizon,
            train_size=train_size,
            test_size=test_size,
            embargo=embargo,
            model_name=model_name,
            label_transform=label_transform,
            model_params=params,
        )
        score = float(metrics.get("prediction_mean_ic", 0.0)) + 0.25 * float(
            metrics.get("prediction_ic_ir", 0.0)
        )
        rows.append(
            {
                "trial": trial,
                "score": score,
                **params,
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
