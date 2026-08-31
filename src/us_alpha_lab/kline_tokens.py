from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from math import log

import numpy as np
import pandas as pd

BASE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]
DEFAULT_WINDOWS = (20, 60)
MAX_TOKEN_STATES = 5 * 4 * 4 * 3

STATE_COLUMNS = [
    "kline_return_state",
    "kline_range_state",
    "kline_volume_state",
    "kline_close_location_state",
    "kline_token",
]


def kline_factor_columns(windows: Iterable[int] = DEFAULT_WINDOWS) -> list[str]:
    columns: list[str] = []
    for window in windows:
        columns.extend(
            [
                f"alpha_kline_state_entropy_{window}d",
                f"alpha_kline_bull_volume_ngram_{window}d",
                f"alpha_kline_reversal_transition_{window}d",
                f"alpha_kline_path_similarity_{window}d",
            ]
        )
    return columns


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _as_state(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    bucketed = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
    return bucketed.astype("object").where(bucketed.notna(), "missing").astype(str)


def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan) - 1


def make_kline_states(bars: pd.DataFrame, volume_window: int = 21) -> pd.DataFrame:
    """Convert OHLCV bars into compact discrete K-line states and tokens."""
    _require_columns(bars, BASE_COLUMNS)
    data = bars.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = data.groupby("ticker", group_keys=False)
    previous_close = grouped["close"].shift(1)
    ret_1d = _safe_pct(data["close"], previous_close)
    range_pct = (data["high"] - data["low"]) / previous_close.replace(0, np.nan)

    volume_mean = grouped["volume"].rolling(volume_window).mean().reset_index(level=0, drop=True)
    volume_std = grouped["volume"].rolling(volume_window).std().reset_index(level=0, drop=True)
    volume_z = (data["volume"] - volume_mean) / volume_std.replace(0, np.nan)

    day_range = (data["high"] - data["low"]).replace(0, np.nan)
    close_location = ((data["close"] - data["low"]) / day_range).clip(lower=0, upper=1)

    data["kline_return_state"] = _as_state(
        ret_1d,
        bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
        labels=["strong_down", "down", "flat", "up", "strong_up"],
    )
    data["kline_range_state"] = _as_state(
        range_pct,
        bins=[-np.inf, 0.01, 0.02, 0.04, np.inf],
        labels=["quiet", "normal", "wide", "shock"],
    )
    data["kline_volume_state"] = _as_state(
        volume_z,
        bins=[-np.inf, -1.0, 1.0, 2.0, np.inf],
        labels=["low", "normal", "high", "extreme"],
    )
    data["kline_close_location_state"] = _as_state(
        close_location,
        bins=[-np.inf, 0.25, 0.75, np.inf],
        labels=["near_low", "middle", "near_high"],
    )
    data["kline_token"] = (
        data["kline_return_state"]
        + "|"
        + data["kline_range_state"]
        + "|"
        + data["kline_volume_state"]
        + "|"
        + data["kline_close_location_state"]
    )
    return data


def _normalized_entropy(tokens: list[str]) -> float:
    clean = [token for token in tokens if "missing" not in token]
    if len(clean) < 2:
        return np.nan
    counts = Counter(clean)
    entropy = -sum((count / len(clean)) * log(count / len(clean)) for count in counts.values())
    denominator = log(min(len(clean), MAX_TOKEN_STATES))
    return float(entropy / denominator) if denominator > 0 else 0.0


def _bull_volume_share(return_states: list[str], volume_states: list[str]) -> float:
    if not return_states:
        return np.nan
    hits = sum(
        return_state in {"up", "strong_up"} and volume_state in {"high", "extreme"}
        for return_state, volume_state in zip(return_states, volume_states, strict=True)
    )
    return float(hits / len(return_states))


def _reversal_transition_rate(return_states: list[str]) -> float:
    if len(return_states) < 2:
        return np.nan
    hits = sum(
        previous in {"down", "strong_down"} and current in {"up", "strong_up"}
        for previous, current in pairwise(return_states)
    )
    return float(hits / (len(return_states) - 1))


def _path_similarity(tokens: list[str], current_token: str) -> float:
    history = [token for token in tokens if "missing" not in token]
    if not history or "missing" in current_token:
        return np.nan
    return float(history.count(current_token) / len(history))


def _sequence_features_for_group(group: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    output = pd.DataFrame(index=group.index)
    tokens = group["kline_token"].tolist()
    return_states = group["kline_return_state"].tolist()
    volume_states = group["kline_volume_state"].tolist()

    for window in windows:
        entropy_values: list[float] = []
        bull_volume_values: list[float] = []
        reversal_values: list[float] = []
        similarity_values: list[float] = []
        for index, current_token in enumerate(tokens):
            if index + 1 < window:
                entropy_values.append(np.nan)
                bull_volume_values.append(np.nan)
                reversal_values.append(np.nan)
                similarity_values.append(np.nan)
                continue

            start = index - window + 1
            token_window = tokens[start : index + 1]
            return_window = return_states[start : index + 1]
            volume_window = volume_states[start : index + 1]
            token_history = tokens[start:index]

            entropy_values.append(_normalized_entropy(token_window))
            bull_volume_values.append(_bull_volume_share(return_window, volume_window))
            reversal_values.append(_reversal_transition_rate(return_window))
            similarity_values.append(_path_similarity(token_history, current_token))

        output[f"alpha_kline_state_entropy_{window}d"] = entropy_values
        output[f"alpha_kline_bull_volume_ngram_{window}d"] = bull_volume_values
        output[f"alpha_kline_reversal_transition_{window}d"] = reversal_values
        output[f"alpha_kline_path_similarity_{window}d"] = similarity_values

    return output


def add_kline_sequence_factors(
    bars: pd.DataFrame,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    volume_window: int = 21,
) -> pd.DataFrame:
    """Add experimental K-line token sequence factors without touching the main registry."""
    windows_tuple = tuple(sorted({int(window) for window in windows}))
    if not windows_tuple:
        raise ValueError("At least one K-line factor window is required.")
    if any(window < 2 for window in windows_tuple):
        raise ValueError("K-line factor windows must be at least 2.")

    data = make_kline_states(bars, volume_window=volume_window)
    factor_frames = [
        _sequence_features_for_group(group, windows_tuple)
        for _, group in data.groupby("ticker", sort=False)
    ]
    if factor_frames:
        factors = pd.concat(factor_frames).sort_index()
        data[factors.columns] = factors
    return data
