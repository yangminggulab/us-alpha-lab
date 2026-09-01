from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]
OBSERVATION_COLUMNS = [
    "latent_return_z",
    "latent_gap_z",
    "latent_intraday_z",
    "latent_range_z",
    "latent_volume_z",
    "latent_close_location",
    "latent_trend_5d_z",
]


@dataclass(frozen=True)
class LatentParticipantState:
    name: str
    label: str
    description: str
    prototype: tuple[float, ...]


LATENT_STATES = (
    LatentParticipantState(
        name="accumulation",
        label="机构吸筹",
        description="放量、收盘靠高位、日内走势偏强，近几日趋势温和延续。",
        prototype=(0.45, 0.10, 0.85, 0.15, 1.10, 0.85, 0.45),
    ),
    LatentParticipantState(
        name="distribution",
        label="机构派发",
        description="放量、收盘靠低位、日内走势偏弱，价格上行动能开始被卖压吸收。",
        prototype=(-0.45, -0.10, -0.85, 0.20, 1.10, -0.85, -0.25),
    ),
    LatentParticipantState(
        name="forced_selling",
        label="被迫卖出",
        description="高成交量、高振幅、跳空或单日下跌明显，收盘靠低位。",
        prototype=(-1.25, -0.75, -0.85, 1.25, 1.60, -0.95, -0.85),
    ),
    LatentParticipantState(
        name="market_making_pressure",
        label="做市库存压力",
        description="成交和振幅偏高但方向不明显，收盘接近区间中部，像流动性提供者在吸收冲击。",
        prototype=(0.00, 0.00, 0.00, 1.00, 0.80, 0.00, 0.00),
    ),
    LatentParticipantState(
        name="arbitrage_repair",
        label="套利修复",
        description="隔夜跳空后日内出现反向修复，价格冲击被快速消化。",
        prototype=(0.15, -0.90, 0.95, 0.75, 0.70, 0.70, 0.10),
    ),
    LatentParticipantState(
        name="noise",
        label="噪声交易",
        description="成交、振幅和价格方向都不突出，缺少明确参与者痕迹。",
        prototype=(0.00, 0.00, 0.00, -0.45, -0.45, 0.00, 0.00),
    ),
)


def latent_probability_columns(states: Iterable[LatentParticipantState] = LATENT_STATES) -> list[str]:
    return [f"latent_{state.name}_prob" for state in states]


def latent_factor_columns(states: Iterable[LatentParticipantState] = LATENT_STATES) -> list[str]:
    return [f"alpha_latent_{state.name}_prob" for state in states]


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan) - 1


def _rolling_zscore_by_ticker(tickers: pd.Series, values: pd.Series, window: int) -> pd.Series:
    frame = pd.DataFrame({"ticker": tickers.to_numpy(), "value": values.to_numpy()})
    min_periods = max(5, window // 3)
    grouped = frame.groupby("ticker", sort=False)["value"]
    mean = grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).mean())
    std = grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).std())
    return ((frame["value"] - mean) / std.replace(0, np.nan)).clip(-4, 4)


def make_latent_observations(bars: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Build low-frequency observable proxies for participant-state inference."""
    _require_columns(bars, BASE_COLUMNS)
    data = bars.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = data.groupby("ticker", group_keys=False)

    previous_close = grouped["close"].shift(1)
    ret_1d = _safe_pct(data["close"], previous_close)
    gap = _safe_pct(data["open"], previous_close)
    intraday = _safe_pct(data["close"], data["open"])
    range_pct = (data["high"] - data["low"]) / previous_close.replace(0, np.nan)
    trend_5d = grouped["close"].pct_change(5)
    volume_log = np.log1p(data["volume"])

    day_range = (data["high"] - data["low"]).replace(0, np.nan)
    close_location = ((data["close"] - data["low"]) / day_range).clip(0, 1) * 2 - 1

    data["latent_return_z"] = _rolling_zscore_by_ticker(data["ticker"], ret_1d, lookback).to_numpy()
    data["latent_gap_z"] = _rolling_zscore_by_ticker(data["ticker"], gap, lookback).to_numpy()
    data["latent_intraday_z"] = _rolling_zscore_by_ticker(
        data["ticker"],
        intraday,
        lookback,
    ).to_numpy()
    data["latent_range_z"] = _rolling_zscore_by_ticker(
        data["ticker"],
        range_pct,
        lookback,
    ).to_numpy()
    data["latent_volume_z"] = _rolling_zscore_by_ticker(
        data["ticker"],
        pd.Series(volume_log),
        lookback,
    ).to_numpy()
    data["latent_close_location"] = close_location
    data["latent_trend_5d_z"] = _rolling_zscore_by_ticker(
        data["ticker"],
        trend_5d,
        lookback,
    ).to_numpy()
    return data


def _transition_matrix(n_states: int, persistence: float) -> np.ndarray:
    if n_states < 2:
        return np.ones((n_states, n_states))
    off_diagonal = (1.0 - persistence) / (n_states - 1)
    matrix = np.full((n_states, n_states), off_diagonal)
    np.fill_diagonal(matrix, persistence)
    return matrix


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.nanmax(logits)
    weights = np.exp(shifted)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(len(logits), 1.0 / len(logits))
    return weights / total


def _emission_probabilities(features: np.ndarray, prototypes: np.ndarray, emission_scale: float) -> np.ndarray:
    if np.isnan(features).all():
        return np.full(len(prototypes), 1.0 / len(prototypes))
    observed = np.nan_to_num(features, nan=0.0)
    distances = ((prototypes - observed) ** 2).sum(axis=1)
    return _softmax(-0.5 * distances / (emission_scale**2))


def _filter_group(
    group: pd.DataFrame,
    prototypes: np.ndarray,
    transition: np.ndarray,
    emission_scale: float,
) -> np.ndarray:
    posterior = np.full(len(prototypes), 1.0 / len(prototypes))
    rows = []
    for features in group[OBSERVATION_COLUMNS].to_numpy(dtype=float):
        emission = _emission_probabilities(features, prototypes, emission_scale=emission_scale)
        prior = posterior @ transition
        posterior = prior * emission
        posterior_sum = posterior.sum()
        if posterior_sum <= 0 or not np.isfinite(posterior_sum):
            posterior = np.full(len(prototypes), 1.0 / len(prototypes))
        else:
            posterior = posterior / posterior_sum
        rows.append(posterior.copy())
    return np.vstack(rows)


def add_latent_participant_states(
    bars: pd.DataFrame,
    lookback: int = 21,
    persistence: float = 0.86,
    emission_scale: float = 1.25,
) -> pd.DataFrame:
    """Infer low-frequency hidden participant-state probabilities from daily OHLCV.

    This is a deliberately lightweight HMM-style filter. State prototypes are
    interpretable research hypotheses, not learned identity labels.
    """
    if lookback < 5:
        raise ValueError("lookback must be at least 5.")
    if not 0 <= persistence < 1:
        raise ValueError("persistence must be in [0, 1).")
    if emission_scale <= 0:
        raise ValueError("emission_scale must be positive.")

    data = make_latent_observations(bars, lookback=lookback)
    prototypes = np.array([state.prototype for state in LATENT_STATES], dtype=float)
    transition = _transition_matrix(len(LATENT_STATES), persistence=persistence)
    probability_columns = latent_probability_columns()

    probability_frames = []
    for _, group in data.groupby("ticker", sort=False):
        probabilities = _filter_group(
            group,
            prototypes=prototypes,
            transition=transition,
            emission_scale=emission_scale,
        )
        probability_frames.append(pd.DataFrame(probabilities, index=group.index, columns=probability_columns))

    if probability_frames:
        probabilities = pd.concat(probability_frames).sort_index()
        data[probability_columns] = probabilities

    state_names = [state.name for state in LATENT_STATES]
    probability_values = data[probability_columns].to_numpy(dtype=float)
    dominant_indices = np.nanargmax(probability_values, axis=1)
    data["latent_dominant_state"] = [state_names[index] for index in dominant_indices]
    data["latent_state_confidence"] = np.nanmax(probability_values, axis=1)

    for probability_column, factor_column in zip(
        latent_probability_columns(),
        latent_factor_columns(),
        strict=True,
    ):
        data[factor_column] = data[probability_column]
    return data
