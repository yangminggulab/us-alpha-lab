from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_alpha_lab.analysis import daily_factor_ic
from us_alpha_lab.kline_tokens import STATE_COLUMNS, make_kline_states
from us_alpha_lab.labels import add_forward_return_label
from us_alpha_lab.verdict import orthogonalized_ic_series

RETURN_STATES = ("strong_down", "down", "flat", "up", "strong_up")
RANGE_STATES = ("quiet", "normal", "wide", "shock")
VOLUME_STATES = ("low", "normal", "high", "extreme")
CLOSE_LOCATION_STATES = ("near_low", "middle", "near_high")

DEFAULT_DISCOVERY_WINDOWS = (20, 60)
DEFAULT_ORTHOGONAL_POOL = (
    "alpha_range_compression_21d",
    "alpha_gap_pressure_21d",
    "alpha_intraday_quality_21d",
)


@dataclass(frozen=True)
class KlinePatternSpec:
    name: str
    family: str
    description: str
    current: tuple[tuple[str, str], ...] = ()
    previous: tuple[tuple[str, str], ...] = ()


def _slug(value: str) -> str:
    return (
        value.replace("|", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


def _ensure_kline_states(frame: pd.DataFrame) -> pd.DataFrame:
    if all(column in frame.columns for column in STATE_COLUMNS):
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        return data.sort_values(["ticker", "date"]).reset_index(drop=True)
    return make_kline_states(frame)


def _condition_name(column: str, value: str) -> str:
    return f"{column.removeprefix('kline_').removesuffix('_state')}_{value}"


def _single_state_specs() -> list[KlinePatternSpec]:
    columns = {
        "kline_return_state": RETURN_STATES,
        "kline_range_state": RANGE_STATES,
        "kline_volume_state": VOLUME_STATES,
        "kline_close_location_state": CLOSE_LOCATION_STATES,
    }
    specs = []
    for column, values in columns.items():
        for value in values:
            name = _condition_name(column, value)
            specs.append(
                KlinePatternSpec(
                    name=name,
                    family="single_state",
                    description=f"窗口内 {name} 状态出现频率",
                    current=((column, value),),
                )
            )
    return specs


def _return_transition_specs() -> list[KlinePatternSpec]:
    specs = []
    for previous in RETURN_STATES:
        for current in RETURN_STATES:
            name = f"return_transition_{previous}_to_{current}"
            specs.append(
                KlinePatternSpec(
                    name=name,
                    family="return_transition",
                    description=f"窗口内日收益状态从 {previous} 转为 {current} 的频率",
                    previous=(("kline_return_state", previous),),
                    current=(("kline_return_state", current),),
                )
            )
    return specs


def _composite_state_specs() -> list[KlinePatternSpec]:
    pair_groups = [
        ("return_volume", "kline_return_state", RETURN_STATES, "kline_volume_state", VOLUME_STATES),
        (
            "return_close_location",
            "kline_return_state",
            RETURN_STATES,
            "kline_close_location_state",
            CLOSE_LOCATION_STATES,
        ),
        (
            "range_close_location",
            "kline_range_state",
            RANGE_STATES,
            "kline_close_location_state",
            CLOSE_LOCATION_STATES,
        ),
        ("range_volume", "kline_range_state", RANGE_STATES, "kline_volume_state", VOLUME_STATES),
    ]
    specs = []
    for family, left_column, left_values, right_column, right_values in pair_groups:
        for left in left_values:
            for right in right_values:
                name = f"{family}_{left}_{right}"
                specs.append(
                    KlinePatternSpec(
                        name=name,
                        family=family,
                        description=f"窗口内 {left_column}={left} 且 {right_column}={right} 的频率",
                        current=((left_column, left), (right_column, right)),
                    )
                )
    return specs


def _observed_token_specs(data: pd.DataFrame, max_token_patterns: int) -> list[KlinePatternSpec]:
    if "kline_token" not in data.columns or max_token_patterns <= 0:
        return []
    tokens = (
        data.loc[~data["kline_token"].astype(str).str.contains("missing"), "kline_token"]
        .value_counts()
        .head(max_token_patterns)
        .index
    )
    return [
        KlinePatternSpec(
            name=f"token_{_slug(str(token))}",
            family="observed_token",
            description=f"窗口内完整 K 线 token {token} 出现频率",
            current=(("kline_token", str(token)),),
        )
        for token in tokens
    ]


def build_kline_pattern_specs(
    data: pd.DataFrame,
    max_token_patterns: int = 24,
) -> list[KlinePatternSpec]:
    """Generate interpretable K-line pattern specs for automatic discovery."""
    return [
        *_single_state_specs(),
        *_return_transition_specs(),
        *_composite_state_specs(),
        *_observed_token_specs(data, max_token_patterns=max_token_patterns),
    ]


def _event_series(data: pd.DataFrame, spec: KlinePatternSpec) -> pd.Series:
    event = pd.Series(True, index=data.index)
    valid = pd.Series(True, index=data.index)
    grouped = data.groupby("ticker", sort=False)

    for column, value in spec.current:
        current = data[column].astype(str)
        event &= current.eq(value)
        valid &= current.ne("missing")

    for column, value in spec.previous:
        previous = grouped[column].shift(1).astype(str)
        event &= previous.eq(value)
        valid &= previous.ne("missing") & previous.ne("nan")

    return event.astype(float).where(valid)


def _rolling_share(data: pd.DataFrame, event: pd.Series, window: int) -> pd.Series:
    min_periods = max(5, window // 2)
    return (
        event.groupby(data["ticker"], sort=False)
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _candidate_profile(data: pd.DataFrame, values: pd.Series) -> dict[str, float]:
    clean_values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    daily_unique = data.assign(_candidate=clean_values).groupby("date")["_candidate"].nunique()
    return {
        "coverage": float(clean_values.notna().mean()),
        "active_days": float((daily_unique >= 5).sum()),
        "median_daily_unique": float(daily_unique.median()) if not daily_unique.empty else 0.0,
    }


def _ic_summary(daily_ic: pd.Series) -> dict[str, float | int]:
    if daily_ic.empty:
        return {
            "mean_ic": np.nan,
            "ic_ir": np.nan,
            "positive_ic_rate": np.nan,
            "days": 0,
        }
    ic_std = float(daily_ic.std())
    return {
        "mean_ic": float(daily_ic.mean()),
        "ic_ir": float(daily_ic.mean() / ic_std) if ic_std else 0.0,
        "positive_ic_rate": float((daily_ic > 0).mean()),
        "days": len(daily_ic),
    }


def _directional_rate(mean_ic: float, positive_ic_rate: float) -> float:
    if pd.isna(mean_ic) or pd.isna(positive_ic_rate):
        return np.nan
    return float(positive_ic_rate if mean_ic >= 0 else 1.0 - positive_ic_rate)


def _candidate_verdict(row: dict[str, float | int | str]) -> str:
    abs_orth_ic = abs(float(row.get("ic_orth", 0.0) or 0.0))
    orth_ir = abs(float(row.get("orth_ir", 0.0) or 0.0))
    directional = float(row.get("directional_ic_rate", 0.0) or 0.0)
    coverage = float(row.get("coverage", 0.0) or 0.0)
    if abs_orth_ic >= 0.015 and orth_ir > 0 and directional >= 0.55 and coverage >= 0.5:
        return "强增量候选"
    if abs_orth_ic >= 0.01 and orth_ir > 0 and directional >= 0.52 and coverage >= 0.35:
        return "增量候选"
    if abs_orth_ic >= 0.005:
        return "观察"
    return "剔除"


def build_kline_pattern_discovery(
    bars_or_kline: pd.DataFrame,
    base_factors: pd.DataFrame,
    horizon: int = 5,
    windows: Iterable[int] = DEFAULT_DISCOVERY_WINDOWS,
    orthogonal_pool: Iterable[str] = DEFAULT_ORTHOGONAL_POOL,
    max_token_patterns: int = 24,
    min_coverage: float = 0.2,
) -> pd.DataFrame:
    """Search K-line token pattern factors and rank them by incremental IC.

    Each candidate is the rolling share of an interpretable K-line event, such as
    a return-state transition or a return/volume composite state. The key metric
    is orthogonal IC against a small pool of public price-action factors.
    """
    data = _ensure_kline_states(bars_or_kline)
    windows_tuple = tuple(sorted({int(window) for window in windows}))
    if not windows_tuple:
        raise ValueError("At least one discovery window is required.")
    if any(window < 2 for window in windows_tuple):
        raise ValueError("Discovery windows must be at least 2.")

    specs = build_kline_pattern_specs(data, max_token_patterns=max_token_patterns)
    pool_columns = [column for column in orthogonal_pool if column in base_factors.columns]
    base_subset = base_factors[["ticker", "date", *pool_columns]].copy()

    scored = add_forward_return_label(
        data[["ticker", "date", "close"]].copy(),
        horizon=horizon,
    ).merge(base_subset, on=["ticker", "date"], how="left")
    label = f"future_return_{horizon}d"

    rows = []
    for spec in specs:
        event = _event_series(data, spec)
        for window in windows_tuple:
            candidate_name = f"alpha_kline_pattern_{_slug(spec.name)}_{window}d"
            candidate_values = _rolling_share(data, event, window=window)
            profile = _candidate_profile(scored, candidate_values)
            if profile["coverage"] < min_coverage:
                continue

            scored["_candidate"] = candidate_values.to_numpy()
            daily_ic = daily_factor_ic(scored, factor="_candidate", label=label).dropna()
            raw_stats = _ic_summary(daily_ic)
            orth_ic = (
                orthogonalized_ic_series(scored, "_candidate", pool_columns, label).dropna()
                if pool_columns
                else pd.Series(dtype=float)
            )
            orth_stats = _ic_summary(orth_ic)
            directional_ic_rate = _directional_rate(
                float(raw_stats["mean_ic"]),
                float(raw_stats["positive_ic_rate"]),
            )
            row = {
                "factor": candidate_name,
                "family": spec.family,
                "window": window,
                "pattern": spec.name,
                "description": spec.description,
                "formula": f"rolling_mean({spec.name}, {window})",
                **profile,
                **raw_stats,
                "ic_orth": orth_stats["mean_ic"],
                "orth_ir": orth_stats["ic_ir"],
                "orth_positive_ic_rate": orth_stats["positive_ic_rate"],
                "orth_days": orth_stats["days"],
                "directional_ic_rate": directional_ic_rate,
            }
            row["abs_ic_orth"] = abs(float(row["ic_orth"])) if pd.notna(row["ic_orth"]) else np.nan
            row["discovery_score"] = (
                float(row["abs_ic_orth"] if pd.notna(row["abs_ic_orth"]) else 0.0)
                + abs(float(row["orth_ir"] if pd.notna(row["orth_ir"]) else 0.0)) * 0.05
                + abs(float(row["mean_ic"] if pd.notna(row["mean_ic"]) else 0.0)) * 0.25
                + max(float(row["directional_ic_rate"] if pd.notna(row["directional_ic_rate"]) else 0.5) - 0.5, 0.0)
                * 0.05
            )
            row["verdict"] = _candidate_verdict(row)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["discovery_score", "abs_ic_orth"], ascending=[False, False])
        .reset_index(drop=True)
    )
