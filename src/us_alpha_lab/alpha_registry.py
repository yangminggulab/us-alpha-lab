from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class AlphaSpec:
    name: str
    category: str
    formula: str
    source: str
    expected_direction: int
    lookback: int
    description: str
    enabled: bool = True


ALPHA_REGISTRY: tuple[AlphaSpec, ...] = (
    AlphaSpec(
        name="reversal_1d",
        category="price_reversal",
        formula="-pct_change(close, 1)",
        source="local_baseline",
        expected_direction=1,
        lookback=1,
        description="One-day short-term reversal signal.",
    ),
    AlphaSpec(
        name="momentum_5d",
        category="price_momentum",
        formula="pct_change(close, 5)",
        source="local_baseline",
        expected_direction=1,
        lookback=5,
        description="Five-day close-to-close momentum.",
    ),
    AlphaSpec(
        name="momentum_21d",
        category="price_momentum",
        formula="pct_change(close, 21)",
        source="local_baseline",
        expected_direction=1,
        lookback=21,
        description="One-month close-to-close momentum.",
    ),
    AlphaSpec(
        name="momentum_63d",
        category="price_momentum",
        formula="pct_change(close, 63)",
        source="local_baseline",
        expected_direction=1,
        lookback=63,
        description="Three-month close-to-close momentum.",
    ),
    AlphaSpec(
        name="volatility_21d",
        category="risk",
        formula="std(pct_change(close, 1), 21)",
        source="local_baseline",
        expected_direction=1,
        lookback=21,
        description="Twenty-one day realized volatility.",
    ),
    AlphaSpec(
        name="volume_z_21d",
        category="volume",
        formula="(volume - mean(volume, 21)) / std(volume, 21)",
        source="local_baseline",
        expected_direction=1,
        lookback=21,
        description="Twenty-one day volume z-score.",
    ),
    AlphaSpec(
        name="dollar_volume",
        category="liquidity",
        formula="close * volume",
        source="local_baseline",
        expected_direction=1,
        lookback=1,
        description="Dollar volume proxy for liquidity and institutional attention.",
    ),
    AlphaSpec(
        name="close_to_high",
        category="intraday_position",
        formula="(high - close) / (high - low)",
        source="local_baseline",
        expected_direction=1,
        lookback=1,
        description="Close distance from daily high, normalized by range.",
    ),
    AlphaSpec(
        name="close_to_low",
        category="intraday_position",
        formula="(close - low) / (high - low)",
        source="local_baseline",
        expected_direction=1,
        lookback=1,
        description="Close distance from daily low, normalized by range.",
    ),
    AlphaSpec(
        name="vwap_gap",
        category="intraday_position",
        formula="close / vwap - 1",
        source="local_baseline",
        expected_direction=1,
        lookback=1,
        description="Close premium or discount versus daily VWAP.",
    ),
    AlphaSpec(
        name="alpha_open_volume_corr_10",
        category="formula_alpha",
        formula="-corr(rank(open), rank(volume), 10)",
        source="alpha101_inspired",
        expected_direction=1,
        lookback=10,
        description="Negative rolling correlation between cross-sectional open rank and volume rank.",
    ),
    AlphaSpec(
        name="alpha_vwap_reversion_10",
        category="formula_alpha",
        formula="-rank(open - decay_linear(vwap, 10))",
        source="alpha101_inspired",
        expected_direction=1,
        lookback=10,
        description="VWAP reversion signal using decayed VWAP anchor.",
    ),
    AlphaSpec(
        name="alpha_range_ts_rank_9",
        category="formula_alpha",
        formula="-ts_rank(delta(high - low, 1), 9)",
        source="alpha101_inspired",
        expected_direction=1,
        lookback=9,
        description="Time-series rank of daily range expansion, reversed.",
    ),
    AlphaSpec(
        name="alpha_overnight_gap_reversal_5",
        category="formula_alpha",
        formula="-ts_rank(open / delay(close, 1) - 1, 5)",
        source="alpha101_inspired",
        expected_direction=1,
        lookback=5,
        description="Short-term reversal after overnight gaps.",
    ),
    AlphaSpec(
        name="alpha_intraday_strength_5",
        category="formula_alpha",
        formula="ts_rank((close - open) / open, 5)",
        source="local_formula",
        expected_direction=1,
        lookback=5,
        description="Recent persistence of intraday close-open strength.",
    ),
    AlphaSpec(
        name="alpha_volume_price_divergence_21",
        category="formula_alpha",
        formula="-corr(rank(close), rank(volume), 21)",
        source="alpha101_inspired",
        expected_direction=1,
        lookback=21,
        description="Negative rolling correlation between price rank and volume rank.",
    ),
    AlphaSpec(
        name="alpha_low_volatility_21d",
        category="risk",
        formula="-std(pct_change(close, 1), 21)",
        source="local_formula",
        expected_direction=1,
        lookback=21,
        description="Low-volatility tilt expressed so higher values mean lower realized volatility.",
    ),
    AlphaSpec(
        name="alpha_price_to_21d_high",
        category="price_momentum",
        formula="close / rolling_max(high, 21) - 1",
        source="local_formula",
        expected_direction=1,
        lookback=21,
        description="Distance from the 21-day high, closer to high ranks higher.",
    ),
    AlphaSpec(
        name="alpha_ma_gap_21d",
        category="price_momentum",
        formula="close / rolling_mean(close, 21) - 1",
        source="local_formula",
        expected_direction=1,
        lookback=21,
        description="Close price premium or discount versus the 21-day moving average.",
    ),
    AlphaSpec(
        name="alpha_liquidity_quality_21d",
        category="liquidity",
        formula="-mean(abs(ret_1d) / dollar_volume, 21)",
        source="local_formula",
        expected_direction=1,
        lookback=21,
        description="Negative Amihud-style illiquidity proxy, higher means better liquidity quality.",
    ),
    AlphaSpec(
        name="alpha_close_position_5",
        category="intraday_position",
        formula="ts_rank((close - low) / (high - low), 5)",
        source="local_formula",
        expected_direction=1,
        lookback=5,
        description="Recent rank of close location within the daily range.",
    ),
    AlphaSpec(
        name="ml_prediction_5d",
        category="machine_learning",
        formula="walk_forward_random_forest(factors -> future_return_5d)",
        source="local_ml",
        expected_direction=1,
        lookback=252,
        description="Out-of-sample five-day return prediction from walk-forward random forest.",
    ),
)


def enabled_alpha_specs() -> list[AlphaSpec]:
    return [spec for spec in ALPHA_REGISTRY if spec.enabled]


def enabled_alpha_names() -> list[str]:
    return [spec.name for spec in enabled_alpha_specs()]


def get_alpha_spec(name: str) -> AlphaSpec:
    for spec in ALPHA_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown alpha: {name}")


def alpha_registry_frame(enabled_only: bool = True) -> pd.DataFrame:
    specs = enabled_alpha_specs() if enabled_only else list(ALPHA_REGISTRY)
    return pd.DataFrame([asdict(spec) for spec in specs])
