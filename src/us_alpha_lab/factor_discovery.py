from __future__ import annotations

import numpy as np
import pandas as pd

from us_alpha_lab.alpha_registry import alpha_registry_frame, enabled_alpha_names
from us_alpha_lab.analysis import factor_ic_report


def _factor_profile(factors: pd.DataFrame, factor: str) -> dict[str, float | str]:
    values = pd.to_numeric(factors[factor], errors="coerce").replace([np.inf, -np.inf], np.nan)
    daily_unique = factors.assign(_factor_value=values).groupby("date")["_factor_value"].nunique()
    return {
        "factor": factor,
        "coverage": float(values.notna().mean()),
        "active_days": float((daily_unique >= 5).sum()),
        "median_daily_unique": float(daily_unique.median()) if not daily_unique.empty else 0.0,
        "mean_abs_value": float(values.abs().mean()) if values.notna().any() else 0.0,
    }


def build_factor_discovery_report(
    factors: pd.DataFrame,
    horizon: int = 5,
    factor_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a fast candidate-factor report before full portfolio backtests."""
    factor_columns = factor_columns or [factor for factor in enabled_alpha_names() if factor in factors.columns]
    profiles = pd.DataFrame([_factor_profile(factors, factor) for factor in factor_columns])
    registry = alpha_registry_frame(enabled_only=False)
    ic = factor_ic_report(factors, horizon=horizon)

    report = registry.merge(profiles, left_on="name", right_on="factor", how="inner")
    report = report.merge(ic, on="factor", how="left")
    report["discovery_score"] = (
        report["mean_ic"].fillna(0).abs()
        + report["ic_ir"].fillna(0).abs() * 0.25
        + (report["positive_ic_rate"].fillna(0) - 0.5).abs() * 0.1
    )
    return report.sort_values("discovery_score", ascending=False).reset_index(drop=True)
