from __future__ import annotations

import pandas as pd

from us_alpha_lab.factors import FACTOR_COLUMNS
from us_alpha_lab.labels import add_forward_return_label


def daily_factor_ic(data: pd.DataFrame, factor: str, label: str) -> pd.Series:
    """Compute daily cross-sectional Spearman IC for one factor."""
    rows = []
    for date, group in data.groupby("date"):
        clean = group[[factor, label]].dropna()
        if len(clean) < 2 or clean[factor].nunique() < 2 or clean[label].nunique() < 2:
            continue
        rows.append((date, clean[factor].corr(clean[label], method="spearman")))

    if not rows:
        return pd.Series(dtype=float, name=factor)

    dates, values = zip(*rows)
    return pd.Series(values, index=pd.to_datetime(dates), name=factor).sort_index()


def factor_ic_report(factors: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Compute simple cross-sectional IC stats for each factor."""
    data = add_forward_return_label(factors, horizon=horizon)
    label = f"future_return_{horizon}d"
    rows = []

    for factor in FACTOR_COLUMNS:
        if factor not in data.columns:
            continue

        daily_ic = daily_factor_ic(data, factor=factor, label=label).dropna()
        clean = data[[factor, label]].dropna()
        if daily_ic.empty or clean.empty:
            continue
        overall_spearman = (
            clean[factor].corr(clean[label], method="spearman")
            if clean[factor].nunique() > 1 and clean[label].nunique() > 1
            else 0.0
        )

        rows.append(
            {
                "factor": factor,
                "mean_ic": daily_ic.mean(),
                "ic_std": daily_ic.std(),
                "ic_ir": daily_ic.mean() / daily_ic.std() if daily_ic.std() else 0.0,
                "positive_ic_rate": (daily_ic > 0).mean(),
                "overall_spearman": overall_spearman,
                "days": len(daily_ic),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)


def quantile_return_report(
    factors: pd.DataFrame,
    factor: str,
    horizon: int = 5,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Compute average forward return by daily factor quantile."""
    data = add_forward_return_label(factors, horizon=horizon)
    label = f"future_return_{horizon}d"
    rows = []

    for date, group in data.groupby("date"):
        clean = group[["ticker", factor, label]].dropna()
        if clean[factor].nunique() < 2 or len(clean) < quantiles:
            continue

        clean = clean.copy()
        clean["quantile"] = pd.qcut(
            clean[factor].rank(method="first"),
            q=quantiles,
            labels=False,
            duplicates="drop",
        )
        clean["quantile"] = clean["quantile"].astype(int) + 1
        rows.append(clean[["ticker", "quantile", label]].assign(date=date))

    if not rows:
        return pd.DataFrame(columns=["date", "quantile", "mean_return", "count"])

    stacked = pd.concat(rows, ignore_index=True)
    return (
        stacked.groupby(["date", "quantile"])[label]
        .agg(mean_return="mean", count="size")
        .reset_index()
    )
