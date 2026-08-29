from __future__ import annotations

import math

import numpy as np
import pandas as pd

from us_alpha_lab.alpha_registry import enabled_alpha_names
from us_alpha_lab.analysis import daily_factor_ic, quantile_return_report
from us_alpha_lab.labels import add_forward_return_label

# 五关通过阈值
SIGNAL_ICIR = 0.5
SIGNAL_T = 2.0
MONOTONIC_SPEARMAN = 0.7
NET_SHARPE = 1.0
MAX_DRAWDOWN = -0.30
STABILITY_AGREEMENT = 0.80
MIN_YEARS = 2
INCREMENTAL_IC = 0.01
INCREMENTAL_IR = 0.3


def newey_west_t_stat(daily_ic: pd.Series, lag: int = 5) -> float:
    """IC 的 t 统计量，用 Newey–West（Bartlett 核）修正日度 IC 的自相关。

    重叠的前瞻收益使日度 IC 序列高度自相关，直接拿 ICIR×sqrt(N) 会高估显著性。
    """
    ic = daily_ic.dropna()
    n = len(ic)
    if n < 2:
        return 0.0

    mean = float(ic.mean())
    demeaned = ic - mean
    gamma0 = float((demeaned**2).sum()) / n
    autocovariances = [gamma0]
    for order in range(1, lag + 1):
        if order >= n:
            break
        autocovariances.append(float((demeaned.iloc[order:] * demeaned.iloc[:-order]).sum()) / n)

    bandwidth = len(autocovariances) - 1
    sigma2 = gamma0 + 2.0 * sum(
        (1.0 - order / (bandwidth + 1)) * autocovariances[order]
        for order in range(1, bandwidth + 1)
    )
    if sigma2 <= 0.0:
        sigma2 = gamma0

    standard_error = math.sqrt(sigma2 / n)
    if standard_error == 0.0:
        return 0.0
    return mean / standard_error


def quantile_monotonicity(
    factors: pd.DataFrame,
    factor: str,
    horizon: int = 5,
    quantiles: int = 5,
) -> float:
    """分位数收益与分位序的截面 Spearman，衡量单调性。

    ≈+1 单调上升、≈−1 单调下降、≈0 是两头翘（波动率型信号），不构成方向 alpha。
    """
    qr = quantile_return_report(factors, factor, horizon=horizon, quantiles=quantiles)
    if qr.empty:
        return 0.0
    aggregate = qr.groupby("quantile")["mean_return"].mean()
    if aggregate.nunique() < 2:
        return 0.0
    quantile_axis = pd.Series(aggregate.index.to_numpy(dtype=float))
    return float(pd.Series(aggregate.to_numpy(dtype=float)).corr(quantile_axis, method="spearman"))


def yearly_ic_sign_agreement(daily_ic: pd.Series, min_years: int = MIN_YEARS) -> tuple[float, int]:
    """分年 IC 与总体主导方向同号的比例。样本不足 min_years 年时返回 (0.0, n_years) 表示数据不足。"""
    if daily_ic.empty:
        return 0.0, 0
    yearly_mean = daily_ic.groupby(daily_ic.index.year).mean()
    n_years = len(yearly_mean)
    if n_years < min_years:
        return 0.0, n_years
    dominant = 1.0 if float(yearly_mean.mean()) >= 0.0 else -1.0
    agreement = float((yearly_mean * dominant > 0).mean())
    return agreement, n_years


def orthogonalized_ic_series(
    data: pd.DataFrame,
    factor: str,
    pool_columns: list[str],
    label: str,
    min_obs: int = 5,
) -> pd.Series:
    """日度截面正交化：把 factor 对池内因子回归取残差，残差对未来收益的 Spearman IC。

    IC_orth = corr(ε, future return)，ε 是因子去掉公共因子线性解释后的增量部分。
    """
    rows = []
    for date, group in data.groupby("date"):
        columns = [factor, *pool_columns, label]
        clean = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < min_obs:
            continue
        y = clean[factor].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(y)), clean[pool_columns].to_numpy(dtype=float)])
        try:
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = y - design @ coefficients
        target = clean[label].to_numpy(dtype=float)
        if np.ptp(residual) == 0 or np.ptp(target) == 0:
            continue
        ic = pd.Series(residual).corr(pd.Series(target), method="spearman")
        rows.append((date, float(ic)))

    if not rows:
        return pd.Series(dtype=float)
    dates, values = zip(*rows)
    return pd.Series(values, index=pd.to_datetime(dates), name=factor).sort_index()


def _verdict_label(passed: int, evaluated: int) -> str:
    if evaluated < 3:
        return "数据不足"
    if passed == evaluated == 5:
        return "可盈利候选"
    if passed >= 4:
        return "接近达标"
    if passed >= 3:
        return "部分达标"
    return "未达标"


def build_factor_verdict(
    factors: pd.DataFrame,
    ic_report: pd.DataFrame,
    leaderboard: pd.DataFrame,
    horizon: int = 5,
    quantiles: int = 5,
) -> pd.DataFrame:
    """汇总五关通过情况，返回逐因子判定表。

    五关：信号（ICIR + NW t 值）、单调（分位数收益单调）、净收益（扣费夏普 + 回撤）、
    稳定（分年 IC 同号率）、增量（对公共因子池正交化后的残差 IC）。
    缺回测或样本年限不足的关卡记为 None（不参与判定）。
    """
    data = add_forward_return_label(factors, horizon=horizon)
    label = f"future_return_{horizon}d"
    lb = leaderboard.set_index("factor") if not leaderboard.empty else pd.DataFrame()
    non_ml_pool = [
        name
        for name in enabled_alpha_names()
        if not name.startswith("ml_prediction") and name in factors.columns
    ]
    rows = []

    for _, row in ic_report.iterrows():
        factor = str(row["factor"])
        daily_ic = daily_factor_ic(data, factor=factor, label=label).dropna()
        ic_ir = float(row["ic_ir"])
        ic_t = newey_west_t_stat(daily_ic, lag=horizon)
        signal_ok = ic_ir >= SIGNAL_ICIR and abs(ic_t) >= SIGNAL_T

        monotonicity = quantile_monotonicity(factors, factor, horizon=horizon, quantiles=quantiles)
        monotonic_ok = abs(monotonicity) >= MONOTONIC_SPEARMAN

        net_sharpe: float | None = None
        max_drawdown: float | None = None
        net_return_ok: bool | None = None
        if factor in lb.index:
            net_sharpe = float(lb.loc[factor, "information_ratio"])
            max_drawdown = float(lb.loc[factor, "max_drawdown"])
            net_return_ok = net_sharpe >= NET_SHARPE and max_drawdown > MAX_DRAWDOWN

        agreement, n_years = yearly_ic_sign_agreement(daily_ic)
        stability_ok: bool | None = agreement >= STABILITY_AGREEMENT if n_years >= MIN_YEARS else None

        pool = [column for column in non_ml_pool if column != factor]
        if pool:
            ic_orth_series = orthogonalized_ic_series(data, factor, pool, label)
            if ic_orth_series.empty:
                ic_orth_mean: float | None = None
                ic_orth_ir: float | None = None
            else:
                ic_orth_mean = float(ic_orth_series.mean())
                ic_orth_std = float(ic_orth_series.std())
                ic_orth_ir = ic_orth_mean / ic_orth_std if ic_orth_std else None
        else:
            ic_orth_mean = None
            ic_orth_ir = None
        incremental_ok: bool | None = (
            abs(ic_orth_mean) >= INCREMENTAL_IC and abs(ic_orth_ir) >= INCREMENTAL_IR
            if ic_orth_mean is not None and ic_orth_ir is not None
            else None
        )

        gates = (signal_ok, monotonic_ok, net_return_ok, stability_ok, incremental_ok)
        passed = sum(1 for gate in gates if gate is True)
        evaluated = sum(1 for gate in gates if gate is not None)

        rows.append(
            {
                "factor": factor,
                "mean_ic": float(row["mean_ic"]),
                "ic_ir": ic_ir,
                "ic_t": ic_t,
                "signal_ok": signal_ok,
                "monotonicity": monotonicity,
                "monotonic_ok": monotonic_ok,
                "net_sharpe": net_sharpe,
                "max_drawdown": max_drawdown,
                "net_return_ok": net_return_ok,
                "yearly_agreement": agreement,
                "stability_ok": stability_ok,
                "n_years": n_years,
                "ic_orth": ic_orth_mean,
                "ic_orth_ir": ic_orth_ir,
                "incremental_ok": incremental_ok,
                "passed": passed,
                "evaluated": evaluated,
                "verdict": _verdict_label(passed, evaluated),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["passed", "mean_ic"],
        ascending=[False, False],
    ).reset_index(drop=True)
