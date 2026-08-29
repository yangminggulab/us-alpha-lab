from __future__ import annotations

import numpy as np
import pandas as pd

from us_alpha_lab.labels import add_forward_return_label
from us_alpha_lab.verdict import (
    build_factor_verdict,
    newey_west_t_stat,
    orthogonalized_ic_series,
    quantile_monotonicity,
    yearly_ic_sign_agreement,
)


def test_newey_west_t_stat_positive_signal() -> None:
    rng = np.random.default_rng(0)
    ic = pd.Series(0.03 + rng.normal(0, 0.05, 400))
    assert newey_west_t_stat(ic, lag=5) > 2.0


def test_newey_west_t_stat_more_lags_shrink_t() -> None:
    # 平滑序列有正自相关，拉长 NW 滞后窗口应压低 |t|
    rng = np.random.default_rng(2)
    smooth = pd.Series(rng.normal(0.02, 0.05, 300)).rolling(5).mean().dropna()
    assert abs(newey_west_t_stat(smooth, lag=5)) > abs(newey_west_t_stat(smooth, lag=20))


def test_yearly_ic_sign_agreement() -> None:
    index = pd.date_range("2020-01-01", periods=504, freq="B")
    ic = pd.Series([0.03] * 504, index=index)
    agreement, n_years = yearly_ic_sign_agreement(ic)
    assert n_years >= 2
    assert agreement == 1.0


def test_yearly_ic_sign_agreement_insufficient() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="B")
    ic = pd.Series([0.03] * 40, index=index)
    agreement, n_years = yearly_ic_sign_agreement(ic)
    assert n_years < 2
    assert agreement == 0.0


def _monotonic_factors() -> pd.DataFrame:
    # close(d, i) = 100 + i*d → 前瞻 1 日收益随 ticker i 严格上升，构成单调因子
    # 504 个交易日约 2 年，让稳定关有足够年份可判定
    # reversal_1d 是纯噪声：作为池成员存在，但不携带因子信号，使增量关可判定且通过
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=504, freq="B")
    rows = []
    for date in dates:
        for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100.0 + i * date.day,
                    "factor": float(i),
                    "reversal_1d": rng.normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


def test_quantile_monotonicity_monotonic() -> None:
    factors = _monotonic_factors()
    value = quantile_monotonicity(factors, "factor", horizon=1, quantiles=5)
    assert value > 0.7


def test_build_factor_verdict_schema() -> None:
    factors = _monotonic_factors()
    ic_report = pd.DataFrame(
        [
            {"factor": "factor", "mean_ic": 0.05, "ic_ir": 0.8, "ic_std": 0.06},
        ]
    )
    leaderboard = pd.DataFrame(
        [
            {
                "factor": "factor",
                "information_ratio": 1.2,
                "max_drawdown": -0.2,
                "average_turnover": 0.3,
            },
        ]
    )
    frame = build_factor_verdict(factors, ic_report, leaderboard, horizon=1, quantiles=5)
    assert not frame.empty
    row = frame.iloc[0]
    assert row["signal_ok"] == True
    assert row["monotonic_ok"] == True
    assert row["net_return_ok"] == True
    assert row["stability_ok"] == True
    assert row["incremental_ok"] == True
    assert row["verdict"] == "可盈利候选"


def _pool_combo_frame() -> pd.DataFrame:
    # factor = 2*momentum_5d - 1.5*reversal_1d + 微小噪声 → 信号基本全在池里，正交化后应被剥离
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=504, freq="B")
    rows = []
    for date in dates:
        for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
            momentum_5d = float(i) + rng.normal(0, 0.05)
            reversal_1d = rng.normal(0, 0.05)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100.0 + i * date.day,
                    "momentum_5d": momentum_5d,
                    "reversal_1d": reversal_1d,
                    "factor": 2 * momentum_5d - 1.5 * reversal_1d + rng.normal(0, 1e-4),
                }
            )
    return pd.DataFrame(rows)


def _daily_ic_mean(data: pd.DataFrame, factor: str, label: str) -> float:
    values = []
    for _, group in data.groupby("date"):
        clean = group[[factor, label]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 2 or clean[factor].nunique() < 2 or clean[label].nunique() < 2:
            continue
        values.append(clean[factor].corr(clean[label], method="spearman"))
    return float(pd.Series(values).mean())


def test_orthogonalized_ic_linear_combo_is_zero() -> None:
    data = add_forward_return_label(_pool_combo_frame(), horizon=1)
    label = "future_return_1d"
    raw_mean = _daily_ic_mean(data, "factor", label)
    orth = orthogonalized_ic_series(
        data,
        "factor",
        pool_columns=["momentum_5d", "reversal_1d"],
        label=label,
    )
    assert not orth.empty
    # 原始 IC 承载了池内信号（很高），正交化后应把共享信号剥掉，残差 IC 显著变小
    assert abs(orth.mean()) < abs(raw_mean) * 0.5


def test_orthogonalized_ic_independent_keeps_signal() -> None:
    # 池只有纯噪声 reversal_1d，factor 携带独立信号 → 正交化后残差 IC 应显著为正
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=504, freq="B")
    rows = []
    for date in dates:
        for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100.0 + i * date.day,
                    "factor": float(i) + rng.normal(0, 0.1),
                    "reversal_1d": rng.normal(0, 0.05),
                }
            )
    data = add_forward_return_label(pd.DataFrame(rows), horizon=1)
    series = orthogonalized_ic_series(
        data,
        "factor",
        pool_columns=["reversal_1d"],
        label="future_return_1d",
    )
    assert not series.empty
    assert series.mean() > 0.3
