from __future__ import annotations

import html as html_lib
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from us_alpha_lab import visualization
from us_alpha_lab.analysis import factor_ic_report
from us_alpha_lab.leaderboard import build_factor_leaderboard
from us_alpha_lab.verdict import build_factor_verdict

_IC_CN = {
    "factor": "因子",
    "mean_ic": "平均 IC",
    "ic_std": "IC 标准差",
    "ic_ir": "IC 信息比率",
    "positive_ic_rate": "正向 IC 占比",
    "overall_spearman": "整体 Spearman",
    "days": "样本天数",
}

# 因子一句话解读。key 为因子名；value 用大白话说明"度量什么 + 为什么可能预测未来收益"。
FACTOR_GLOSSARY: dict[str, str] = {
    "reversal_1d": "看昨日涨跌，跌得越多后续越容易短期反弹，属于典型短线反转信号。",
    "momentum_5d": "看近 5 天累计涨幅，涨得多的股票短期往往延续惯性继续走强。",
    "momentum_21d": "看近一个月涨了多少，涨幅居前的股票中期通常保持强势延续。",
    "momentum_63d": "看近三个月累计涨幅，长期涨得强的股票往往继续跑赢，趋势延续。",
    "volatility_21d": "看近 21 天股价波动幅度，衡量风险大小，常用于识别低波动稳健型标的。",
    "volume_z_21d": "看近 21 天成交量相对平时放大还是萎缩，放量常伴随资金关注和行情启动。",
    "dollar_volume": "看每日成交额大小，成交额高代表流动性好、机构关注度高的活跃股。",
    "close_to_high": "看收盘价距当日高点多远，收得越接近高点说明买盘越强、日内占优。",
    "close_to_low": "看收盘价距当日低点多远，越远离低点说明收在高位、日内偏强。",
    "vwap_gap": "看收盘价比当日成交均价高还是低，站上均价线说明买方主导、多头强势。",
    "alpha_open_volume_corr_10": "看近 10 天开盘价高低与成交量是否背离，开盘量价不配合常预示反转。",
    "alpha_vwap_reversion_10": "看开盘价相对近期成交均价的偏离程度，偏离越大越可能向均价回归。",
    "alpha_range_ts_rank_9": "看近 9 天日内振幅的扩张与收敛，振幅收缩到极端往往酝酿均值回归。",
    "alpha_overnight_gap_reversal_5": "看近期隔夜跳空的强弱，跳空幅度过大往往随后反向修正。",
    "alpha_intraday_strength_5": "看近 5 天日内收盘相对开盘的强弱，日内买盘越强越可能延续。",
    "alpha_volume_price_divergence_21": "看近 21 天价与量的背离程度，价升量缩常预示后续均值回归。",
    "alpha_low_volatility_21d": "看 21 天波动率的反向，波动越低越受资金偏爱，低波动异象。",
    "alpha_price_to_21d_high": "看股价距 21 日高点多远，越接近高点越强，通常延续上行趋势。",
    "alpha_ma_gap_21d": "看股价比 21 日均线高出多少，站上均线越多说明多头越强势。",
    "alpha_liquidity_quality_21d": "看涨跌幅相对成交额的比值，流动性越好、冲击成本越低越受青睐。",
    "alpha_close_position_5": "看近 5 天收盘价在日内高低区间中的位置，收在区间高位说明买盘占优。",
    "alpha_residual_momentum_21d": "看剔除市场同涨同跌后的 21 日个股强弱，捕捉更纯的个股趋势。",
    "alpha_residual_reversal_5d": "看剔除市场影响后的 5 日过度涨跌，寻找短线残差反转机会。",
    "alpha_range_compression_21d": "看当日振幅相对近 21 日是否收缩，低振幅状态可能酝酿后续突破或均值回归。",
    "alpha_volatility_contraction_21d": "看短期波动是否低于月度波动，波动收缩常代表风险释放或趋势蓄势。",
    "alpha_volume_trend_5_21d": "看 5 日成交量相对 21 日均量是否放大，衡量近期资金关注升温。",
    "alpha_intraday_quality_21d": "看近 21 日日内上涨是否稳定，买盘越连续越像持续性资金行为。",
    "alpha_gap_pressure_21d": "看近 21 日隔夜跳空是否持续偏强，捕捉盘后信息和开盘重定价压力。",
    "ml_prediction_5d": "机器学习模型用全部因子走样外预测未来 5 日收益，作为集成信号参与排名。",
}


_LEADERBOARD_CN = {
    "factor": "因子",
    "mean_ic": "平均 IC",
    "ic_ir": "IC 信息比率",
    "positive_ic_rate": "正向 IC 占比",
    "periods": "回测期数",
    "total_return": "累计收益",
    "annualized_return": "年化收益",
    "annualized_volatility": "年化波动",
    "information_ratio": "信息比率",
    "max_drawdown": "最大回撤",
    "hit_rate": "胜率",
    "downside_volatility": "下行波动",
    "calmar_ratio": "Calmar 比率",
    "long_total_return": "多头累计收益",
    "benchmark_total_return": "基准累计收益",
    "long_excess_total_return": "多头超额收益",
    "benchmark_beta": "基准 Beta",
    "benchmark_correlation": "基准相关性",
    "excess_total_return": "超额累计收益",
    "excess_information_ratio": "超额信息比率",
    "excess_max_drawdown": "超额最大回撤",
    "average_turnover": "平均换手",
    "average_cost": "平均成本",
    "score": "综合得分",
}

_CSS = """
:root {
    --accent: #1f4e79;
    --accent-soft: #2a6db5;
    --bg: #f5f7fa;
    --card: #ffffff;
    --text: #1a1a1a;
    --muted: #6b7280;
    --line: #e5e7eb;
}
* { box-sizing: border-box; }
body {
    font-family: -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
    line-height: 1.6;
}
.header {
    background: linear-gradient(135deg, var(--accent), var(--accent-soft));
    color: #ffffff;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.header h1 { margin: 0 0 8px; font-size: 26px; }
.header p { margin: 2px 0; font-size: 14px; opacity: .95; }
.header .note { opacity: .75; font-size: 12px; }
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}
.card {
    background: var(--card);
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, .08);
}
.card .label { font-size: 12px; color: var(--muted); }
.card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
.section {
    background: var(--card);
    border-radius: 10px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, .08);
}
.section h2 {
    margin: 0 0 14px;
    font-size: 18px;
    color: var(--accent);
    border-bottom: 2px solid var(--line);
    padding-bottom: 10px;
}
.section h3 { font-size: 15px; color: #374151; margin: 18px 0 6px; }
table.dataframe {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
}
table.dataframe th {
    background: #f0f4f8;
    text-align: right;
    padding: 8px 10px;
    border-bottom: 2px solid #d1d9e0;
    white-space: nowrap;
}
table.dataframe td {
    text-align: right;
    padding: 6px 10px;
    border-bottom: 1px solid #eef1f4;
    white-space: nowrap;
}
table.dataframe tr:nth-child(even) td { background: #fafbfc; }
table.dataframe th:first-child,
table.dataframe td:first-child { text-align: left; }
img.chart {
    max-width: 100%;
    height: auto;
    border: 1px solid var(--line);
    border-radius: 8px;
    margin: 6px 0;
    background: #ffffff;
}
.chart-grid { display: grid; gap: 4px; }
details {
    margin: 14px 0;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 16px;
    background: #fcfdfe;
}
details summary {
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
    color: #111827;
}
.footer {
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    padding: 16px;
}
/* 导航 */
.nav {
    position: sticky;
    top: 0;
    z-index: 10;
    background: rgba(255, 255, 255, .94);
    backdrop-filter: blur(6px);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 10px 18px;
    margin-bottom: 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, .06);
}
.nav a {
    color: var(--accent);
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
}
.nav a:hover { text-decoration: underline; }
.nav .nav-method { margin-left: auto; }
.nav .nav-method a {
    color: #0f7b3e;
    background: #d9f2e0;
    padding: 2px 10px;
    border-radius: 999px;
}
.nav .nav-method a:hover { text-decoration: none; background: #c3e9cf; }
/* 摘要框 */
.summary {
    border-left: 4px solid var(--accent-soft);
    background: #f0f6fc;
    border-radius: 8px;
    padding: 16px 22px;
    margin-bottom: 20px;
}
.summary .verdict { font-size: 16px; font-weight: 700; margin-bottom: 10px; color: var(--accent); }
.summary ul { margin: 0; padding-left: 20px; font-size: 14px; }
.summary li { margin: 3px 0; }
/* 徽章 */
.badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    margin-left: 6px;
    vertical-align: 1px;
}
.badge.rec { background: #d9f2e0; color: #0f7b3e; }
.badge.watch { background: #fdf0d5; color: #9a6a00; }
.badge.weak { background: #eef1f4; color: #6b7280; }
.badge.neg { background: #fde7e5; color: #b42318; }
/* 因子卡片 */
.factor-cards { display: grid; grid-template-columns: 1fr; gap: 10px; }
.factor-card {
    display: flex;
    align-items: center;
    gap: 14px;
    background: #fcfdfe;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 16px;
}
.factor-card .rank {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: var(--accent);
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    flex-shrink: 0;
}
.factor-card .name { font-weight: 700; font-size: 14px; min-width: 220px; }
.factor-card .desc { font-size: 13px; color: #4b5563; flex: 1; }
.factor-card .metrics { font-size: 12px; color: var(--muted); text-align: right; white-space: nowrap; }
.factor-card .metrics b { color: var(--text); }
.section h3 { font-size: 15px; color: #374151; margin: 18px 0 6px; }
.guide { font-size: 13px; color: var(--muted); margin: 0 0 12px; }
table.dataframe.dim th, table.dataframe.dim td { padding: 5px 9px; font-size: 12px; }
"""


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    return f"{value:.{digits}f}"


def _table_html(frame: pd.DataFrame, rename: dict[str, str], digits: int = 4) -> str:
    keep = [column for column in frame.columns if column in rename]
    view = frame[keep].rename(columns=rename).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: _fmt(value, digits))
    return view.to_html(index=False, border=0, escape=True, classes="dataframe")


def _metric_cards(factors: pd.DataFrame, ic_report: pd.DataFrame) -> str:
    cards = [
        ("启用因子数", len(ic_report)),
        ("样本标的", int(factors["ticker"].nunique())),
        ("交易日数", int(pd.to_datetime(factors["date"]).nunique())),
        ("平均 |IC|", _fmt(ic_report["mean_ic"].abs().mean())),
        ("平均 IC 信息比率", _fmt(ic_report["ic_ir"].mean())),
    ]
    return "\n".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in cards
    )


def _signal_class(mean_ic: float) -> tuple[str, str]:
    """Return (class_name, label) for an IC value."""
    if mean_ic >= 0.03:
        return "rec", "强信号"
    if mean_ic >= 0.01:
        return "watch", "待观察"
    if mean_ic <= -0.01:
        return "neg", "负向信号"
    return "weak", "弱信号"

def _badge(badge_class: str, text: str) -> str:
    return f'<span class="badge {badge_class}">{html_lib.escape(text)}</span>'

def _factor_verdict(mean_ic: float, turnover: float | None = None) -> tuple[str, str, str]:
    """Return (badge_class, badge_label, verdict_text) for a factor."""
    cls, label = _signal_class(mean_ic)
    parts = []
    if mean_ic >= 0.03:
        parts.append("正向信号稳定，有预测力")
    elif mean_ic >= 0.01:
        parts.append("正向信号偏弱，需结合其它因子")
    elif mean_ic <= -0.01:
        parts.append("方向为负，可考虑反向使用")
    else:
        parts.append("信号不明显")
    if turnover is not None:
        parts.append("换手偏高" if turnover > 0.4 else "换手可控")
    return cls, label, "；".join(parts)


def _factor_cards_html(
    ic_report: pd.DataFrame,
    leaderboard: pd.DataFrame,
    top_n: int,
) -> str:
    """Build the executive-summary factor cards (rank, name, one-line verdict, metrics)."""
    lb = leaderboard.set_index("factor") if not leaderboard.empty else pd.DataFrame()
    rows = []
    for rank, (_, row) in enumerate(ic_report.head(top_n).iterrows(), 1):
        name = str(row["factor"])
        mean_ic = float(row["mean_ic"])
        ic_ir = float(row["ic_ir"]) if pd.notna(row["ic_ir"]) else None
        turnover = None
        if name in lb.index and pd.notna(lb.loc[name, "average_turnover"]):
            turnover = float(lb.loc[name, "average_turnover"])
        cls, label, verdict = _factor_verdict(mean_ic, turnover)
        gloss = FACTOR_GLOSSARY.get(name, "")
        desc = f"{verdict}。{gloss}" if gloss else verdict
        metrics = f"IC <b>{_fmt(mean_ic)}</b>　IR <b>{_fmt(ic_ir) if ic_ir is not None else '—'}</b>"
        if turnover is not None:
            metrics += f"　换手 <b>{_fmt(turnover)}</b>"
        rows.append(
            '<div class="factor-card">'
            f'<div class="rank">{rank}</div>'
            f'<div class="name">{html_lib.escape(name)}{_badge(cls, label)}</div>'
            f'<div class="desc">{html_lib.escape(desc)}</div>'
            f'<div class="metrics">{metrics}</div>'
            "</div>"
        )
    return "\n".join(rows)


def _summary_html(ic_report: pd.DataFrame, leaderboard: pd.DataFrame) -> str:
    """Build the top executive-summary block: verdict line + recommended/observed lists."""
    rec: list[str] = []
    watch: list[str] = []
    neg: list[str] = []
    # 以 ic_report 为准，leaderboard 提供换手
    lb = leaderboard.set_index("factor")
    for _, row in ic_report.iterrows():
        name = str(row["factor"])
        mean_ic = float(row["mean_ic"])
        turnover = None
        if name in lb.index and pd.notna(lb.loc[name, "average_turnover"]):
            turnover = float(lb.loc[name, "average_turnover"])
        cls, _, _ = _factor_verdict(mean_ic, turnover)
        if cls == "rec":
            rec.append(name)
        elif cls == "watch":
            watch.append(name)
        elif cls == "neg":
            neg.append(name)

    lines = []
    if rec:
        lines.append(f"推荐关注（正向强信号）：{'、'.join(html_lib.escape(x) for x in rec)}")
    if neg:
        lines.append(f"负向信号（可考虑反向使用）：{'、'.join(html_lib.escape(x) for x in neg)}")
    if watch:
        lines.append(f"待观察（信号偏弱）：{'、'.join(html_lib.escape(x) for x in watch)}")
    if not rec and not neg and not watch:
        lines.append("本次未发现有效信号因子，建议扩大样本或调整因子集。")
    items = "".join(f"<li>{line}</li>" for line in lines)
    verdict = f"共 {len(ic_report)} 个因子，其中正向强信号 {len(rec)} 个、负向信号 {len(neg)} 个、待观察 {len(watch)} 个"
    return (
        '<div class="summary">'
        f'<div class="verdict">{html_lib.escape(verdict)}</div>'
        f"<ul>{items}</ul>"
        "</div>"
    )


def _gate_cell(value: bool | None) -> str:
    if value is None:
        return '<span class="badge weak">—</span>'
    if value:
        return '<span class="badge rec">通过</span>'
    return '<span class="badge neg">未过</span>'


def _verdict_table_html(frame: pd.DataFrame) -> str:
    """渲染四关裁决表：每关用通过/未过徽章，最后一列给综合判定。"""
    if frame.empty:
        return "<p>当前没有可判定的因子。</p>"
    headers = [
        "因子",
        "平均IC",
        "ICIR",
        "IC t值",
        "信号关",
        "单调性",
        "单调关",
        "扣费夏普",
        "最大回撤",
        "净收益关",
        "分年同号率",
        "稳定关",
        "IC正交",
        "正交IR",
        "增量关",
        "通过",
        "判定",
    ]
    thead = "<thead><tr>" + "".join(f"<th>{name}</th>" for name in headers) + "</tr></thead>"
    body_rows = []
    for _, row in frame.iterrows():
        cells = [
            f"<td>{html_lib.escape(str(row['factor']))}</td>",
            f"<td>{_fmt(row['mean_ic'])}</td>",
            f"<td>{_fmt(row['ic_ir'])}</td>",
            f"<td>{_fmt(row['ic_t'])}</td>",
            f"<td>{_gate_cell(row['signal_ok'])}</td>",
            f"<td>{_fmt(row['monotonicity'])}</td>",
            f"<td>{_gate_cell(row['monotonic_ok'])}</td>",
            f"<td>{_fmt(row['net_sharpe'])}</td>",
            f"<td>{_fmt(row['max_drawdown'])}</td>",
            f"<td>{_gate_cell(row['net_return_ok'])}</td>",
            f"<td>{_fmt(row['yearly_agreement'])}</td>",
            f"<td>{_gate_cell(row['stability_ok'])}</td>",
            f"<td>{_fmt(row['ic_orth'])}</td>",
            f"<td>{_fmt(row['ic_orth_ir'])}</td>",
            f"<td>{_gate_cell(row['incremental_ok'])}</td>",
            f"<td>{int(row['passed'])}/{int(row['evaluated'])}</td>",
            f"<td>{html_lib.escape(str(row['verdict']))}</td>",
        ]
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="dataframe">{thead}<tbody>{"".join(body_rows)}</tbody></table>'


def _verdict_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    counts = {"可盈利候选": 0, "接近达标": 0, "部分达标": 0, "未达标": 0, "数据不足": 0}
    for verdict in frame["verdict"]:
        if verdict in counts:
            counts[verdict] += 1
    parts = [f"{label} {count} 个" for label, count in counts.items() if count]
    text = "、".join(parts) if parts else "无有效判定"
    return f'<p class="guide">判定汇总：{html_lib.escape(text)}。五关（信号、单调、净收益、稳定、增量）全过才算可盈利候选。</p>'


def _factor_blocks(
    factors: pd.DataFrame,
    ic_report: pd.DataFrame,
    horizon: int,
    top_n: int,
) -> str:
    blocks = []
    for rank, (_, row) in enumerate(ic_report.head(top_n).iterrows(), 1):
        name = str(row["factor"])
        timeseries = visualization.plot_ic_timeseries(factors, None, horizon=horizon, factor=name)
        quantile = visualization.plot_quantile_returns(factors, None, horizon=horizon, factor=name)
        spread = visualization.plot_cumulative_spread(factors, None, horizon=horizon, factor=name)
        summary = (
            f"#{rank} {html_lib.escape(name)}　·　平均 IC {_fmt(row['mean_ic'])}"
            f"　·　IC 信息比率 {_fmt(row['ic_ir'])}"
            f"　·　正向 IC 占比 {_fmt(row['positive_ic_rate'])}"
        )
        blocks.append(
            "<details open>"
            f"<summary>{summary}</summary>"
            '<div class="chart-grid">'
            f'<img class="chart" alt="{html_lib.escape(name)} 日度 IC 时序" '
            f'src="data:image/png;base64,{timeseries}">'
            f'<img class="chart" alt="{html_lib.escape(name)} 分位数收益" '
            f'src="data:image/png;base64,{quantile}">'
            f'<img class="chart" alt="{html_lib.escape(name)} 累计多空" '
            f'src="data:image/png;base64,{spread}">'
            "</div></details>"
        )
    return "\n".join(blocks)


def build_html_report(
    factors: pd.DataFrame,
    output_path: str | Path = "research_report.html",
    horizon: int = 5,
    top_n: int = 5,
    backtest_horizon: int = 1,
    quantiles: int = 5,
    cost_bps: float = 5.0,
    methodology_link: str | None = "methodology.html",
) -> Path:
    """Build a self-contained HTML research report (charts embedded as base64)."""
    ic_report = factor_ic_report(factors, horizon=horizon)
    generated_on = datetime.now(timezone.utc).date().isoformat()

    ic_summary = visualization.plot_ic_summary(factors, None, horizon=horizon)
    factor_blocks = _factor_blocks(factors, ic_report, horizon, top_n)

    leaderboard_frame, results = build_factor_leaderboard(
        factors,
        ic_report=ic_report,
        horizon=backtest_horizon,
        quantiles=quantiles,
        cost_bps=cost_bps,
    )
    if leaderboard_frame.empty:
        raise RuntimeError("回测没有产出有效因子，无法生成回测区。请检查因子数据。")

    best_factor = str(leaderboard_frame.iloc[0]["factor"])
    best_returns = results[best_factor].daily_returns
    leaderboard_chart = visualization.plot_leaderboard(leaderboard_frame, None)
    scatter_chart = visualization.plot_metric_scatter(leaderboard_frame, None)
    equity_chart = visualization.plot_backtest_equity(best_returns, best_factor, None)
    drawdown_chart = visualization.plot_backtest_drawdown(best_returns, best_factor, None)

    summary_html = _summary_html(ic_report, leaderboard_frame)
    factor_cards_html = _factor_cards_html(ic_report, leaderboard_frame, top_n)
    ic_table_html = _table_html(ic_report, _IC_CN)
    leaderboard_html = _table_html(leaderboard_frame.head(10), _LEADERBOARD_CN)

    verdict_frame = build_factor_verdict(
        factors,
        ic_report=ic_report,
        leaderboard=leaderboard_frame,
        horizon=horizon,
        quantiles=quantiles,
    )
    verdict_html = _verdict_table_html(verdict_frame)
    verdict_summary_html = _verdict_summary(verdict_frame)

    method_entry = (
        f'<a class="link-method" href="{html_lib.escape(methodology_link)}">'
        "研究方法与流水线 →</a>"
        if methodology_link
        else ""
    )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>量化因子研究报告 — {generated_on}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
    <h1>量化因子研究报告</h1>
    <p>生成日期：{generated_on}　·　前瞻收益窗口：{horizon} 个交易日　·　回测多空调仓窗口：{backtest_horizon} 个交易日</p>
    <p class="note">仅供学习研究，不构成投资建议；历史表现不代表未来收益。</p>
</div>

<nav class="nav">
    <a href="#summary">执行摘要</a>
    <a href="#cards">候选因子</a>
    <a href="#verdict">可盈利判定</a>
    <a href="#ic">IC 检验</a>
    <a href="#charts">因子图表</a>
    <a href="#backtest">回测验证</a>
    <span class="nav-method">{method_entry}</span>
</nav>

<div class="cards">
{_metric_cards(factors, ic_report)}
</div>

<div class="section" id="summary">
    <h2>执行摘要</h2>
    <p class="guide">这一节给出本次研究的结论：哪些因子值得关注、哪些需要观察，方便快速抓住重点。</p>
    {summary_html}
</div>

<div class="section" id="cards">
    <h2>候选因子</h2>
    <p class="guide">IC 排名前 {top_n} 的因子逐条说明：含义、信号强弱与回测换手。排名越靠前代表对未来 {horizon} 日收益的预测力越强。</p>
    <div class="factor-cards">
    {factor_cards_html}
    </div>
</div>

<div class="section" id="verdict">
    <h2>可盈利判定（五关裁决表）</h2>
    <p class="guide">五关漏斗：信号关（ICIR≥0.5 且 |IC t值|≥2，t 值已做 Newey–West 自相关修正）→ 单调关（分位数收益单调）→ 净收益关（扣费后夏普≥1 且回撤&lt;30%）→ 稳定关（分年 IC 同号率≥80%）→ 增量关（对公共因子池正交化后残差 IC≥0.01 且正交 IR≥0.3）。五关全过才判"可盈利候选"。</p>
    {verdict_summary_html}
    <details>
        <summary>查看全部因子四关判定明细</summary>
        {verdict_html}
    </details>
</div>

<div class="section" id="ic">
    <h2>IC 检验</h2>
    <p class="guide">Spearman IC 衡量因子值与未来收益的单调相关性。平均 IC 越接近 ±1 预测力越强，IC 信息比率（IC 均值/标准差）越高越稳定。下表为全部因子明细，可按需展开。</p>
    <details>
        <summary>查看全部 {len(ic_report)} 个因子的 IC 明细表</summary>
        {ic_table_html}
    </details>
    <h3>全部因子平均 IC 汇总</h3>
    <img class="chart" alt="IC 汇总" src="data:image/png;base64,{ic_summary}">
</div>

<div class="section" id="charts">
    <h2>因子图表</h2>
    <p class="guide">对 IC 排名前 {top_n} 的因子逐一看：IC 时序是否稳定、分位数收益是否单调、多空累计是否持续。每个因子一套图，可折叠。</p>
    {factor_blocks}
</div>

<div class="section" id="backtest">
    <h2>回测验证</h2>
    <p class="guide">把因子分层做多空组合（每 {backtest_horizon} 个交易日调仓，考虑换手成本），验证"预测力"能否变成真实收益。下表是最优 {top_n} 因子的回测指标。</p>
    {leaderboard_html}
    <h3>因子综合得分对比</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        <img class="chart" alt="因子榜单得分" src="data:image/png;base64,{leaderboard_chart}">
        <img class="chart" alt="平均 IC 与信息比率" src="data:image/png;base64,{scatter_chart}">
    </div>
    <h3>最优因子回测：{html_lib.escape(best_factor)}</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        <img class="chart" alt="回测净值" src="data:image/png;base64,{equity_chart}">
        <img class="chart" alt="回测回撤" src="data:image/png;base64,{drawdown_chart}">
    </div>
</div>

<div class="footer">
    本报告由 us_alpha_lab 自动生成　·　仅供学习研究，不构成投资建议，历史表现不代表未来收益
</div>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path
