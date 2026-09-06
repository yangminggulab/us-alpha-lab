from __future__ import annotations

import html as html_lib
from datetime import datetime, timezone
from pathlib import Path

from us_alpha_lab.alpha_registry import enabled_alpha_specs
from us_alpha_lab.config import ResearchConfig


# 流水线各步：标题、阶段描述、算法、输入、输出、命令行入口
def _pipeline_steps(cfg: ResearchConfig) -> list[dict[str, str]]:
    raw_path = str(cfg.raw_path)
    factors_path = str(cfg.factors_path)
    return [
        {
            "title": "① 数据获取 fetch",
            "desc": "从行情数据源下载美股日线行情，落盘为原始 Parquet。",
            "algo": "Massive API → 日线 OHLCV（open/high/low/close/volume/vwap）",
            "inputs": "tickers、start、end、timespan、multiplier",
            "outputs": raw_path,
            "cmd": "alpha-lab fetch --config <config>",
        },
        {
            "title": "② 因子计算 factors",
            "desc": "在原始行情上叠加全部已启用因子，并计算横截面排名与 z-score。",
            "algo": "公式因子（动量/波动/量价/位置等）+ add_cross_sectional_ranks（截面分位数排名 + z-score）",
            "inputs": raw_path,
            "outputs": factors_path,
            "cmd": "alpha-lab factors --config <config>",
        },
        {
            "title": "③ K 线路径 token（实验）",
            "desc": "把 OHLCV 路径离散成 K 线 token，生成独立的路径表征因子，不覆盖主因子池。",
            "algo": "涨跌幅桶 + 振幅桶 + 成交量冲击桶 + 收盘位置桶 → 20/60 日 entropy、n-gram、状态转移、路径相似度",
            "inputs": raw_path,
            "outputs": "data/processed/kline_sequence_factors.parquet",
            "cmd": "alpha-lab kline-factors --config <config>",
        },
        {
            "title": "④ K 线路径发现 discover-kline-patterns（实验）",
            "desc": "批量枚举 K 线 token 的状态、转移和组合模式，寻找公共价格行为因子解释不了的路径候选。",
            "algo": "单状态/状态转移/组合 token → 滚动出现频率 → 原始 IC → 对 range_compression、gap_pressure、intraday_quality 正交 IC → discovery_score 排序",
            "inputs": "data/processed/kline_sequence_factors.parquet + " + factors_path,
            "outputs": "reports/kline_discovery/kline_pattern_candidates.csv",
            "cmd": "alpha-lab discover-kline-patterns --config <config>",
        },
        {
            "title": "⑤ 隐藏参与者状态 latent-participants（实验）",
            "desc": "用日线 proxy 做低频版 HMM/Bayesian filtering，估计机构吸筹、派发、被迫卖出、做市压力、套利修复、噪声交易等状态概率。",
            "algo": "成交量冲击 + 跳空 + 日内强弱 + 振幅 + 收盘位置 + 5 日趋势 → 状态原型 Gaussian emission → HMM 递推平滑 → alpha_latent_*_prob",
            "inputs": raw_path,
            "outputs": "data/processed/latent_participant_states.parquet",
            "cmd": "alpha-lab latent-participants --config <config>",
        },
        {
            "title": "⑥ 统一实验检验 validate-experiment",
            "desc": "所有独立实验产物都接入同一套检验入口，避免 K 线、隐藏参与者、ML 候选各自写一套不可比较的标准。",
            "algo": "实验 alpha_* 列 → 覆盖率 → IC / Newey-West t → 分位单调性 → 正交 IC → 扣费多空回测 → 分年稳定性 → 五关 verdict",
            "inputs": "任意实验 parquet/csv + " + factors_path,
            "outputs": "reports/experiment_validation/latest_validation.csv",
            "cmd": "alpha-lab validate-experiment <experiment-path> --config <config>",
        },
        {
            "title": "⑦ 机器学习预测 ml-alpha（可选）",
            "desc": "用随机森林或 LightGBM 以全部因子预测未来收益，产物作为新因子进入因子池。",
            "algo": "Walk-forward ML 模型（random_forest / lightgbm / LambdaRank / rank_xendcg）→ SimpleImputer(median)；滚动 train/test + embargo 防前视偏差",
            "inputs": factors_path,
            "outputs": "ml_prediction_5d 因子（写回 factors.parquet 或指定 output-path）",
            "cmd": "alpha-lab ml-alpha --config <config>",
        },
        {
            "title": "⑧ 因子检验 IC 分析",
            "desc": "对每个因子计算对未来收益的预测力指标，筛出候选。",
            "algo": "Spearman IC（截面单调相关）→ IC_IR（IC 均值/标准差）→ 分位数收益 → discovery_score 排序",
            "inputs": factors_path,
            "outputs": "IC 报告 / 分位数收益表",
            "cmd": "alpha-lab discover-factors --config <config>",
        },
        {
            "title": "⑨ 组合回测 backtest",
            "desc": "把因子分层做多空组合，验证预测力能否转化为真实收益。",
            "algo": "每日按因子分位数分层 → Top−Bottom 多空组合 → 换手成本扣减 → 基准对比（等权）→ 风险指标（回撤/夏普/IR）",
            "inputs": f"{factors_path} + IC 报告",
            "outputs": "leaderboard（综合得分 score）/ 每日收益序列 / 回测指标",
            "cmd": "alpha-lab leaderboard --config <config>",
        },
        {
            "title": "⑩ 结果汇总 html-report",
            "desc": "把结论、候选因子、图表、回测聚合到一份自包含 HTML。",
            "algo": "matplotlib 图表（中文注释，base64 内嵌）→ 单文件 HTML（结论优先）",
            "inputs": "IC 报告 + leaderboard + K 线路径发现 + 隐藏参与者状态 + 图表",
            "outputs": "research_report.html",
            "cmd": "alpha-lab html-report --config <config>",
        },
    ]


def _l2_l3_pipeline_steps() -> list[dict[str, str]]:
    return [
        {
            "title": "M1 覆盖与字段语义 a-share-l3-coverage",
            "desc": "逐日扫描 A 股 L2/L3 本地样本，锁定交易所覆盖、字段含义、订单回连键与撤单来源。",
            "algo": "逐笔委托 + 逐笔成交/撤单 → 按 wind_code/date 统计覆盖 → ask/bid/cancel 回连率 → 交易所覆盖边界",
            "inputs": "data/raw/a_share_l2_l3/<date>/ 逐笔委托、逐笔成交、行情快照",
            "outputs": "reports/l2_l3/daily_coverage.csv",
            "cmd": "alpha-lab a-share-l3-coverage --dates <YYYYMMDD,...>",
        },
        {
            "title": "M2 订单生命周期 a-share-l3-lifecycle",
            "desc": "把每条深市委托回连成交与撤单事件，生成 per-order 生命周期表。",
            "algo": "orders.ex_order_id ↔ trades.ask_order_id/bid_order_id/cancel_order_id → filled/canceled/remaining 三分类终态 → 生命周期下界",
            "inputs": "单日单股逐笔委托 + 逐笔成交/撤单",
            "outputs": "reports/l2_l3/lifecycle_<date>_<wind_code>.parquet",
            "cmd": "alpha-lab a-share-l3-lifecycle --date <YYYYMMDD> --wind-code <000725.SZ>",
        },
        {
            "title": "M3 分钟特征 a-share-l3-minute-features",
            "desc": "把订单生命周期、成交事件和盘口快照聚合到分钟，形成可供聚类和状态模型使用的微观结构面板。",
            "algo": "submit/trade/cancel/quote 四类时钟 → 填单率、撤单率、开放率、订单集中度、主动侧不平衡、盘口成交量对账",
            "inputs": "per-order 生命周期表 + 逐笔成交/撤单 + 行情快照",
            "outputs": "reports/l2_l3/minute_features_<date>_<wind_code>.parquet",
            "cmd": "alpha-lab a-share-l3-minute-features --date <YYYYMMDD> --wind-code <000725.SZ>",
        },
        {
            "title": "M4 质量门 a-share-l3-quality-gates",
            "desc": "对分钟特征做会计恒等式、边界、缺失和主动侧修复检查，只让可信样本进入聚类。",
            "algo": "成交量守恒 + 撤单量守恒 + 盘口累计量对账 + ratio/imbalance 边界 + bs_flag 主动侧恢复",
            "inputs": "分钟特征表",
            "outputs": "reports/l2_l3/minute_quality_gates_sample.csv",
            "cmd": "alpha-lab a-share-l3-quality-gates --dates <YYYYMMDD,...> --wind-codes <000725.SZ,...>",
        },
        {
            "title": "M5 静态势力聚类 a-share-l3-static-clusters",
            "desc": "先不假设状态转移，只在高质量分钟窗口上做行为聚类，得到微观结构行为 taxonomy。",
            "algo": "连续竞价分钟 → winsorize/log1p/z-score → KMeans K=2..6 → silhouette 选 K → profile/shares 解释",
            "inputs": "通过质量门的分钟特征表",
            "outputs": "reports/l2_l3/static_clusters_sample/labels.parquet + diagnostics/profile/shares.csv",
            "cmd": "alpha-lab a-share-l3-static-clusters --dates <YYYYMMDD,...> --wind-codes <000725.SZ,...>",
        },
        {
            "title": "M6 状态序列 HMM（下一步）",
            "desc": "在静态行为簇的基础上加入时间转移约束，识别订单生命周期阶段和势力切换。",
            "algo": "分钟特征或静态簇标签 → HMM/状态转移矩阵 → regime duration、transition surprise、session-level 标签",
            "inputs": "M3/M5 产物",
            "outputs": "待定：l2_l3_hmm_states.parquet + 状态解释报告",
            "cmd": "计划中",
        },
    ]


def _escape(value: object) -> str:
    return html_lib.escape(str(value))


def _step_card(step: dict[str, str]) -> str:
    rows = "".join(
        f"<tr><td class='k'>{_escape(label)}</td><td>{_escape(step[key])}</td></tr>"
        for label, key in (
            ("算法", "algo"),
            ("输入", "inputs"),
            ("输出", "outputs"),
            ("命令", "cmd"),
        )
    )
    return (
        '<div class="step">'
        f"<h3>{_escape(step['title'])}</h3>"
        f"<p class='desc'>{_escape(step['desc'])}</p>"
        f"<table class='kv'>{rows}</table>"
        "</div>"
    )


def _factor_table() -> str:
    specs = enabled_alpha_specs()
    if not specs:
        return "<p>当前没有启用的因子。</p>"
    header = (
        "<thead><tr><th>因子</th><th>分类</th><th>公式</th>"
        "<th>回看窗口</th><th>预期方向</th><th>说明</th></tr></thead>"
    )
    body_rows = []
    for spec in specs:
        body_rows.append(
            "<tr>"
            f"<td>{_escape(spec.name)}</td>"
            f"<td>{_escape(spec.category)}</td>"
            f"<td class='formula'>{_escape(spec.formula)}</td>"
            f"<td>{spec.lookback}</td>"
            f"<td>{spec.expected_direction}</td>"
            f"<td class='desc'>{_escape(spec.description)}</td>"
            "</tr>"
        )
    return f"<table class='dataframe'>{header}<tbody>{''.join(body_rows)}</tbody></table>"


def _ticker_summary(tickers: list[str], preview: int = 10) -> str:
    shown = "、".join(tickers[:preview])
    suffix = "..." if len(tickers) > preview else ""
    return f"{len(tickers)} 个（{shown}{suffix}）"


def _config_cards(cfg: ResearchConfig) -> str:
    items = [
        ("配置标的", _ticker_summary(cfg.tickers)),
        ("起始日期", cfg.start),
        ("结束日期", cfg.end),
        ("K 线周期", f"{cfg.multiplier} {cfg.timespan}"),
        ("因子数", str(len(enabled_alpha_specs()))),
    ]
    return "\n".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{_escape(value)}</div></div>'
        for label, value in items
    )


def _l2_l3_status_cards() -> str:
    items = [
        ("研究域", "A 股深市逐笔委托 + 逐笔成交/撤单"),
        ("回连键", "orders.ex_order_id"),
        ("撤单来源", "trades.trade_code = C"),
        ("当前阶段", "生命周期、分钟特征、质量门、静态聚类已接入"),
        ("下一步", "HMM 状态序列"),
    ]
    return "\n".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{_escape(value)}</div></div>'
        for label, value in items
    )


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
    color: #fff;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.header h1 { margin: 0 0 8px; font-size: 26px; }
.header p { margin: 2px 0; font-size: 14px; opacity: .95; }
.header .note { opacity: .75; font-size: 12px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 20px; }
.card { background: var(--card); border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.card .label { font-size: 12px; color: var(--muted); }
.card .value { font-size: 20px; font-weight: 600; margin-top: 4px; }
.section { background: var(--card); border-radius: 10px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.section h2 { margin: 0 0 14px; font-size: 18px; color: var(--accent); border-bottom: 2px solid var(--line); padding-bottom: 10px; }
.step { border: 1px solid var(--line); border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; background: #fcfdfe; }
.step h3 { margin: 0 0 6px; font-size: 15px; color: #111827; }
.step .desc { margin: 0 0 8px; font-size: 13px; color: #4b5563; }
table.kv { border-collapse: collapse; width: 100%; font-size: 13px; }
table.kv td { padding: 4px 8px; border-bottom: 1px solid #eef1f4; vertical-align: top; }
table.kv td.k { width: 60px; color: var(--muted); white-space: nowrap; }
table.dataframe { border-collapse: collapse; width: 100%; font-size: 13px; }
table.dataframe th { background: #f0f4f8; text-align: left; padding: 8px 10px; border-bottom: 2px solid #d1d9e0; }
table.dataframe td { padding: 6px 10px; border-bottom: 1px solid #eef1f4; vertical-align: top; }
table.dataframe tr:nth-child(even) td { background: #fafbfc; }
td.formula { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; color: #0f4c81; }
.footer { color: var(--muted); font-size: 12px; text-align: center; padding: 16px; }
.link-back {
    display: inline-block;
    margin: 4px 0 16px;
    color: var(--accent);
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
}
.link-back:hover { text-decoration: underline; }
"""


def build_methodology_html(
    cfg: ResearchConfig,
    output_path: str | Path = "methodology.html",
    report_link: str | None = "research_report.html",
) -> Path:
    """Build a dynamic methodology/pipeline HTML from the current code state."""
    generated_on = datetime.now(timezone.utc).date().isoformat()
    pipeline_steps = _pipeline_steps(cfg)
    steps = "\n".join(_step_card(step) for step in pipeline_steps)
    l2_l3_pipeline_steps = _l2_l3_pipeline_steps()
    l2_l3_steps = "\n".join(_step_card(step) for step in l2_l3_pipeline_steps)

    total_enabled = len(enabled_alpha_specs())
    ml_enabled = any(spec.source == "local_ml" for spec in enabled_alpha_specs())
    ml_note = (
        "<p class='desc'>机器学习步（③）当前已启用：随机森林生成的 "
        "<code>ml_prediction_5d</code> 因子会作为普通因子参与后续检验与回测。</p>"
        if ml_enabled
        else "<p class='desc'>机器学习步（③）当前未启用，因子池仅含公式因子。</p>"
    )

    back_link = (
        f'<a class="link-back" href="{html_lib.escape(report_link)}">'
        "← 查看本次研究报告 research_report.html</a>"
        if report_link
        else ""
    )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>研究方法与流水线 — {generated_on}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
    <h1>研究方法与流水线</h1>
    <p>生成日期：{generated_on}　·　动态读取当前代码状态与配置生成</p>
    <p class="note">仅供学习研究，不构成投资建议；历史表现不代表未来收益。</p>
</div>

{back_link}

<div class="cards">
{_config_cards(cfg)}
</div>

<div class="section">
    <h2>研究流水线</h2>
    <p class="desc">数据从原始行情流转到最终报告，共 {len(pipeline_steps)} 步。每一步标注了算法、输入、输出与命令行入口。</p>
    {steps}
    {ml_note}
</div>

<div class="section">
    <h2>A 股 L2/L3 逐笔研究流水线</h2>
    <p class="desc">这条链路是独立研究分支，服务于订单生命周期、撤单压力、成交主动性和隐藏状态识别；当前结论先限定在有委托流的深市样本。</p>
    <div class="cards">
    {_l2_l3_status_cards()}
    </div>
    {l2_l3_steps}
</div>

<div class="section">
    <h2>当前因子池（{total_enabled} 个启用）</h2>
    <p class="desc">以下清单动态读取自 alpha_registry，随代码中因子的新增 / 禁用 / 修改而更新。</p>
    {_factor_table()}
</div>

<div class="footer">
    本方法文档由 us_alpha_lab 动态生成　·　仅供学习研究，不构成投资建议，历史表现不代表未来收益
</div>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path
