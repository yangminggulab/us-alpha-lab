from __future__ import annotations

import html as html_lib
from datetime import datetime, timezone
from pathlib import Path

from us_alpha_lab.alpha_registry import enabled_alpha_specs
from us_alpha_lab.config import ResearchConfig

# 流水线各步：标题、阶段描述、算法、输入、输出、命令行入口
_PIPELINE_STEPS = [
    {
        "title": "① 数据获取 fetch",
        "desc": "从行情数据源下载美股日线行情，落盘为原始 Parquet。",
        "algo": "Massive API → 日线 OHLCV（open/high/low/close/volume/vwap）",
        "inputs": "tickers、start、end、timespan、multiplier",
        "outputs": "data/raw/daily_bars.parquet",
        "cmd": "alpha-lab fetch",
    },
    {
        "title": "② 因子计算 factors",
        "desc": "在原始行情上叠加全部已启用因子，并计算横截面排名与 z-score。",
        "algo": "公式因子（动量/波动/量价/位置等）+ add_cross_sectional_ranks（截面分位数排名 + z-score）",
        "inputs": "data/raw/daily_bars.parquet",
        "outputs": "data/processed/factors.parquet",
        "cmd": "alpha-lab factors",
    },
    {
        "title": "③ 机器学习预测 ml-alpha（可选）",
        "desc": "用随机森林或 LightGBM 以全部因子预测未来收益，产物作为新因子进入因子池。",
        "algo": "Walk-forward ML 模型（random_forest / lightgbm）→ SimpleImputer(median)；滚动 train/test + embargo 防前视偏差",
        "inputs": "data/processed/factors.parquet",
        "outputs": "ml_prediction_5d 因子（写回 factors.parquet）",
        "cmd": "alpha-lab ml-alpha",
    },
    {
        "title": "④ 因子检验 IC 分析",
        "desc": "对每个因子计算对未来收益的预测力指标，筛出候选。",
        "algo": "Spearman IC（截面单调相关）→ IC_IR（IC 均值/标准差）→ 分位数收益 → discovery_score 排序",
        "inputs": "data/processed/factors.parquet",
        "outputs": "IC 报告 / 分位数收益表",
        "cmd": "alpha-lab discover-factors / report",
    },
    {
        "title": "⑤ 组合回测 backtest",
        "desc": "把因子分层做多空组合，验证预测力能否转化为真实收益。",
        "algo": "每日按因子分位数分层 → Top−Bottom 多空组合 → 换手成本扣减 → 基准对比（等权）→ 风险指标（回撤/夏普/IR）",
        "inputs": "data/processed/factors.parquet + IC 报告",
        "outputs": "leaderboard（综合得分 score）/ 每日收益序列 / 回测指标",
        "cmd": "alpha-lab leaderboard / backtest",
    },
    {
        "title": "⑥ 结果汇总 html-report",
        "desc": "把结论、候选因子、图表、回测聚合到一份自包含 HTML。",
        "algo": "matplotlib 图表（中文注释，base64 内嵌）→ 单文件 HTML（结论优先）",
        "inputs": "IC 报告 + leaderboard + 图表",
        "outputs": "reports/research_report.html",
        "cmd": "alpha-lab html-report",
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


def _config_cards(cfg: ResearchConfig) -> str:
    items = [
        ("标的", "、".join(cfg.tickers)),
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
    steps = "\n".join(_step_card(step) for step in _PIPELINE_STEPS)

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
    <p class="desc">数据从原始行情流转到最终报告，共 {len(_PIPELINE_STEPS)} 步。每一步标注了算法、输入、输出与命令行入口。</p>
    {steps}
    {ml_note}
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
