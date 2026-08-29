from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from us_alpha_lab.report import build_html_report
from us_alpha_lab.visualization import _setup_chinese_font


def _synthetic_factors() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": 100 + date_index + ticker_index,
                    "dollar_volume": ticker_index + date_index * 0.1,
                    "volatility_21d": ticker_index * -1 + date_index * 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_build_html_report_embeds_charts(tmp_path) -> None:
    path = build_html_report(
        _synthetic_factors(),
        output_path=tmp_path / "research_report.html",
        horizon=2,
        top_n=2,
    )

    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert len(html) > 10_000
    # ic_summary(1) + top2*3 + leaderboard + scatter + equity + drawdown = 11 charts
    assert html.count("data:image/png;base64,") == 11


def test_build_html_report_contains_chinese_labels(tmp_path) -> None:
    path = build_html_report(
        _synthetic_factors(),
        output_path=tmp_path / "research_report.html",
        horizon=2,
        top_n=1,
    )

    html = path.read_text(encoding="utf-8")
    for keyword in (
        "量化因子研究报告",
        "执行摘要",
        "候选因子",
        "IC 检验",
        "因子图表",
        "回测验证",
        "平均 IC",
        "多空",
        "回撤",
        "研究",
    ):
        assert keyword in html


def test_build_html_report_has_summary_and_cards(tmp_path) -> None:
    path = build_html_report(
        _synthetic_factors(),
        output_path=tmp_path / "research_report.html",
        horizon=2,
        top_n=2,
    )

    html = path.read_text(encoding="utf-8")
    # 摘要区含"正向强信号"结论 + 因子卡片（含因子名）
    assert "正向强信号" in html
    assert "候选因子" in html
    assert "dollar_volume" in html
    # 导航锚点
    for anchor in ("#summary", "#cards", "#verdict", "#ic", "#charts", "#backtest"):
        assert anchor in html


def test_build_html_report_has_verdict_gate_table(tmp_path) -> None:
    path = build_html_report(
        _synthetic_factors(),
        output_path=tmp_path / "research_report.html",
        horizon=2,
        top_n=2,
    )

    html = path.read_text(encoding="utf-8")
    assert "可盈利判定（五关裁决表）" in html
    for keyword in ("信号关", "单调关", "净收益关", "稳定关", "增量关", "IC t值", "通过"):
        assert keyword in html


def test_chinese_font_setup_does_not_raise() -> None:
    _setup_chinese_font()
    assert plt.rcParams["axes.unicode_minus"] is False
    assert plt.rcParams["font.sans-serif"]
