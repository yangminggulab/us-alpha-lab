from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from us_alpha_lab.kline_tokens import add_kline_sequence_factors
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


def _synthetic_bars(periods: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            trend = date_index * (0.18 + ticker_index * 0.03)
            wave = ((date_index + ticker_index) % 7 - 3) * 0.35
            close = 80 + ticker_index * 4 + trend + wave
            open_ = close * (1 + (((date_index * (ticker_index + 1)) % 5) - 2) * 0.001)
            high = max(open_, close) * (1 + 0.005 + (ticker_index % 3) * 0.002)
            low = min(open_, close) * (1 - 0.004 - (date_index % 4) * 0.001)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1_000_000
                    + date_index * 5_000
                    + ticker_index * 25_000
                    + ((date_index + ticker_index) % 9) * 30_000,
                    "dollar_volume": close * (1_000 + ticker_index * 100),
                    "volatility_21d": (ticker_index + 1) * 0.01 + (date_index % 5) * 0.001,
                    "alpha_range_compression_21d": (date_index % 11) / 10 + ticker_index * 0.01,
                    "alpha_gap_pressure_21d": ((date_index + ticker_index) % 13) / 12,
                    "alpha_intraday_quality_21d": ((date_index * 2 + ticker_index) % 17) / 16,
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


def test_build_html_report_has_kline_section(tmp_path) -> None:
    factors = _synthetic_bars()
    kline_factors = add_kline_sequence_factors(factors)
    path = build_html_report(
        factors,
        output_path=tmp_path / "research_report.html",
        horizon=2,
        top_n=1,
        factor_top_n=2,
        verdict_top_n=2,
        kline_factors=kline_factors,
    )

    html = path.read_text(encoding="utf-8")
    assert "K 线路径 token 实验" in html
    assert "alpha_kline" in html
    assert "正交 IC" in html
    assert "#kline" in html


def test_chinese_font_setup_does_not_raise() -> None:
    _setup_chinese_font()
    assert plt.rcParams["axes.unicode_minus"] is False
    assert plt.rcParams["font.sans-serif"]
