from __future__ import annotations

from us_alpha_lab.alpha_registry import enabled_alpha_specs
from us_alpha_lab.config import ResearchConfig
from us_alpha_lab.methodology import build_methodology_html


def _config(tmp_path) -> ResearchConfig:
    return ResearchConfig(
        tickers=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-12-31",
        factors_path=tmp_path / "factors.parquet",
    )


def test_build_methodology_html_dynamic_content(tmp_path) -> None:
    path = build_methodology_html(_config(tmp_path), output_path=tmp_path / "methodology.html")

    assert path.exists()
    html = path.read_text(encoding="utf-8")
    # 动态因子数：与 registry 实际启用数一致
    total = len(enabled_alpha_specs())
    assert f"当前因子池（{total} 个启用）" in html
    # 每个启用因子的名字都出现在表格里
    for spec in enabled_alpha_specs():
        assert spec.name in html


def test_build_methodology_html_contains_pipeline_and_chinese(tmp_path) -> None:
    path = build_methodology_html(_config(tmp_path), output_path=tmp_path / "methodology.html")

    html = path.read_text(encoding="utf-8")
    for keyword in (
        "研究方法与流水线",
        "研究流水线",
        "数据获取",
        "因子计算",
        "机器学习预测",
        "因子检验",
        "组合回测",
        "结果汇总",
        "当前因子池",
    ):
        assert keyword in html


def test_methodology_backlink_points_to_report(tmp_path) -> None:
    path = build_methodology_html(_config(tmp_path), output_path=tmp_path / "methodology.html")

    html = path.read_text(encoding="utf-8")
    assert 'href="research_report.html"' in html
