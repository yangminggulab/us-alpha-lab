from __future__ import annotations

import pandas as pd

from us_alpha_lab.kline_discovery import build_kline_pattern_discovery
from us_alpha_lab.kline_tokens import add_kline_sequence_factors


def _synthetic_bars(periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            pulse = 1.5 if (date_index + ticker_index) % 11 == 0 else 0.0
            close = 50 + ticker_index * 3 + date_index * (0.12 + ticker_index * 0.01) + pulse
            open_ = close * (1 - 0.002 + ((date_index + ticker_index) % 5) * 0.001)
            high = max(open_, close) * (1 + 0.006 + (ticker_index % 3) * 0.001)
            low = min(open_, close) * (1 - 0.005 - (date_index % 4) * 0.001)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 800_000 + date_index * 2_000 + ticker_index * 20_000,
                    "alpha_range_compression_21d": (date_index % 9) / 8 + ticker_index * 0.01,
                    "alpha_gap_pressure_21d": ((date_index + ticker_index) % 7) / 6,
                    "alpha_intraday_quality_21d": ((date_index * 2 + ticker_index) % 13) / 12,
                }
            )
    return pd.DataFrame(rows)


def test_build_kline_pattern_discovery_returns_ranked_candidates() -> None:
    bars = _synthetic_bars()
    kline = add_kline_sequence_factors(bars)
    report = build_kline_pattern_discovery(
        kline,
        bars,
        horizon=2,
        windows=(10, 20),
        max_token_patterns=5,
        min_coverage=0.1,
    )

    assert not report.empty
    assert "alpha_kline_pattern_" in report.iloc[0]["factor"]
    assert "ic_orth" in report.columns
    assert "discovery_score" in report.columns
    assert report["discovery_score"].is_monotonic_decreasing
    assert set(report["verdict"]).issubset({"强增量候选", "增量候选", "观察", "剔除"})
