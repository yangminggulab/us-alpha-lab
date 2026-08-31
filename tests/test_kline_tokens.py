from __future__ import annotations

import numpy as np
import pandas as pd

from us_alpha_lab.kline_tokens import (
    STATE_COLUMNS,
    add_kline_sequence_factors,
    kline_factor_columns,
    make_kline_states,
)


def _sample_bars() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    rows = []
    for ticker, offset in [("AAA", 0.0), ("BBB", 12.0)]:
        for index, date in enumerate(dates):
            wave = np.sin(index / 4) * 1.5
            close = 100 + offset + index * 0.25 + wave
            open_price = close * (1 - 0.002 + (index % 3) * 0.001)
            high = max(open_price, close) * (1 + 0.004 + (index % 5) * 0.001)
            low = min(open_price, close) * (1 - 0.004 - (index % 4) * 0.001)
            volume = 1_000_000 + index * 2_500 + (index % 7) * 40_000
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "vwap": (high + low + close) / 3,
                }
            )
    return pd.DataFrame(rows)


def test_make_kline_states_adds_discrete_tokens() -> None:
    states = make_kline_states(_sample_bars())

    for column in STATE_COLUMNS:
        assert column in states.columns

    assert states["kline_token"].notna().all()
    assert states["kline_token"].str.contains("|", regex=False).all()
    assert states.groupby("ticker")["date"].is_monotonic_increasing.all()


def test_add_kline_sequence_factors_creates_bounded_features() -> None:
    factors = add_kline_sequence_factors(_sample_bars(), windows=(20, 60))

    expected_columns = kline_factor_columns((20, 60))
    for column in expected_columns:
        assert column in factors.columns

    mature = factors.groupby("ticker").tail(30)
    assert mature[expected_columns].notna().any().all()
    assert ((mature[expected_columns] >= 0) & (mature[expected_columns] <= 1)).all().all()


def test_add_kline_sequence_factors_rejects_tiny_windows() -> None:
    try:
        add_kline_sequence_factors(_sample_bars(), windows=(1,))
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("Expected tiny K-line windows to be rejected.")
