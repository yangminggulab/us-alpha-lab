from __future__ import annotations

import pandas as pd
import pytest

from us_alpha_lab.latent_participant import (
    add_latent_participant_states,
    latent_factor_columns,
    latent_probability_columns,
)


def _synthetic_bars(periods: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rows = []
    for ticker_index, ticker in enumerate(["AAA", "BBB", "CCC"]):
        for date_index, date in enumerate(dates):
            shock = -2.0 if (date_index + ticker_index) % 17 == 0 else 0.0
            close = 100 + ticker_index * 5 + date_index * 0.2 + shock
            open_ = close * (1 + ((date_index + ticker_index) % 5 - 2) * 0.002)
            high = max(open_, close) * (1 + 0.006 + ticker_index * 0.001)
            low = min(open_, close) * (1 - 0.006 - (date_index % 3) * 0.001)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1_000_000 + ticker_index * 20_000 + date_index * 5_000,
                }
            )
    return pd.DataFrame(rows)


def test_add_latent_participant_states_outputs_probabilities() -> None:
    frame = add_latent_participant_states(_synthetic_bars(), lookback=10)
    probability_columns = latent_probability_columns()

    assert set(probability_columns).issubset(frame.columns)
    assert set(latent_factor_columns()).issubset(frame.columns)
    assert "latent_dominant_state" in frame.columns
    assert "latent_state_confidence" in frame.columns
    assert frame[probability_columns].notna().all().all()
    assert frame[probability_columns].sum(axis=1).between(0.999, 1.001).all()
    assert frame["latent_state_confidence"].between(0, 1).all()


def test_add_latent_participant_states_rejects_bad_parameters() -> None:
    bars = _synthetic_bars()
    with pytest.raises(ValueError, match="lookback"):
        add_latent_participant_states(bars, lookback=4)
    with pytest.raises(ValueError, match="persistence"):
        add_latent_participant_states(bars, persistence=1.0)
    with pytest.raises(ValueError, match="emission_scale"):
        add_latent_participant_states(bars, emission_scale=0.0)
