from __future__ import annotations

from us_alpha_lab.alpha_registry import alpha_registry_frame, enabled_alpha_names, get_alpha_spec
from us_alpha_lab.factors import FACTOR_COLUMNS


def test_alpha_registry_matches_factor_columns() -> None:
    assert enabled_alpha_names() == FACTOR_COLUMNS
    assert get_alpha_spec("volatility_21d").lookback == 21


def test_alpha_registry_frame_has_metadata() -> None:
    frame = alpha_registry_frame()

    assert {"name", "category", "formula", "source", "expected_direction", "lookback"}.issubset(
        frame.columns
    )
    assert frame["name"].is_unique
