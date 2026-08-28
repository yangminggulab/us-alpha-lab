from __future__ import annotations

import pandas as pd

from us_alpha_lab.operators import cs_rank, delta, ts_rank


def test_operator_shapes() -> None:
    index = pd.MultiIndex.from_product(
        [["AAA", "BBB"], pd.date_range("2024-01-01", periods=5, freq="B")],
        names=["ticker", "date"],
    )
    series = pd.Series(range(len(index)), index=index, dtype=float)

    assert delta(series).index.equals(index)
    assert ts_rank(series, 3).index.equals(index)
    assert cs_rank(series).index.equals(index)
