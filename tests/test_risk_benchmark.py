from __future__ import annotations

import pandas as pd

from us_alpha_lab.benchmark import benchmark_diagnostics, equal_weight_benchmark
from us_alpha_lab.risk import beta_to_benchmark, return_metrics


def test_return_metrics_and_beta() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.01])
    benchmark = pd.Series([0.005, -0.01, 0.02, 0.01])
    metrics = return_metrics(returns)

    assert "calmar_ratio" in metrics
    assert beta_to_benchmark(returns, benchmark) > 0


def test_equal_weight_benchmark_and_diagnostics() -> None:
    frame = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-01-01", "close": 100},
            {"ticker": "AAA", "date": "2024-01-02", "close": 101},
            {"ticker": "BBB", "date": "2024-01-01", "close": 200},
            {"ticker": "BBB", "date": "2024-01-02", "close": 198},
        ]
    )
    benchmark = equal_weight_benchmark(frame)
    diagnostics = benchmark_diagnostics(benchmark, benchmark)

    assert len(benchmark) == 1
    assert diagnostics["benchmark_correlation"] == 0.0
