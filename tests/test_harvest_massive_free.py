from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "harvest_massive_free.py"
SPEC = importlib.util.spec_from_file_location("harvest_massive_free", SCRIPT_PATH)
harvest_massive_free = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = harvest_massive_free
SPEC.loader.exec_module(harvest_massive_free)


def test_normalize_tickers_deduplicates_and_uppercases() -> None:
    assert harvest_massive_free.normalize_tickers([" aapl ", "AAPL", "msft"]) == ["AAPL", "MSFT"]


def test_free_tier_start_uses_two_year_window() -> None:
    assert harvest_massive_free.free_tier_start(date(2026, 8, 14), 2.0).isoformat() == "2024-08-14"


def test_shard_covers_window(tmp_path: Path) -> None:
    shard = tmp_path / "AAPL.parquet"
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-08-15", "2026-08-14"]),
            "close": [1.0, 2.0],
        }
    )
    frame.to_parquet(shard, index=False)

    assert harvest_massive_free.shard_covers_window(shard, "2024-08-15", "2026-08-14")
    assert not harvest_massive_free.shard_covers_window(shard, "2024-08-14", "2026-08-14")


def test_resolve_missing_start_only_fetches_new_dates(tmp_path: Path) -> None:
    shard = tmp_path / "AAPL.parquet"
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-08-15", "2026-08-13"]),
            "close": [1.0, 2.0],
        }
    )
    frame.to_parquet(shard, index=False)

    assert harvest_massive_free.resolve_missing_start(shard, "2024-08-15", "2026-08-14") == "2026-08-14"
    assert harvest_massive_free.resolve_missing_start(shard, "2024-08-15", "2026-08-13") is None


def test_url_with_api_key_replaces_or_adds_key() -> None:
    url = "https://api.massive.com/v3/reference/tickers?cursor=abc&apiKey=old"

    assert harvest_massive_free.url_with_api_key(url, "new").endswith("cursor=abc&apiKey=new")


def test_seed_checkpoint_shards_reuses_existing_panel(tmp_path: Path) -> None:
    seed = tmp_path / "existing.parquet"
    checkpoint_dir = tmp_path / "shards"
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "close": [1.0, 2.0, 3.0],
        }
    )
    frame.to_parquet(seed, index=False)

    seeded = harvest_massive_free.seed_checkpoint_shards([seed], checkpoint_dir)

    assert seeded == 2
    assert (checkpoint_dir / "AAPL.parquet").exists()
    assert (checkpoint_dir / "MSFT.parquet").exists()
