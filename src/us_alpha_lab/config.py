from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ResearchConfig:
    tickers: list[str]
    start: str
    end: str
    timespan: str = "day"
    multiplier: int = 1
    raw_path: Path = Path("data/raw/daily_bars.parquet")
    factors_path: Path = Path("data/processed/factors.parquet")
    model_path: Path = Path("data/models/latest_model.joblib")
    request_pause_seconds: float = 13.0


def load_config(path: str | Path = "configs/universe.yaml") -> ResearchConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return ResearchConfig(
        tickers=[str(ticker).upper() for ticker in data["tickers"]],
        start=str(data["start"]),
        end=str(data["end"]),
        timespan=str(data.get("timespan", "day")),
        multiplier=int(data.get("multiplier", 1)),
        raw_path=Path(data.get("raw_path", "data/raw/daily_bars.parquet")),
        factors_path=Path(data.get("factors_path", "data/processed/factors.parquet")),
        model_path=Path(data.get("model_path", "data/models/latest_model.joblib")),
        request_pause_seconds=float(data.get("request_pause_seconds", 13)),
    )
