from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


def _agg_to_record(ticker: str, agg: Any) -> dict[str, Any]:
    get = agg.get if isinstance(agg, dict) else lambda name, default=None: getattr(agg, name, default)
    timestamp = get("timestamp", get("t"))

    return {
        "ticker": ticker,
        "date": pd.to_datetime(timestamp, unit="ms", utc=True).date(),
        "timestamp": timestamp,
        "open": get("open", get("o")),
        "high": get("high", get("h")),
        "low": get("low", get("l")),
        "close": get("close", get("c")),
        "volume": get("volume", get("v")),
        "vwap": get("vwap", get("vw")),
        "transactions": get("transactions", get("n")),
        "otc": get("otc"),
    }


def fetch_daily_bars(
    tickers: list[str],
    start: str,
    end: str,
    output_path: str | Path,
    multiplier: int = 1,
    timespan: str = "day",
    request_pause_seconds: float = 0,
) -> pd.DataFrame:
    """Download aggregate bars from Massive and save them as parquet."""
    load_dotenv()

    from massive import RESTClient

    client = RESTClient()
    rows: list[dict[str, Any]] = []

    for index, ticker in enumerate(tqdm(tickers, desc="Fetching Massive bars")):
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start,
            to=end,
            limit=50000,
        ):
            rows.append(_agg_to_record(ticker, agg))
        if request_pause_seconds > 0 and index < len(tickers) - 1:
            sleep(request_pause_seconds)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No bars were downloaded. Check tickers, date range, plan, and API key.")

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame
