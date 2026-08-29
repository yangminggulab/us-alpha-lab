from __future__ import annotations

import argparse
import html.parser
from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class SP500TableParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_constituents_table = False
        self.in_row = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_by_name = dict(attrs)
        if tag == "table" and attrs_by_name.get("id") == "constituents":
            self.in_constituents_table = True
        elif self.in_constituents_table and tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_row and tag in {"td", "th"}:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_cell and tag in {"td", "th"}:
            self.current_row.append("".join(self.current_cell).strip())
            self.in_cell = False
        elif self.in_constituents_table and self.in_row and tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
            self.in_row = False
        elif self.in_constituents_table and tag == "table":
            self.in_constituents_table = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)


def previous_weekday(today: date | None = None) -> date:
    current = (today or datetime.now(timezone.utc).date()) - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def free_window_start(end: date, years: float = 2.0) -> date:
    return end - timedelta(days=round(365.25 * years))


def normalize_tickers(values: Iterable[str]) -> list[str]:
    tickers = []
    for value in values:
        ticker = value.strip().upper()
        if ticker and ticker != "SYMBOL":
            tickers.append(ticker)
    return tickers


def load_sp500_tickers(url: str = SP500_WIKIPEDIA_URL) -> list[str]:
    request = Request(url, headers={"User-Agent": "us-alpha-lab-sp500-universe/0.1"})
    with urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")

    parser = SP500TableParser()
    parser.feed(text)
    if not parser.rows:
        raise RuntimeError("Could not find the S&P 500 constituents table.")
    return normalize_tickers(row[0] for row in parser.rows if row)


def build_config(tickers: list[str], output_path: Path, start: date, end: date) -> None:
    data = {
        "tickers": tickers,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "timespan": "day",
        "multiplier": 1,
        "raw_path": "data/raw/daily_bars_sp500.parquet",
        "factors_path": "data/processed/factors_sp500.parquet",
        "model_path": "data/models/sp500_model.joblib",
        "request_pause_seconds": 13,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a current S&P 500 Massive universe config.")
    parser.add_argument("--output", type=Path, default=Path("configs/universe_sp500.yaml"))
    parser.add_argument("--source-url", default=SP500_WIKIPEDIA_URL)
    parser.add_argument("--end", help="End date, YYYY-MM-DD. Defaults to previous weekday.")
    parser.add_argument("--history-years", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    end = date.fromisoformat(args.end) if args.end else previous_weekday()
    start = free_window_start(end, years=args.history_years)
    tickers = load_sp500_tickers(args.source_url)
    build_config(tickers, output_path=args.output, start=start, end=end)
    print(f"Saved {len(tickers):,} tickers to {args.output}")
    print(f"Window: {start.isoformat()} -> {end.isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
