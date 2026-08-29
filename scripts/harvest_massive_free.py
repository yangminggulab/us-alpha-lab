from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_CONFIG = Path("configs/universe_free_50.yaml")
DEFAULT_CHECKPOINT_DIR = Path("data/raw/massive_free_tier_shards")
MASSIVE_TICKERS_URL = "https://api.massive.com/v3/reference/tickers"


@dataclass(frozen=True)
class FreeTierSettings:
    calls_per_minute: float = 5.0
    safety_seconds: float = 1.0
    history_years: float = 2.0

    @property
    def pause_seconds(self) -> float:
        return (60.0 / self.calls_per_minute) + self.safety_seconds


class RateLimiter:
    def __init__(self, pause_seconds: float) -> None:
        self.pause_seconds = pause_seconds
        self._last_call_started_at: float | None = None

    def wait(self) -> None:
        if self._last_call_started_at is not None:
            elapsed = time.monotonic() - self._last_call_started_at
            remaining = self.pause_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call_started_at = time.monotonic()


def normalize_tickers(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    tickers: list[str] = []
    for value in values:
        ticker = value.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def parse_ticker_text(text: str | None) -> list[str]:
    if not text:
        return []
    return normalize_tickers(text.replace("\n", ",").split(","))


def load_ticker_file(path: Path) -> list[str]:
    return normalize_tickers(path.read_text(encoding="utf-8").replace("\n", ",").split(","))


def parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    lowered = value.lower()
    today = datetime.now(timezone.utc).date()
    if lowered == "today":
        return today
    if lowered == "yesterday":
        return today - timedelta(days=1)
    if lowered in {"previous-weekday", "last-weekday"}:
        current = today - timedelta(days=1)
        while current.weekday() >= 5:
            current -= timedelta(days=1)
        return current
    return date.fromisoformat(value)


def free_tier_start(end: date, history_years: float) -> date:
    return end - timedelta(days=round(365.25 * history_years))


def resolve_window(
    cfg: Any,
    start: str | None,
    end: str | None,
    rolling_free_window: bool,
    settings: FreeTierSettings,
) -> tuple[str, str]:
    end_date = parse_date(end) or parse_date(cfg.end) or datetime.now(timezone.utc).date()
    if rolling_free_window:
        start_date = parse_date(start) or free_tier_start(end_date, settings.history_years)
    else:
        start_date = parse_date(start) or parse_date(cfg.start)
    if start_date is None:
        start_date = free_tier_start(end_date, settings.history_years)
    if start_date > end_date:
        raise ValueError(f"start date {start_date} is after end date {end_date}")
    return start_date.isoformat(), end_date.isoformat()


def ticker_shard_path(checkpoint_dir: Path, ticker: str) -> Path:
    safe_ticker = ticker.replace("/", "_")
    return checkpoint_dir / f"{safe_ticker}.parquet"


def url_with_api_key(url: str, api_key: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["apiKey"] = api_key
    return urlunparse(parsed._replace(query=urlencode(query)))


def read_json_url(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "us-alpha-lab-free-tier-harvest/0.1"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_active_ticker_universe(
    api_key: str,
    limiter: RateLimiter,
    market: str,
    allowed_types: set[str],
    output_path: Path | None,
    max_tickers: int | None,
) -> list[str]:
    params = {
        "market": market,
        "active": "true",
        "order": "asc",
        "limit": "1000",
        "sort": "ticker",
        "apiKey": api_key,
    }
    url: str | None = f"{MASSIVE_TICKERS_URL}?{urlencode(params)}"
    records: list[dict[str, Any]] = []

    print(f"Discovering active {market} tickers from Massive reference data...")
    while url:
        limiter.wait()
        try:
            payload = read_json_url(url)
        except HTTPError as exc:
            if exc.code == 429:
                retry_after = float(exc.headers.get("Retry-After", limiter.pause_seconds))
                time.sleep(retry_after)
                continue
            raise
        for item in payload.get("results", []):
            ticker_type = str(item.get("type", "")).upper()
            if allowed_types and ticker_type not in allowed_types:
                continue
            records.append(item)
            if max_tickers is not None and len(records) >= max_tickers:
                url = None
                break
        if url is None:
            break
        next_url = payload.get("next_url")
        url = url_with_api_key(next_url, api_key) if next_url else None

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(output_path, index=False)

    tickers = normalize_tickers(str(record["ticker"]) for record in records if record.get("ticker"))
    print(f"Discovered {len(tickers):,} active tickers.")
    return tickers


def shard_covers_window(path: Path, start: str, end: str) -> bool:
    coverage = shard_date_bounds(path)
    if coverage is None:
        return False
    first_date, last_date = coverage
    return first_date <= pd.Timestamp(start) and last_date >= pd.Timestamp(end)


def shard_date_bounds(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["date"])
    if frame.empty:
        return None
    dates = pd.to_datetime(frame["date"])
    return dates.min(), dates.max()


def resolve_missing_start(path: Path, start: str, end: str) -> str | None:
    coverage = shard_date_bounds(path)
    if coverage is None:
        return start

    first_date, last_date = coverage
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)
    if first_date <= requested_start and last_date >= requested_end:
        return None
    if first_date <= requested_start and last_date < requested_end:
        next_date = (last_date + pd.Timedelta(days=1)).date().isoformat()
        return next_date if pd.Timestamp(next_date) <= requested_end else None
    return start


def fetch_ticker_daily_bars(
    client: Any,
    ticker: str,
    start: str,
    end: str,
    multiplier: int,
    timespan: str,
) -> pd.DataFrame:
    from us_alpha_lab.massive_data import _agg_to_record

    rows = [
        _agg_to_record(ticker, agg)
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start,
            to=end,
            limit=50000,
        )
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def write_parquet_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def seed_checkpoint_shards(seed_paths: list[Path], checkpoint_dir: Path, force: bool = False) -> int:
    seeded = 0
    for seed_path in seed_paths:
        if not seed_path.exists():
            continue
        frame = pd.read_parquet(seed_path)
        if frame.empty or "ticker" not in frame.columns:
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        for ticker, group in frame.groupby("ticker"):
            shard_path = ticker_shard_path(checkpoint_dir, str(ticker))
            if shard_path.exists() and not force:
                continue
            write_parquet_atomically(group.sort_values("date").reset_index(drop=True), shard_path)
            seeded += 1
    if seeded:
        print(f"Seeded {seeded:,} checkpoint shards from existing parquet data.")
    return seeded


def merge_shard_frame(path: Path, new_frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frames = []
    if path.exists():
        frames.append(pd.read_parquet(path))
    frames.append(new_frame)

    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return frame.reset_index(drop=True)


def combine_shards(checkpoint_dir: Path, output_path: Path) -> pd.DataFrame:
    shard_paths = sorted(checkpoint_dir.glob("*.parquet"))
    if not shard_paths:
        raise RuntimeError(f"No shard files found under {checkpoint_dir}")
    frames = [pd.read_parquet(path) for path in shard_paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    frame = frame.reset_index(drop=True)
    write_parquet_atomically(frame, output_path)
    return frame


def resolve_allowed_types(args: argparse.Namespace) -> set[str]:
    if args.include_all_types:
        return set()
    return set(parse_ticker_text(args.ticker_types))


def resolve_tickers(
    args: argparse.Namespace,
    cfg: Any,
    api_key: str | None,
    limiter: RateLimiter,
) -> list[str]:
    tickers = parse_ticker_text(args.tickers)
    if args.tickers_file is not None:
        tickers.extend(load_ticker_file(args.tickers_file))
    if args.discover_active:
        if api_key is None:
            raise RuntimeError("MASSIVE_API_KEY is required when --discover-active is used.")
        tickers.extend(
            fetch_active_ticker_universe(
                api_key=api_key,
                limiter=limiter,
                market=args.market,
                allowed_types=resolve_allowed_types(args),
                output_path=args.reference_output,
                max_tickers=args.max_tickers,
            )
        )
    if not tickers:
        tickers = cfg.tickers
    tickers = normalize_tickers(tickers)
    if args.max_tickers is not None:
        tickers = tickers[: args.max_tickers]
    if not tickers:
        raise ValueError("No tickers were provided by --tickers, --tickers-file, or config.")
    return tickers


def harvest(args: argparse.Namespace) -> int:
    from us_alpha_lab.config import load_config

    load_dotenv()
    api_key = os.getenv("MASSIVE_API_KEY")
    if not args.dry_run and not api_key:
        raise RuntimeError("MASSIVE_API_KEY is not set. Add it to .env or export it before running.")

    cfg = load_config(args.config)
    settings = FreeTierSettings(
        calls_per_minute=args.calls_per_minute,
        safety_seconds=args.safety_seconds,
        history_years=args.history_years,
    )
    start, end = resolve_window(cfg, args.start, args.end, args.rolling_free_window, settings)
    limiter = RateLimiter(settings.pause_seconds)
    tickers = (
        []
        if args.dry_run and args.discover_active and not args.tickers and args.tickers_file is None
        else resolve_tickers(args, cfg, api_key, limiter)
    )
    output_path = args.output or cfg.raw_path
    checkpoint_dir = args.checkpoint_dir

    if args.dry_run:
        if args.discover_active and not tickers:
            cap = f", capped at {args.max_tickers}" if args.max_tickers is not None else ""
            print(f"Tickers: active Massive {args.market} universe{cap}")
        else:
            print(f"Tickers: {len(tickers)}")
        print(f"Window: {start} -> {end}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Combined output: {output_path}")
        if args.seed_from:
            print(f"Seed from: {', '.join(str(path) for path in args.seed_from)}")
        print(f"Pause: {settings.pause_seconds:.2f}s between ticker requests")
        return 0

    seed_checkpoint_shards(args.seed_from, checkpoint_dir, force=args.force_seed)

    if not args.combine_only:
        from massive import RESTClient

        client = RESTClient()
        failures: list[tuple[str, str]] = []

        for ticker in tqdm(tickers, desc="Harvesting Massive free tier"):
            shard_path = ticker_shard_path(checkpoint_dir, ticker)
            fetch_start = start if args.force else resolve_missing_start(shard_path, start, end)
            if fetch_start is None:
                continue

            try:
                limiter.wait()
                frame = fetch_ticker_daily_bars(
                    client=client,
                    ticker=ticker,
                    start=fetch_start,
                    end=end,
                    multiplier=cfg.multiplier,
                    timespan=cfg.timespan,
                )
                if frame.empty:
                    failures.append((ticker, "no rows returned"))
                    continue
                shard_frame = frame if args.force else merge_shard_frame(shard_path, frame, start, end)
                write_parquet_atomically(shard_frame, shard_path)
            except Exception as exc:  # noqa: BLE001
                failures.append((ticker, str(exc)))

        if failures:
            failure_path = checkpoint_dir / "failures.csv"
            failure_frame = pd.DataFrame(failures, columns=["ticker", "error"])
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_frame.to_csv(failure_path, index=False)
            print(f"Saved {len(failures)} failures to {failure_path}")

    combined = combine_shards(checkpoint_dir, output_path)
    print(f"Saved {len(combined):,} rows for {combined['ticker'].nunique():,} tickers to {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest Massive Stocks Basic free-tier daily bars with resume checkpoints."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Research config path.")
    parser.add_argument("--tickers", help="Comma-separated tickers. Overrides config if provided.")
    parser.add_argument("--tickers-file", type=Path, help="Text/CSV file with tickers.")
    parser.add_argument(
        "--discover-active",
        action="store_true",
        help="Fetch active tickers from Massive reference data before downloading bars.",
    )
    parser.add_argument("--market", default="stocks", help="Massive reference market for discovery.")
    parser.add_argument(
        "--ticker-types",
        default="CS,ETF",
        help="Comma-separated Massive ticker types to keep when discovering.",
    )
    parser.add_argument(
        "--include-all-types",
        action="store_true",
        help="Keep every discovered ticker type instead of filtering to --ticker-types.",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=Path("data/raw/massive_active_tickers.csv"),
        help="Optional CSV path for discovered reference records.",
    )
    parser.add_argument("--max-tickers", type=int, help="Cap ticker count for smoke tests.")
    parser.add_argument("--start", help="Start date, YYYY-MM-DD.")
    parser.add_argument(
        "--end",
        help="End date, YYYY-MM-DD, today, yesterday, or previous-weekday.",
    )
    parser.add_argument(
        "--rolling-free-window",
        action="store_true",
        help="Use the maximum free historical window ending at --end or the config end date.",
    )
    parser.add_argument("--history-years", type=float, default=2.0, help="Free history window.")
    parser.add_argument("--calls-per-minute", type=float, default=5.0, help="Massive free-tier rate.")
    parser.add_argument("--safety-seconds", type=float, default=1.0, help="Extra pause per request.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Per-ticker parquet checkpoint directory.",
    )
    parser.add_argument(
        "--seed-from",
        type=Path,
        action="append",
        default=[],
        help="Existing combined parquet to split into per-ticker checkpoint shards before fetching.",
    )
    parser.add_argument(
        "--force-seed",
        action="store_true",
        help="Overwrite checkpoint shards when seeding from existing parquet data.",
    )
    parser.add_argument("--output", type=Path, help="Combined parquet output path.")
    parser.add_argument("--force", action="store_true", help="Re-download existing complete shards.")
    parser.add_argument("--combine-only", action="store_true", help="Only combine existing shards.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without API calls.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return harvest(args)


if __name__ == "__main__":
    raise SystemExit(main())
