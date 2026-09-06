from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pyarrow.parquet as pq


REPO = "venvoo/china-a-share-l2-level2-limit-order-book-tick-data"
REVISION = "main"
BASE = f"https://huggingface.co"
API_TREE = f"{BASE}/api/datasets/{REPO}/tree/{REVISION}?recursive=1"
OUT_DIR = Path("data/raw/a_share_l2_hf")
MANIFEST = OUT_DIR / "manifests.parquet"
TREE = OUT_DIR / "hf_tree.json"
STREAM_FILES = {
    "quotes": "行情.parquet",
    "orders": "逐笔委托.parquet",
    "trades": "逐笔成交.parquet",
}


def request_json(url: str) -> tuple[list[dict], str | None]:
    req = Request(url, headers={"User-Agent": "a-share-l2-downloader/0.1"})
    with urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
        return payload, next_link(resp.headers.get("Link", ""))


def next_link(link_header: str) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        match = re.search(r"<([^>]+)>", part)
        if match:
            return match.group(1)
    return None


def resolve_url(path: str) -> str:
    encoded_path = "/".join(quote(part) for part in path.split("/"))
    return f"{BASE}/datasets/{REPO}/resolve/{REVISION}/{encoded_path}"


def download(path: str, dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        print(f"exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(resolve_url(path), headers={"User-Agent": "a-share-l2-downloader/0.1"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(req, timeout=600) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = done / total * 100
                print(
                    f"\r{dest.name}: {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r{dest.name}: {done / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def fetch_tree(overwrite: bool) -> list[dict]:
    if TREE.exists() and not overwrite:
        return json.loads(TREE.read_text())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    url = API_TREE
    entries: list[dict] = []
    while url:
        page, url = request_json(url)
        entries.extend(page)
        print(f"listed {len(entries)} paths")

    TREE.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    return entries


def summarize(entries: list[dict], top_n: int) -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing manifest: {MANIFEST}")

    table = pq.read_table(
        MANIFEST,
        columns=[
            "date",
            "gates_all_green",
            "streams_行情_parquet_bytes",
            "streams_逐笔委托_parquet_bytes",
            "streams_逐笔成交_parquet_bytes",
            "streams_行情_rows",
            "streams_逐笔委托_rows",
            "streams_逐笔成交_rows",
        ],
    )
    df = table.to_pandas()
    paths = {entry["path"] for entry in entries if entry.get("type") == "file"}
    needed = set(STREAM_FILES.values())

    complete_days = []
    for day in df["date"]:
        present = {name for name in needed if f"{day}/{name}" in paths}
        if present == needed:
            complete_days.append(day)

    online = df[df["date"].isin(complete_days)].copy()
    online["total_bytes"] = (
        online["streams_行情_parquet_bytes"]
        + online["streams_逐笔委托_parquet_bytes"]
        + online["streams_逐笔成交_parquet_bytes"]
    )
    online = online.sort_values("total_bytes")

    print(f"manifest days: {len(df)}")
    print(f"online complete days in manifest: {len(online)}")
    print("\nsmallest complete online days:")
    cols = [
        "date",
        "total_bytes",
        "streams_行情_parquet_bytes",
        "streams_逐笔委托_parquet_bytes",
        "streams_逐笔成交_parquet_bytes",
    ]
    for _, row in online.head(top_n)[cols].iterrows():
        print(
            f"{row['date']}: total={row['total_bytes'] / 1e9:.2f} GB "
            f"quotes={row['streams_行情_parquet_bytes'] / 1e9:.2f} "
            f"orders={row['streams_逐笔委托_parquet_bytes'] / 1e9:.2f} "
            f"trades={row['streams_逐笔成交_parquet_bytes'] / 1e9:.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the A-share L2 Hugging Face archive.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--download-day", help="YYYYMMDD trading day to download.")
    parser.add_argument(
        "--stream",
        choices=["quotes", "orders", "trades", "all"],
        action="append",
        help="Stream to download for --download-day. May be repeated.",
    )
    args = parser.parse_args()

    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        download("manifests.parquet", MANIFEST, args.overwrite)
        entries = fetch_tree(args.overwrite)
        summarize(entries, args.top_n)

        if args.download_day:
            streams = args.stream or ["quotes"]
            selected = list(STREAM_FILES) if "all" in streams else streams
            for stream in selected:
                filename = STREAM_FILES[stream]
                download(f"{args.download_day}/{filename}", OUT_DIR / args.download_day / filename, args.overwrite)

    except (HTTPError, URLError, OSError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
