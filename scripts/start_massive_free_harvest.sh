#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/universe_free_50.yaml}"
OUTPUT="${OUTPUT:-data/raw/daily_bars_free_50.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/raw/massive_free_tier_shards}"
END_DATE="${END_DATE:-previous-weekday}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-runs/massive_free_harvest}"
LOG_DIR="${LOG_DIR:-logs}"
SEED_FROM="${SEED_FROM:-data/raw/daily_bars_free_50.parquet}"
DISCOVER_ACTIVE="${DISCOVER_ACTIVE:-0}"
MAX_TICKERS="${MAX_TICKERS:-}"
PID_FILE="${PID_FILE:-$RUN_DIR/harvest.pid}"
LATEST_LOG_FILE="${LATEST_LOG_FILE:-$RUN_DIR/latest.log}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE")"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Massive free-tier harvest is already running."
    echo "PID: $EXISTING_PID"
    echo "Log: $(cat "$LATEST_LOG_FILE" 2>/dev/null || true)"
    exit 0
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/massive_free_harvest_$STAMP.log"

COMMAND=(
  "$PYTHON_BIN"
  -u
  scripts/harvest_massive_free.py
  --config "$CONFIG"
  --rolling-free-window
  --end "$END_DATE"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --output "$OUTPUT"
)

if [[ "$DISCOVER_ACTIVE" == "1" || "$DISCOVER_ACTIVE" == "true" ]]; then
  COMMAND+=(--discover-active)
fi

if [[ -n "$MAX_TICKERS" ]]; then
  COMMAND+=(--max-tickers "$MAX_TICKERS")
fi

if [[ -n "$SEED_FROM" && -f "$SEED_FROM" ]]; then
  COMMAND+=(--seed-from "$SEED_FROM")
fi

if command -v caffeinate >/dev/null 2>&1; then
  RUNNER=(caffeinate -dimsu -- "${COMMAND[@]}" "$@")
else
  RUNNER=("${COMMAND[@]}" "$@")
fi

{
  echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "Working dir: $ROOT_DIR"
  echo "Command: ${RUNNER[*]}"
  echo
} >"$LOG_FILE"

nohup bash -c '
"$@"
STATUS="$?"
echo
echo "Finished at: $(date "+%Y-%m-%d %H:%M:%S %z")"
echo "Exit code: $STATUS"
exit "$STATUS"
' harvest-runner "${RUNNER[@]}" >>"$LOG_FILE" 2>&1 &
PID="$!"
echo "$PID" >"$PID_FILE"
echo "$LOG_FILE" >"$LATEST_LOG_FILE"

echo "Started Massive free-tier harvest in the background."
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Output: $OUTPUT"
if [[ "$#" -gt 0 ]]; then
  echo "Extra args: $*"
fi
echo
echo "Check progress with:"
echo "  bash scripts/status_massive_free_harvest.sh"
