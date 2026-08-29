#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LABEL="${LABEL:-com.us-alpha-lab.massive-free-harvest}"
CONFIG="${CONFIG:-configs/universe_free_50.yaml}"
OUTPUT="${OUTPUT:-data/raw/daily_bars_free_500.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/raw/massive_free_500_shards}"
END_DATE="${END_DATE:-previous-weekday}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/massive_free_harvest}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
SEED_FROM="${SEED_FROM:-data/raw/daily_bars_free_50.parquet}"
DISCOVER_ACTIVE="${DISCOVER_ACTIVE:-0}"
MAX_TICKERS="${MAX_TICKERS:-}"
PLIST_PATH="$RUN_DIR/$LABEL.plist"
LOG_FILE="$LOG_DIR/massive_free_harvest_launchd_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

COMMAND=(
  "$PYTHON_BIN"
  -u
  "$ROOT_DIR/scripts/harvest_massive_free.py"
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

COMMAND+=("$@")

if command -v caffeinate >/dev/null 2>&1; then
  RUNNER=(/usr/bin/caffeinate -dimsu -- "${COMMAND[@]}")
else
  RUNNER=("${COMMAND[@]}")
fi

RUN_COMMAND="cd $(printf "%q" "$ROOT_DIR"); exec"
for part in "${RUNNER[@]}"; do
  RUN_COMMAND="$RUN_COMMAND $(printf "%q" "$part")"
done

cat >"$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>$RUN_COMMAND</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>$LOG_FILE</string>
  <key>StandardErrorPath</key>
  <string>$LOG_FILE</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "$LOG_FILE" >"$RUN_DIR/latest_launchd.log"
echo "Started launchd Massive free-tier harvest."
echo "Label: $LABEL"
echo "Plist: $PLIST_PATH"
echo "Log: $LOG_FILE"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Output: $OUTPUT"
echo
echo "Check with:"
echo "  launchctl print gui/$(id -u)/$LABEL"
echo "  tail -f $LOG_FILE"
