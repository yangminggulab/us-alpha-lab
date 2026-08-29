#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LABEL="${LABEL:-com.us-alpha-lab.massive-free-harvest}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/massive_free_harvest}"
LATEST_LOG_FILE="$RUN_DIR/latest_launchd.log"

launchctl print "gui/$(id -u)/$LABEL" | sed -n '1,80p' || true

if [[ -f "$LATEST_LOG_FILE" ]]; then
  LOG_FILE="$(cat "$LATEST_LOG_FILE")"
  echo
  echo "Log: $LOG_FILE"
  if [[ -f "$LOG_FILE" ]]; then
    echo
    echo "Last 60 log lines:"
    tail -n 60 "$LOG_FILE"
  fi
fi
