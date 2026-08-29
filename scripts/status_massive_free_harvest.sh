#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-runs/massive_free_harvest}"
PID_FILE="${PID_FILE:-$RUN_DIR/harvest.pid}"
LATEST_LOG_FILE="${LATEST_LOG_FILE:-$RUN_DIR/latest.log}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No Massive free-tier harvest PID file found."
else
  PID="$(cat "$PID_FILE")"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "Status: running"
    echo "PID: $PID"
  else
    echo "Status: not running"
    echo "Last PID: $PID"
  fi
fi

if [[ -f "$LATEST_LOG_FILE" ]]; then
  LOG_FILE="$(cat "$LATEST_LOG_FILE")"
  echo "Log: $LOG_FILE"
  if [[ -f "$LOG_FILE" ]]; then
    echo
    echo "Last 40 log lines:"
    tail -n 40 "$LOG_FILE"
  fi
fi
