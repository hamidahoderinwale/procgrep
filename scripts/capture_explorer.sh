#!/usr/bin/env bash
# Regenerate the explorer screenshots used in the essay. Runs the Space locally,
# warms the default compare, and renders the compare view with headless chromium.
# Usage: bash scripts/capture_explorer.sh
# Requires: CHROME_BIN (or a Playwright chromium headless shell install).
set -euo pipefail
cd "$(dirname "$0")/.."

## Resolve the headless browser: CHROME_BIN wins, else newest Playwright shell
BIN="${CHROME_BIN:-$(ls -d "$HOME"/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-mac-arm64/chrome-headless-shell 2>/dev/null | sort -V | tail -1)}"
[ -x "${BIN:-}" ] || { echo "no headless chromium: set CHROME_BIN or run 'npx playwright install chromium'" >&2; exit 1; }

## Serve the Space locally, kill it on exit
(cd space && uv run uvicorn app:app --host 127.0.0.1 --port 7861 > /tmp/capshot.log 2>&1) &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do curl -sf http://127.0.0.1:7861/datasets >/dev/null 2>&1 && break; sleep 1; done

## Warm the default compare so the screenshot shows results, not a spinner
curl -s -X POST http://127.0.0.1:7861/compare -H 'Content-Type: application/json' \
  -d '{"axis":"outcome","dataset":"nebius/SWE-rebench-openhands-trajectories","left":"resolved","right":"unresolved"}' >/dev/null
sleep 2

"$BIN" --headless --disable-gpu --no-sandbox --hide-scrollbars --force-device-scale-factor=2 \
  --window-size=1180,1180 --virtual-time-budget=9000 \
  --screenshot=docs/figures/explorer_compare.png "http://127.0.0.1:7861/#compare:outcome"
echo "wrote docs/figures/explorer_compare.png"
