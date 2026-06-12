#!/usr/bin/env bash
# Regenerate the explorer screenshots used in the essay. Runs the Space locally,
# warms the default compare, and renders the compare view with headless chromium.
# Usage: bash scripts/capture_explorer.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="$HOME/Library/Caches/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-mac-arm64/chrome-headless-shell"
tmux kill-session -t capshot 2>/dev/null || true
tmux new-session -d -s capshot "cd $PWD/space && uv run uvicorn app:app --host 127.0.0.1 --port 7861 > /tmp/capshot.log 2>&1"
for i in $(seq 1 40); do curl -sf http://127.0.0.1:7861/datasets >/dev/null 2>&1 && break; sleep 1; done
curl -s -X POST http://127.0.0.1:7861/compare -H 'Content-Type: application/json' \
  -d '{"axis":"outcome","dataset":"nebius/SWE-rebench-openhands-trajectories","left":"resolved","right":"unresolved"}' >/dev/null
sleep 2
"$BIN" --headless --disable-gpu --no-sandbox --hide-scrollbars --force-device-scale-factor=2 \
  --window-size=1180,1180 --virtual-time-budget=9000 \
  --screenshot=docs/figures/explorer_compare.png "http://127.0.0.1:7861/#compare:outcome"
tmux kill-session -t capshot 2>/dev/null || true
echo "wrote docs/figures/explorer_compare.png"
