"""Live procedural fingerprint: the rolling procedure mix as a session unfolds.

A demonstration of "procedural liveness" -- that an agent's procedure is a
live, measurable signal, not just a post-hoc artifact. It streams a trace,
keeps a fixed-N window of recent atoms, and renders the window's procedure mix
as Tufte-style sparklines plus an autonomy readout. This is the lightweight,
zero-infra sibling of the D3 liveness panel; the actionable alert/circuit-
breaker is a separate concern handled in `07_live_monitor.py` (guard mode).

Privacy: the only thing read or shown is the abstract atom representation. The
bundled exemplar (`data/live_demo.jsonl`) is one real session reduced to atoms
plus a hashed id -- no paths, file names, prompt text, or raw session id, so
the source is unrecoverable. The reduction to atoms *is* the obfuscation.
Pointed at your own session with ``--transcript`` it stays local and is shown
the same way: atoms and percentages only, a hashed label, nothing of the source.

Design decisions, grounded in the data:

- Fixed-N *atom* window, not a time window. Agent activity is bursty (many tool
  calls in seconds after a prompt, then a human gap); procedure is defined by
  the action sequence, so a last-N-atoms window is always informative where a
  time window would flood then go empty.
- Adaptive lanes. Only atoms with material mass get a row, in procedural order
  (explore, handoff, generate, verify, residual). A source dominated by one
  atom (e.g. a prompt-only export) is flagged as too sparse to fingerprint
  rather than drawn as a misleading flat strip.
- Autonomy (agent actions per human prompt) is shown only when the source is
  interactive (has prompt_ai).
- Sparklines are self-normalized per lane (the trend); the datum at the right is
  the current share (the level). Reading top-to-bottom is the procedural arc.

Run from the repo root:

    python examples/python/16_live_fingerprint.py                 # bundled exemplar
    python examples/python/16_live_fingerprint.py --realtime      # animate it
    python examples/python/16_live_fingerprint.py --transcript path/to/session.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

from procgrep.ingest.adapters.claude_code import (
    ATOM_PROMPT_AI,
    claude_code_adapter,
    load_claude_transcript,
)

_BUNDLED = Path(__file__).resolve().parent / "data" / "live_demo.jsonl"
_BLOCKS = "▁▂▃▄▅▆▇█"
_PHASE_ORDER = [
    "search_repo",
    "read_file",
    "localize",
    ATOM_PROMPT_AI,
    "edit",
    "create_file",
    "delete_file",
    "run_test",
    "submit",
    "think",
    "error",
    "other",
]


def _spark(values: list[float]) -> str:
    """A self-normalized unicode sparkline (the trend)."""
    if not values:
        return ""
    hi = max(values)
    if hi <= 0:
        return _BLOCKS[0] * len(values)
    return "".join(_BLOCKS[min(len(_BLOCKS) - 1, int(v / hi * (len(_BLOCKS) - 1)))] for v in values)


def _load(transcript: str | None) -> tuple[list[str], str]:
    """Return (atoms, hashed label). Source is never exposed -- atoms only."""
    if transcript:
        record = load_claude_transcript(transcript)  # anonymize=True: hashed id, no paths
        return claude_code_adapter(record), str(record["trace_id"])[:8]
    record = json.loads(_BUNDLED.read_text().splitlines()[0])
    return list(record["atoms"]), str(record.get("trace_id", "exemplar"))[:8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", default=None, help="a local Claude Code .jsonl (default: bundled exemplar)")
    parser.add_argument("--window", type=int, default=40, help="atoms in the rolling window")
    parser.add_argument("--stride", type=int, default=5, help="atoms between snapshots")
    parser.add_argument("--history", type=int, default=16, help="snapshots shown per sparkline")
    parser.add_argument("--realtime", action="store_true", help="animate in place")
    parser.add_argument("--step-delay", type=float, default=0.15, help="seconds/snapshot when --realtime")
    parser.add_argument("--min-share", type=float, default=0.03, help="min mean share for a lane")
    args = parser.parse_args()

    atoms, label = _load(args.transcript)
    if len(atoms) < args.window:
        print(f"trace has {len(atoms)} atoms; need >= --window ({args.window})")
        sys.exit(1)

    window: deque[str] = deque(maxlen=args.window)
    history: dict[str, deque[float]] = {a: deque(maxlen=args.history) for a in _PHASE_ORDER}
    autonomy_hist: deque[float] = deque(maxlen=args.history)

    def snapshot() -> None:
        total = len(window) or 1
        prompts = sum(1 for atom in window if atom == ATOM_PROMPT_AI)
        for a in _PHASE_ORDER:
            history[a].append(window.count(a) / total)
        autonomy_hist.append((total - prompts) / prompts if prompts else 0.0)

    def render(step: int) -> str:
        total = len(window) or 1
        shares = {a: window.count(a) / total for a in _PHASE_ORDER}
        lines = [f"procgrep · live fingerprint · {label} · window {args.window} · atom {step}"]
        material = [a for a in _PHASE_ORDER if sum(history[a]) / max(len(history[a]), 1) >= args.min_share]
        top = max(material, key=lambda a: shares[a], default=None)
        if top and shares[top] > 0.9:
            lines.append(f"  {top} {shares[top] * 100:.0f}% — single-atom source; too sparse to fingerprint")
            return "\n".join(lines)
        for a in material:
            name = "bash/other" if a == "other" else a
            lines.append(f"  {name:12} {_spark(list(history[a]))}  {shares[a] * 100:4.0f}%")
        if ATOM_PROMPT_AI in material:
            current = autonomy_hist[-1] if autonomy_hist else 0.0
            lines.append(f"  {'autonomy':12} {_spark(list(autonomy_hist))}  {current:4.1f} /prompt")
        return "\n".join(lines)

    prev_lines = 0
    for i, atom in enumerate(atoms, start=1):
        window.append(atom)
        if i % args.stride == 0 or i == len(atoms):
            snapshot()
            frame = render(i)
            if args.realtime:
                if prev_lines:
                    sys.stdout.write(f"\033[{prev_lines}A\033[J")
                sys.stdout.write(frame + "\n")
                sys.stdout.flush()
                prev_lines = frame.count("\n") + 1
                time.sleep(args.step_delay)
    if not args.realtime:
        print(render(len(atoms)))


if __name__ == "__main__":
    main()
