"""Fingerprint Claude Code sessions, and contrast human+AI working styles.

Claude Code stores each session as a JSONL transcript under
``~/.claude/projects/<project>/<session>.jsonl``. This example ingests those
transcripts with the ``claude-code`` adapter, learns a procedure vocabulary,
and reports two things:

1. The procedural fingerprint -- the mix of atoms and the recurring multi-step
   procedures a session is built from.
2. An *autonomy* read: how many agent actions run between consecutive human
   prompts. A long run means the human lets the agent iterate; a short run
   means tight, interleaved steering.

Only the action structure is read (tool names, a Bash command, event types) --
never message text -- so a transcript can be fingerprinted without exposing its
content.

Usage:
    python examples/claude_code_fingerprint.py                 # your local transcripts
    python examples/claude_code_fingerprint.py --path DIR      # a directory of .jsonl
    python examples/claude_code_fingerprint.py --cursor export.jsonl   # contrast vs Cursor
"""

from __future__ import annotations

import argparse
import collections
import glob
from pathlib import Path

from procgrep import encode, fit_bpe
from procgrep.ingest.adapters.claude_code import (
    ATOM_PROMPT_AI,
    claude_code_adapter,
    load_claude_transcript,
)
from procgrep.types import PROCEDURE_SEPARATOR, Trace

# A tiny built-in session so the example runs even with no local transcripts.
_SYNTHETIC = {
    "trace_id": "synthetic",
    "agent": "demo",
    "events": [
        {"type": "user", "message": {"content": "add a flag and test it"}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Grep", "input": {}},
            {"type": "tool_use", "name": "Read", "input": {}},
            {"type": "tool_use", "name": "Edit", "input": {}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}},
        ]}},
        {"type": "file-history-snapshot"},
    ],
}


def _autonomy_runs(atoms: list[str]) -> list[int]:
    """Lengths of agent-action runs between human prompts."""
    runs, current = [], 0
    for atom in atoms:
        if atom == ATOM_PROMPT_AI:
            runs.append(current)
            current = 0
        else:
            current += 1
    runs.append(current)
    return [r for r in runs if r > 0]


def _load_traces(path: str) -> list[Trace]:
    files = sorted(glob.glob(str(Path(path).expanduser() / "*" / "*.jsonl")))
    files += sorted(glob.glob(str(Path(path).expanduser() / "*.jsonl")))
    traces: list[Trace] = []
    for file in files:
        record = load_claude_transcript(file)
        atoms = claude_code_adapter(record)
        if len(atoms) >= 10:
            traces.append(
                Trace(trace_id=record["trace_id"][:8], agent=record["agent"], atoms=atoms, group="claude_code", metadata={})
            )
    if not traces:
        atoms = claude_code_adapter(_SYNTHETIC)
        traces.append(Trace(trace_id="synthetic", agent="demo", atoms=atoms, group="claude_code", metadata={}))
    return traces


def _report(label: str, traces: list[Trace]) -> None:
    counts = collections.Counter(a for t in traces for a in t.atoms)
    total = sum(counts.values())
    runs = [r for t in traces for r in _autonomy_runs(t.atoms)]
    mean_run = sum(runs) / len(runs) if runs else 0.0
    prompt_share = counts[ATOM_PROMPT_AI] / total * 100 if total else 0.0

    print(f"\n=== {label}: {len(traces)} sessions, {total} atoms ===")
    for atom, count in counts.most_common():
        print(f"  {atom:12} {count / total * 100:5.1f}%")
    print(f"  autonomy: mean {mean_run:.1f} agent-actions per human prompt")
    print(f"  human-turn share: {prompt_share:.1f}%")

    vocab = fit_bpe([t.atoms for t in traces], vocab_size=48, seed=0)
    tokens = vocab.tokens()
    procedures: collections.Counter[str] = collections.Counter()
    for fingerprint in encode(traces, vocab=vocab):
        for token, weight in zip(tokens, fingerprint.distribution(), strict=True):
            if PROCEDURE_SEPARATOR in token and weight > 0:
                procedures[token] += weight
    if procedures:
        print("  top procedures:")
        for token, _ in procedures.most_common(6):
            print("   ", token.replace(PROCEDURE_SEPARATOR, " -> "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="~/.claude/projects", help="dir of Claude Code transcripts")
    parser.add_argument("--cursor", default=None, help="optional cursor-companion export.jsonl to contrast")
    args = parser.parse_args()

    _report("Claude Code", _load_traces(args.path))

    if args.cursor:
        from procgrep.canonicalize import canonicalize
        from procgrep.io import read_jsonl

        cursor = [t for t in canonicalize(list(read_jsonl(args.cursor)), adapter="cursor-companion") if len(t.atoms) >= 10]
        if cursor:
            _report("Cursor companion", cursor)
        print("\nContrast: a higher autonomy run-length means the human lets the agent")
        print("iterate longer before steering; a higher human-turn share means tighter,")
        print("more interleaved direction.")


if __name__ == "__main__":
    main()
