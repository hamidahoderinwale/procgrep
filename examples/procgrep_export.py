"""Export YOUR local agent sessions as shareable atoms -- safe to send for a study.

This is the privacy boundary made one command. It reads the Claude Code and
Cursor sessions already on your machine, reduces each to ``to_shareable`` form
(atoms + counts + a hashed id), and writes a single JSON file. Prompt text, code,
file paths, and tool arguments never leave: only the abstract action structure
crosses the boundary, so the output is safe to hand to a collaborator pooling
many developers' traces.

What is in the output, per session: ``trace_id`` (hashed), ``agent``,
``atoms`` (the canonical action sequence), and ``metadata`` (counts only).
What is never in it: prompts, code, paths, command text, session titles.

Usage:
    python examples/procgrep_export.py --out my_sessions.json
    python examples/procgrep_export.py --out my_sessions.json --limit 50
    python examples/procgrep_export.py --no-cursor --out cc_only.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
from typing import Any

from procgrep.ingest.adapters.claude_code import load_claude_transcript, to_shareable
from procgrep.ingest.adapters.cursor_vscdb import build_panel_sessions

_CURSOR_DEFAULT = "~/Library/Application Support/Cursor/User/globalStorage/state.vscdb"


def _cursor_shareables(db: Path, *, limit: int) -> list[dict[str, Any]]:
    """Atoms-and-counts payloads from Cursor sessions, derived from the bounded
    panel builder so this stays fast on a multi-GB store and carries no text."""
    out: list[dict[str, Any]] = []
    for session in build_panel_sessions(db, limit=limit):
        atoms: list[str] = []
        for turn in session["turns"]:
            atoms.append("prompt_ai")
            atoms.extend(turn["seq"])
        out.append({
            "trace_id": session["meta"]["id"],
            "agent": "cursor",
            "atoms": atoms,
            "metadata": {"n_atoms": len(atoms), "atom_counts": dict(collections.Counter(atoms))},
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="procgrep_sessions.json", help="output JSON file")
    parser.add_argument("--path", default="~/.claude/projects", help="Claude Code transcript dir")
    parser.add_argument("--cursor", default=_CURSOR_DEFAULT, help="Cursor state.vscdb")
    parser.add_argument("--limit", type=int, default=50, help="most recent N sessions per source")
    parser.add_argument("--no-cursor", action="store_true", help="skip Cursor")
    parser.add_argument("--no-claude-code", action="store_true", help="skip Claude Code")
    args = parser.parse_args()

    shareables: list[dict[str, Any]] = []
    n_cc = 0
    if not args.no_claude_code:
        base = Path(args.path).expanduser()
        files = sorted(glob.glob(str(base / "*" / "*.jsonl"))) + sorted(glob.glob(str(base / "*.jsonl")))
        for file in files[-args.limit :]:
            record = load_claude_transcript(file, anonymize=True)
            payload = to_shareable(record)
            if payload["atoms"]:
                shareables.append(payload)
                n_cc += 1

    n_cursor = 0
    if not args.no_cursor:
        db = Path(args.cursor).expanduser()
        if db.exists():
            for payload in _cursor_shareables(db, limit=args.limit):
                if payload["atoms"]:
                    shareables.append(payload)
                    n_cursor += 1

    if not shareables:
        print("No local sessions found. Nothing to export.")
        return

    # Belt-and-suspenders: the schema is atoms + counts only, but assert no free
    # text fields slipped in before anything is written.
    leaked = [k for s in shareables for k in s if k not in {"trace_id", "agent", "atoms", "metadata"}]
    assert not leaked, f"unexpected fields in export: {set(leaked)}"

    out = Path(args.out).expanduser()
    out.write_text(json.dumps(shareables, indent=2))
    total_atoms = sum(len(s["atoms"]) for s in shareables)
    print(f"exported {len(shareables)} session(s): {n_cc} Claude Code, {n_cursor} Cursor, {total_atoms} atoms")
    print(f"  {out}")
    print("  atoms + counts only -- no prompts, code, paths, or command text. Safe to share.")


if __name__ == "__main__":
    main()
