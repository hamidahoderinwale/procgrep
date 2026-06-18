"""Open the live procedural fingerprint panel on YOUR local Claude Code sessions.

Local-first and privacy-first. It reads the transcripts Claude Code already
writes under ``~/.claude/projects`` on this machine, reduces each to the panel's
session shape (atoms + prompt-anchored turns) with ``build_panel_session``,
injects them into a copy of ``docs/live_fingerprint.html``, and opens it. Nothing
leaves the machine.

Prompt text is shown only when ``--paraphrase`` is given -- a local command that
rewrites a prompt (stripping writing style and identifiers) read on stdin and
written on stdout. Without it, turns show their action structure with no prompt
text, so raw prompts are never surfaced. The only sanctioned thing to share is
``to_shareable(record)`` (atoms + hashed id + counts), not this local HTML.

Usage:
    python examples/procgrep_view.py
    python examples/procgrep_view.py --path ~/.claude/projects --limit 12
    python examples/procgrep_view.py --paraphrase "ollama run llama3 'rewrite, strip style and identifiers, one line:'"
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import tempfile
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from procgrep.ingest.adapters.claude_code import build_panel_session

PANEL = Path(__file__).resolve().parent.parent / "docs" / "live_fingerprint.html"


def _make_paraphraser(cmd: str | None) -> Callable[[str], str] | None:
    """A local rewriter: pipe the prompt through ``cmd`` (stdin -> stdout)."""
    if not cmd:
        return None

    def paraphrase(text: str) -> str:
        try:
            done = subprocess.run(
                cmd, shell=True, input=text, capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        out = done.stdout.strip().splitlines()
        return out[0].strip() if out else ""

    return paraphrase


def _read_lines(path: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for raw in Path(path).read_text(errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            lines.append(parsed)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="~/.claude/projects", help="directory of Claude Code transcripts")
    parser.add_argument("--limit", type=int, default=12, help="most recent N sessions")
    parser.add_argument(
        "--paraphrase",
        default=None,
        help="local command to rewrite prompts (stdin->stdout); omit to hide prompt text",
    )
    args = parser.parse_args()

    base = Path(args.path).expanduser()
    files = sorted(glob.glob(str(base / "*" / "*.jsonl"))) + sorted(glob.glob(str(base / "*.jsonl")))
    paraphrase = _make_paraphraser(args.paraphrase)

    sessions = []
    for file in files[-args.limit :]:
        panel = build_panel_session({"events": _read_lines(file)}, paraphrase=paraphrase)
        if len(panel["turns"]) >= 3:
            sessions.append(panel)

    if not sessions:
        print(f"No usable transcripts under {base}. Opening the bundled demo instead.")
        webbrowser.open(PANEL.as_uri())
        return

    sessions.reverse()  # newest first in the nav
    html = PANEL.read_text()
    inject = f"<script>window.PROCGREP_SESSIONS={json.dumps(sessions)};</script>\n"
    html = html.replace("<script>", inject + "<script>", 1)
    out = Path(tempfile.gettempdir()) / "procgrep_view.html"
    out.write_text(html)
    webbrowser.open(out.as_uri())

    note = "with paraphrased prompts" if paraphrase else "with prompts hidden (pass --paraphrase to show normalized prompts)"
    print(f"opened {len(sessions)} session(s) {note}")
    print(f"  {out}")
    print("  (local file containing your own data — do not share it; use to_shareable() to export)")


if __name__ == "__main__":
    main()
