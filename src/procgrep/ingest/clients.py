"""Locate the coding clients installed on this machine.

The other ingest paths start from a dataset id or a JSONL path someone already
produced. Interactive sources have no such starting point: the sessions are
already on disk, in a client-specific place, and until now every caller
rediscovered that place by pasting the same platform literal. This module is
the one home for that knowledge, so ``procgrep cursor`` can run with no
arguments::

    discover_clients()      -> [LocalClient]     everything found here
    find_client("cursor")   -> LocalClient|None  one source, by adapter family

Nothing is read here beyond ``stat``: discovery answers *where*, the adapters
answer *what*, and neither knows about the other.

Design decisions (benefit / price):

1. Candidate roots are existence-checked, never assumed. Each platform's
   application-data root is globbed for client directories and a candidate is
   kept only when the session store itself is present.
   Benefit: a wrong or stale root costs nothing -- it yields no candidate
   instead of a path that fails later inside SQLite. The same code runs on all
   three platforms without branching at the call site.
   Price: a client installed somewhere non-standard is invisible, so every
   caller still needs an explicit-path override.

2. Cursor variants are found by glob (``Cursor``, ``Cursor Nightly``, ...)
   rather than an enumerated list.
   Benefit: a new channel or a rename is picked up with no code change.
   Price: the glob can match an unrelated directory whose name starts the same
   way; the ``User/globalStorage/state.vscdb`` check is what excludes it (on a
   real machine this is what separates ``Cursor`` from, say,
   ``cursor-pkl-extension``).

3. Newest-first ordering by mtime, and ``size_bytes`` carried on the result.
   Benefit: the CLI can say "16.0 GB" before it starts scanning, so a slow
   first run on a large store reads as expected rather than hung.
   Price: ``stat`` on every candidate, which is negligible next to the read.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

## Where each platform keeps per-application data. Cursor is a VS Code fork and
## follows the VS Code layout, so one root per platform covers every variant.
_APP_DATA_ROOTS: dict[str, tuple[str, ...]] = {
    "darwin": ("~/Library/Application Support",),
    "win32": ("~/AppData/Roaming", "%APPDATA%"),
}
_LINUX_ROOTS = ("~/.config", "~/.var/app")  # plain install, then Flatpak


def _app_data_roots() -> list[Path]:
    """Existing application-data roots for this platform, in search order."""
    patterns = _APP_DATA_ROOTS.get(sys.platform, _LINUX_ROOTS)
    roots = []
    for pattern in patterns:
        root = Path(pattern.replace("%APPDATA%", "~/AppData/Roaming")).expanduser()
        if root.is_dir():
            roots.append(root)
    return roots


@dataclass(frozen=True)
class LocalClient:
    """One session store found on this machine.

    ``adapter`` is the registered adapter name that reads ``path``, so a caller
    can go straight from a discovery result to ``canonicalize`` without a
    lookup table of its own.
    """

    name: str
    adapter: str
    path: Path
    size_bytes: int

    @property
    def size_label(self) -> str:
        """Human size of the store, for a one-line terminal report."""
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB"):
            if size < 1024:
                return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


def _dir_size(path: Path) -> int:
    """Total size of the files directly under a session directory, best effort."""
    total = 0
    for file in path.rglob("*.jsonl"):
        try:
            total += file.stat().st_size
        except OSError:
            continue
    return total


def _cursor_clients() -> list[LocalClient]:
    """Cursor-family stores: ``<app data>/<Variant>/User/globalStorage/state.vscdb``."""
    found = []
    for root in _app_data_roots():
        for candidate in sorted(root.glob("[Cc]ursor*")):
            db = candidate / "User" / "globalStorage" / "state.vscdb"
            if not db.is_file():
                continue
            try:
                size = db.stat().st_size
            except OSError:
                continue
            found.append(
                LocalClient(name=candidate.name, adapter="cursor-vscdb", path=db, size_bytes=size)
            )
    return found


def _claude_code_clients() -> list[LocalClient]:
    """Claude Code transcripts: ``~/.claude/projects/<project>/<session>.jsonl``."""
    projects = Path("~/.claude/projects").expanduser()
    if not projects.is_dir():
        return []
    return [
        LocalClient(
            name="Claude Code",
            adapter="claude-code",
            path=projects,
            size_bytes=_dir_size(projects),
        )
    ]


_FAMILIES = {"cursor": _cursor_clients, "claude-code": _claude_code_clients}


def discover_clients(family: str | None = None) -> list[LocalClient]:
    """Session stores present on this machine, largest first within a family.

    Args:
        family: Restrict to one family (``cursor``, ``claude-code``); all when
            omitted. An unknown name yields an empty list rather than raising,
            so a caller can pass a user-supplied string straight through.
    """
    if family is None:
        families = list(_FAMILIES.values())
    elif family in _FAMILIES:
        families = [_FAMILIES[family]]
    else:
        return []
    found: list[LocalClient] = []
    for fn in families:
        found.extend(sorted(fn(), key=lambda c: c.size_bytes, reverse=True))
    return found


def find_client(family: str, *, path: Path | str | None = None) -> LocalClient | None:
    """One client for ``family``, or ``None`` when nothing is installed.

    An explicit ``path`` short-circuits discovery and is honoured even for a
    store in a non-standard location; it must exist.
    """
    if path is not None:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            return None
        size = (
            _dir_size(resolved)
            if resolved.is_dir()
            else (resolved.stat().st_size if resolved.is_file() else 0)
        )
        adapter = "cursor-vscdb" if family == "cursor" else family
        return LocalClient(name=family, adapter=adapter, path=resolved, size_bytes=size)
    found = discover_clients(family)
    return found[0] if found else None


__all__ = ["LocalClient", "discover_clients", "find_client"]
