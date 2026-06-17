"""A persistent library of named procedures.

A procedure -- derived from winners or hand-authored -- is a `ProcedureSpec`.
This stores a collection of them as YAML files in a directory, in the same
format `ProcedureSpec.from_yaml` reads, so library entries plug straight into
`enforce` / `verify` / `score` / `optimize` with no new object model. It is
deliberately thin: a typed view over a directory of specs. The directory is
git-friendly and diffable, so your recurring procedures become version-
controlled artifacts -- procedural memory that persists across sessions.

    lib = ProcedureLibrary("procedures/")
    lib.save("test_after_edit", ProcedureSpec.from_winners(traces, vocab))
    spec = lib.load("test_after_edit")          # a ProcedureSpec again
    cfg = enforce(spec, mode="prompt", scaffold="swe-agent")
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from procgrep.reward import ProcedureSpec

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(name: str) -> str:
    """A filesystem-safe stem for a procedure name."""
    return _UNSAFE.sub("_", name).strip("_") or "procedure"


class ProcedureLibrary:
    """A directory of `ProcedureSpec` YAML files, addressed by name."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        return self.root / f"{_slug(name)}.yaml"

    def save(self, name: str, spec: ProcedureSpec) -> Path:
        """Save ``spec`` under ``name``, stamping the spec's own name to match."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(name)
        (spec if spec.name == name else replace(spec, name=name)).to_yaml(path)
        return path

    def load(self, name: str) -> ProcedureSpec:
        path = self._path(name)
        if not path.exists():
            raise KeyError(f"no procedure named {name!r} in {self.root}")
        return ProcedureSpec.from_yaml(path)

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.yaml"))

    def __contains__(self, name: str) -> bool:
        return self._path(name).exists()

    def __len__(self) -> int:
        return len(self.names())

    def __iter__(self) -> Iterator[tuple[str, ProcedureSpec]]:
        for stem in self.names():
            yield stem, ProcedureSpec.from_yaml(self.root / f"{stem}.yaml")


__all__ = ["ProcedureLibrary"]
