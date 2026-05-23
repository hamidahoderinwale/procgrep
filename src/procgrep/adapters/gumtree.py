"""Gumtree-based AST-edit-script adapter.

Gumtree produces an AST edit script between two source files for any
language it has a tree generator for. This adapter turns that script
into procgrep atoms. See https://github.com/GumTreeDiff/gumtree.

Atoms are composite ``"<op>:<node_type>"`` strings preserving both
the operation and the node type::

    insert-tree MethodInvocation  ->  "ast_insert:MethodInvocation"
    delete-node Identifier        ->  "ast_delete:Identifier"
    update-node Literal           ->  "ast_update:Literal"
    move-tree Block               ->  "ast_move:Block"

The vocabulary scales with the corpus's distinct node types. This is
high structural information per atom; cross-scaffold comparison
against the small SWE-agent / Agentless alphabet is not the use case.

Expected record shape::

    {
        "trace_id": "...",
        "agent":    "...",
        "actions":  [
            {"action": "insert-tree", "node_type": "MethodInvocation"},
            ...
        ],
    }

:func:`parse_gumtree_jsondiff` converts raw gumtree ``jsondiff`` JSON
into ``actions`` form. :func:`run_jsondiff` shells out to the gumtree
CLI when both source files are on disk and ``gumtree`` is on PATH.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from procgrep.canonicalize import register_adapter
from procgrep.types import ATOM_OTHER, Atom, AtomSequence

AST_INSERT: str = "ast_insert"
AST_DELETE: str = "ast_delete"
AST_UPDATE: str = "ast_update"
AST_MOVE: str = "ast_move"

NODE_TYPE_SEPARATOR: str = ":"
"""Separator between the AST operation prefix and the node-type label."""

_GUMTREE_OP_PREFIX: dict[str, str] = {
    "insert-node": AST_INSERT,
    "insert-tree": AST_INSERT,
    "delete-node": AST_DELETE,
    "delete-tree": AST_DELETE,
    "update-node": AST_UPDATE,
    "move-node": AST_MOVE,
    "move-tree": AST_MOVE,
}
"""Gumtree operation name -> procgrep atom prefix.

``-node`` and ``-tree`` variants collapse to the same prefix; the
distinction is a gumtree implementation detail.
"""

UNKNOWN_NODE_TYPE: str = "?"
"""Placeholder used when a record omits ``node_type``."""

# Gumtree jsondiff represents a node as "<NodeType>: <label> [s,e]" or
# just "<NodeType> [s,e]" -- we take the leading identifier.
_TREE_TYPE_RE = re.compile(r"^\s*([A-Za-z_][\w.$]*)")


def gumtree_atom(operation: str, node_type: str) -> Atom:
    """Compose a gumtree atom from operation + node type.

    Operations outside the gumtree vocabulary collapse to
    :data:`procgrep.types.ATOM_OTHER`. Empty ``node_type`` falls back
    to :data:`UNKNOWN_NODE_TYPE`.
    """
    prefix = _GUMTREE_OP_PREFIX.get(operation)
    if prefix is None:
        return ATOM_OTHER
    nt = node_type.strip() if node_type else ""
    return f"{prefix}{NODE_TYPE_SEPARATOR}{nt or UNKNOWN_NODE_TYPE}"


def gumtree_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Map a gumtree-shaped record into atoms.

    The record's ``actions`` is a list of dicts with ``action``
    (gumtree operation name) and ``node_type``. Unknown operations
    emit :data:`procgrep.types.ATOM_OTHER`.
    """
    steps = record.get("actions", [])
    if not isinstance(steps, list):
        raise TypeError(f"expected list at record['actions'], got {type(steps).__name__}")
    atoms: AtomSequence = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        op = str(step.get("action", ""))
        nt = str(step.get("node_type", "")) if step.get("node_type") is not None else ""
        atoms.append(gumtree_atom(op, nt))
    return atoms


def parse_gumtree_jsondiff(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Convert raw gumtree ``jsondiff`` output to adapter input shape.

    Returns one ``{"action", "node_type"}`` dict per entry in
    ``payload["actions"]``, ready to embed in a procgrep record.
    """
    out: list[dict[str, str]] = []
    raw_actions = payload.get("actions", [])
    if not isinstance(raw_actions, Sequence):
        return out
    for entry in raw_actions:
        if not isinstance(entry, Mapping):
            continue
        op = str(entry.get("action", ""))
        tree = str(entry.get("tree", ""))
        match = _TREE_TYPE_RE.match(tree)
        node_type = match.group(1) if match else UNKNOWN_NODE_TYPE
        out.append({"action": op, "node_type": node_type})
    return out


def run_jsondiff(
    before: Path | str,
    after: Path | str,
    *,
    gumtree_bin: str = "gumtree",
    timeout: float | None = None,
) -> dict[str, Any]:
    """Invoke ``gumtree jsondiff`` on two source files.

    Convenience helper; never invoked from library code. Gumtree is an
    external tool dependency, not a Python import.

    Args:
        gumtree_bin: Override for non-standard installations.
        timeout: Subprocess timeout in seconds.

    Raises:
        FileNotFoundError: ``gumtree_bin`` not on PATH.
        subprocess.CalledProcessError: Gumtree exited non-zero.
        json.JSONDecodeError: Gumtree's stdout was not valid JSON.
    """
    if shutil.which(gumtree_bin) is None:
        raise FileNotFoundError(
            f"gumtree binary {gumtree_bin!r} not found on PATH. "
            f"Install from https://github.com/GumTreeDiff/gumtree, or "
            f"build records manually following the format documented in "
            f"procgrep.adapters_gumtree."
        )
    result = subprocess.run(
        [gumtree_bin, "jsondiff", str(before), str(after)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    payload: dict[str, Any] = json.loads(result.stdout)
    return payload


def _register() -> None:
    """Register under the name ``"gumtree"``."""
    register_adapter("gumtree", gumtree_adapter, overwrite=True)


_register()


__all__ = [
    "AST_DELETE",
    "AST_INSERT",
    "AST_MOVE",
    "AST_UPDATE",
    "NODE_TYPE_SEPARATOR",
    "UNKNOWN_NODE_TYPE",
    "gumtree_adapter",
    "gumtree_atom",
    "parse_gumtree_jsondiff",
    "run_jsondiff",
]
