"""Gumtree-based AST-edit-script adapter.

`procgrep` itself is post-hoc and language-agnostic; for source-level
agent traces we need a language-neutral *atom* layer. Gumtree
(https://github.com/GumTreeDiff/gumtree) produces an AST edit script
between two source files for any language Gumtree has a tree generator
for (Python, JS, TS, Java, C/C++, Go, Ruby, etc.). This module turns
that edit script into procgrep atoms.

Fine-grained node-typed atoms
-----------------------------

A gumtree action carries both an operation (``insert-node``,
``delete-tree``, ``update-node``, ``move-tree``) and the node type it
operates on (``MethodInvocation``, ``Identifier``, ``StringLiteral``,
...). We preserve both by emitting composite atoms of the form
``"<op>:<node_type>"``::

    insert-tree MethodInvocation  ->  "ast_insert:MethodInvocation"
    delete-node Identifier        ->  "ast_delete:Identifier"
    update-node Literal           ->  "ast_update:Literal"
    move-tree Block               ->  "ast_move:Block"

The vocabulary therefore grows with the number of distinct node types
that appear in the corpus. This is a deliberate tradeoff: structural
information per atom is high; cross-scaffold comparison against the
small SWE-agent / Agentless alphabet is *not* meaningful at this layer
and is not the use-case this adapter targets.

The input shape this adapter expects
------------------------------------

Each record is a dict with at least::

    {
        "trace_id": "...",
        "agent":    "...",
        "actions":  [
            {"action": "insert-tree", "node_type": "MethodInvocation"},
            {"action": "delete-node", "node_type": "Identifier"},
            ...
        ],
    }

If you have raw gumtree ``jsondiff`` output, ``parse_gumtree_jsondiff``
converts it into the ``actions`` shape above. If you have a pair of
source files on disk and ``gumtree`` on your PATH, ``run_jsondiff``
shells out to the gumtree CLI and returns the parsed JSON, ready to
feed into the parser.
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
"""Separator joining the AST operation prefix and the node-type label."""

_GUMTREE_OP_PREFIX: dict[str, str] = {
    "insert-node": AST_INSERT,
    "insert-tree": AST_INSERT,
    "delete-node": AST_DELETE,
    "delete-tree": AST_DELETE,
    "update-node": AST_UPDATE,
    "move-node": AST_MOVE,
    "move-tree": AST_MOVE,
}
"""Gumtree edit-script operation name -> procgrep atom prefix.

Both the ``-node`` and ``-tree`` variants collapse to the same prefix:
the operation matters; the subtree-vs-single-node distinction is a
gumtree implementation detail.
"""

UNKNOWN_NODE_TYPE: str = "?"
"""Node-type placeholder used when the input record omits ``node_type``."""

_TREE_TYPE_RE = re.compile(r"^\s*([A-Za-z_][\w.$]*)")
"""Extract the leading type token from a gumtree ``tree`` string.

Gumtree's ``jsondiff`` represents a node as ``"<NodeType>: <label> [s,e]"``
or simply ``"<NodeType> [s,e]"`` depending on whether the node carries a
label. We treat the leading identifier as the node-type label.
"""


def gumtree_atom(operation: str, node_type: str) -> Atom:
    """Compose a fine-grained gumtree atom from operation + node type.

    Args:
        operation: One of the keys of ``_GUMTREE_OP_PREFIX`` (e.g.
            ``"insert-tree"``).
        node_type: Gumtree AST node-type label (e.g.
            ``"MethodInvocation"``). Falls back to
            :data:`UNKNOWN_NODE_TYPE` if empty.

    Returns:
        A composite atom string. Operations outside the gumtree
        vocabulary collapse to :data:`procgrep.types.ATOM_OTHER`.
    """
    prefix = _GUMTREE_OP_PREFIX.get(operation)
    if prefix is None:
        return ATOM_OTHER
    nt = node_type.strip() if node_type else ""
    return f"{prefix}{NODE_TYPE_SEPARATOR}{nt or UNKNOWN_NODE_TYPE}"


def gumtree_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Map a gumtree-shaped trace record into a sequence of atoms.

    The record must carry an ``actions`` field listing dicts, each with
    an ``action`` (gumtree operation name) and ``node_type`` (AST
    node-type label). Any dict whose ``action`` is not in the gumtree
    vocabulary emits :data:`procgrep.types.ATOM_OTHER`.

    Args:
        record: One gumtree-flavored trace record.

    Returns:
        The ordered list of atoms produced by the record's actions.
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
    """Convert raw gumtree ``jsondiff`` output into the adapter input shape.

    Gumtree's ``jsondiff`` JSON has an ``actions`` array whose entries
    look like::

        {"action": "insert-tree", "tree": "MethodInvocation [12,45]",
         "parent": "...", "at": 3}

    This helper extracts the operation and the leading node-type token
    from each entry and returns a list of ``{"action", "node_type"}``
    dicts ready to be embedded in a procgrep record under ``actions``.

    Args:
        payload: Parsed gumtree ``jsondiff`` JSON (top-level dict).

    Returns:
        List of ``{"action": <op>, "node_type": <type>}`` dicts.
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
    """Invoke the gumtree CLI's ``jsondiff`` subcommand on two source files.

    Procgrep declares gumtree as an external tool dependency, not a
    Python import. This helper is provided for convenience when both
    files exist on the local filesystem and the ``gumtree`` binary is
    on ``PATH``; it is optional and never invoked from procgrep's own
    library code.

    Args:
        before: Path to the *before* source file.
        after: Path to the *after* source file.
        gumtree_bin: Name of the gumtree binary to invoke. Override
            for non-standard installations.
        timeout: Optional subprocess timeout in seconds.

    Returns:
        Parsed gumtree ``jsondiff`` JSON (the dict containing
        ``matches`` and ``actions``).

    Raises:
        FileNotFoundError: If ``gumtree_bin`` is not on PATH.
        subprocess.CalledProcessError: If gumtree exits non-zero.
        json.JSONDecodeError: If gumtree's stdout is not valid JSON.
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
    """Register the gumtree adapter under the name ``"gumtree"``."""
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
