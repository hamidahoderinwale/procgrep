"""Tests for `procgrep.ingest.adapters.gumtree`."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from procgrep.canonicalize import canonicalize, get_adapter
from procgrep.ingest.adapters.gumtree import (
    AST_DELETE,
    AST_INSERT,
    AST_MOVE,
    AST_UPDATE,
    NODE_TYPE_SEPARATOR,
    UNKNOWN_NODE_TYPE,
    gumtree_adapter,
    gumtree_atom,
    parse_gumtree_jsondiff,
    run_jsondiff,
)
from procgrep.types import ATOM_OTHER


def test_gumtree_atom_composes_op_and_node_type() -> None:
    assert gumtree_atom("insert-tree", "MethodInvocation") == (
        f"{AST_INSERT}{NODE_TYPE_SEPARATOR}MethodInvocation"
    )
    assert gumtree_atom("delete-node", "Identifier") == (
        f"{AST_DELETE}{NODE_TYPE_SEPARATOR}Identifier"
    )
    assert gumtree_atom("update-node", "StringLiteral") == (
        f"{AST_UPDATE}{NODE_TYPE_SEPARATOR}StringLiteral"
    )
    assert gumtree_atom("move-tree", "Block") == (f"{AST_MOVE}{NODE_TYPE_SEPARATOR}Block")


def test_gumtree_atom_node_variant_and_tree_variant_share_prefix() -> None:
    # Both -node and -tree variants of insert/delete/move collapse to one prefix.
    assert gumtree_atom("insert-node", "X").startswith(AST_INSERT)
    assert gumtree_atom("insert-tree", "X").startswith(AST_INSERT)
    assert gumtree_atom("delete-node", "X").startswith(AST_DELETE)
    assert gumtree_atom("delete-tree", "X").startswith(AST_DELETE)
    assert gumtree_atom("move-node", "X").startswith(AST_MOVE)
    assert gumtree_atom("move-tree", "X").startswith(AST_MOVE)


def test_gumtree_atom_missing_node_type_uses_placeholder() -> None:
    assert gumtree_atom("insert-tree", "") == (
        f"{AST_INSERT}{NODE_TYPE_SEPARATOR}{UNKNOWN_NODE_TYPE}"
    )
    assert gumtree_atom("insert-tree", "   ") == (
        f"{AST_INSERT}{NODE_TYPE_SEPARATOR}{UNKNOWN_NODE_TYPE}"
    )


def test_gumtree_atom_unknown_operation_collapses_to_atom_other() -> None:
    assert gumtree_atom("teleport-tree", "MethodInvocation") == ATOM_OTHER


def test_gumtree_adapter_emits_atom_sequence() -> None:
    record = {
        "trace_id": "t1",
        "agent": "model-a",
        "actions": [
            {"action": "insert-tree", "node_type": "MethodInvocation"},
            {"action": "delete-node", "node_type": "Identifier"},
            {"action": "update-node", "node_type": "Literal"},
            {"action": "move-tree", "node_type": "Block"},
        ],
    }
    atoms = gumtree_adapter(record)
    assert atoms == [
        f"{AST_INSERT}{NODE_TYPE_SEPARATOR}MethodInvocation",
        f"{AST_DELETE}{NODE_TYPE_SEPARATOR}Identifier",
        f"{AST_UPDATE}{NODE_TYPE_SEPARATOR}Literal",
        f"{AST_MOVE}{NODE_TYPE_SEPARATOR}Block",
    ]


def test_gumtree_adapter_unknown_op_yields_atom_other() -> None:
    record = {
        "trace_id": "t1",
        "agent": "model-a",
        "actions": [
            {"action": "teleport-tree", "node_type": "MethodInvocation"},
            {"action": "insert-node", "node_type": "Identifier"},
        ],
    }
    atoms = gumtree_adapter(record)
    assert atoms[0] == ATOM_OTHER
    assert atoms[1].startswith(AST_INSERT)


def test_gumtree_adapter_skips_non_mapping_entries() -> None:
    record = {
        "trace_id": "t1",
        "agent": "model-a",
        "actions": [
            "not a dict",
            {"action": "insert-tree", "node_type": "MethodInvocation"},
            42,
        ],
    }
    atoms = gumtree_adapter(record)
    assert len(atoms) == 1


def test_gumtree_adapter_rejects_non_list_actions() -> None:
    with pytest.raises(TypeError):
        gumtree_adapter({"trace_id": "t1", "agent": "a", "actions": "not a list"})


def test_gumtree_adapter_registered_under_name_gumtree() -> None:
    adapter = get_adapter("gumtree")
    record = {
        "trace_id": "t1",
        "agent": "a",
        "actions": [
            {"action": "insert-tree", "node_type": "If"},
        ],
    }
    assert adapter(record) == [f"{AST_INSERT}{NODE_TYPE_SEPARATOR}If"]


def test_canonicalize_with_gumtree_adapter_round_trip() -> None:
    records = [
        {
            "trace_id": "g-001",
            "agent": "model-a",
            "group": "python",
            "actions": [
                {"action": "insert-tree", "node_type": "Call"},
                {"action": "update-node", "node_type": "Name"},
            ],
        },
        {
            "trace_id": "g-002",
            "agent": "model-b",
            "group": "javascript",
            "actions": [
                {"action": "delete-tree", "node_type": "BinaryExpression"},
            ],
        },
    ]
    traces = canonicalize(records, adapter="gumtree")
    assert len(traces) == 2
    assert traces[0].agent == "model-a"
    assert traces[0].group == "python"
    assert traces[0].atoms == [
        f"{AST_INSERT}{NODE_TYPE_SEPARATOR}Call",
        f"{AST_UPDATE}{NODE_TYPE_SEPARATOR}Name",
    ]
    assert traces[1].atoms == [f"{AST_DELETE}{NODE_TYPE_SEPARATOR}BinaryExpression"]


# --- parse_gumtree_jsondiff -------------------------------------------------


def test_parse_jsondiff_extracts_op_and_leading_type_token() -> None:
    payload = {
        "matches": [],
        "actions": [
            {"action": "insert-tree", "tree": "MethodInvocation [12,45]", "parent": "x", "at": 3},
            {"action": "delete-tree", "tree": "Identifier: foo [10,11]"},
            {"action": "update-node", "tree": "StringLiteral: 'old' [5,9]", "label": "new"},
            {"action": "move-tree", "tree": "Block [20,40]", "parent": "y", "at": 1},
        ],
    }
    parsed = parse_gumtree_jsondiff(payload)
    assert parsed == [
        {"action": "insert-tree", "node_type": "MethodInvocation"},
        {"action": "delete-tree", "node_type": "Identifier"},
        {"action": "update-node", "node_type": "StringLiteral"},
        {"action": "move-tree", "node_type": "Block"},
    ]


def test_parse_jsondiff_handles_missing_tree_field() -> None:
    payload = {"actions": [{"action": "insert-tree"}]}
    parsed = parse_gumtree_jsondiff(payload)
    assert parsed == [{"action": "insert-tree", "node_type": UNKNOWN_NODE_TYPE}]


def test_parse_jsondiff_returns_empty_on_missing_actions_field() -> None:
    assert parse_gumtree_jsondiff({}) == []


def test_parse_jsondiff_pipes_into_adapter_end_to_end() -> None:
    payload = {
        "actions": [
            {"action": "insert-tree", "tree": "MethodInvocation [12,45]"},
            {"action": "delete-node", "tree": "Identifier: x [1,2]"},
        ],
    }
    record = {
        "trace_id": "raw-gumtree-001",
        "agent": "model-a",
        "actions": parse_gumtree_jsondiff(payload),
    }
    atoms = gumtree_adapter(record)
    assert atoms == [
        f"{AST_INSERT}{NODE_TYPE_SEPARATOR}MethodInvocation",
        f"{AST_DELETE}{NODE_TYPE_SEPARATOR}Identifier",
    ]


# --- run_jsondiff -----------------------------------------------------------


def test_run_jsondiff_raises_when_binary_missing(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("x = 2\n")
    with (
        mock.patch("procgrep.ingest.adapters.gumtree.shutil.which", return_value=None),
        pytest.raises(FileNotFoundError, match="gumtree"),
    ):
        run_jsondiff(a, b)


def test_run_jsondiff_parses_subprocess_stdout(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("x = 2\n")

    fake_stdout = '{"matches": [], "actions": [{"action": "update-node", "tree": "Num [0,1]"}]}'
    fake_result = mock.MagicMock()
    fake_result.stdout = fake_stdout

    with (
        mock.patch(
            "procgrep.ingest.adapters.gumtree.shutil.which", return_value="/usr/bin/gumtree"
        ),
        mock.patch(
            "procgrep.ingest.adapters.gumtree.subprocess.run", return_value=fake_result
        ) as run_mock,
    ):
        payload = run_jsondiff(a, b)

    assert payload["actions"][0]["action"] == "update-node"
    run_mock.assert_called_once()
    args = run_mock.call_args[0][0]
    assert args[0] == "gumtree"
    # v4 uses "textdiff -f JSON"; v3 used "jsondiff" — adapter tries v4 first
    assert args[1] == "textdiff"
    assert args[2] == "-f"
    assert args[3] == "JSON"
    assert args[4] == str(a)
    assert args[5] == str(b)
