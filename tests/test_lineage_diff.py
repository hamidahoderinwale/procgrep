"""Tests for the lineage_diff primitive."""

from __future__ import annotations

import math

import pytest

from procgrep.lineage_diff import (
    DEFAULT_AXES,
    AxisResult,
    LineageDiff,
    lineage_diff,
)
from procgrep.types import (
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    Trace,
)


def _trace(
    trace_id: str,
    atoms: list[str],
    *,
    agent: str = "agent",
    group: str | None = None,
    resolved: bool | None = None,
) -> Trace:
    """Helper to build a Trace with optional outcome metadata."""
    metadata: dict[str, object] = {}
    if resolved is not None:
        metadata["resolved"] = resolved
    return Trace(
        trace_id=trace_id,
        agent=agent,
        atoms=atoms,
        group=group,
        metadata=metadata,
    )


# Top-level lineage_diff -----------------------------------------------------


def test_returns_lineage_diff_with_default_axes() -> None:
    parent = [_trace("p1", [ATOM_EDIT, ATOM_RUN_TEST])]
    child = [_trace("c1", [ATOM_EDIT, ATOM_RUN_TEST])]
    diff = lineage_diff(parent, child)
    assert isinstance(diff, LineageDiff)
    assert len(diff.axes) == len(DEFAULT_AXES)
    assert tuple(a.axis for a in diff.axes) == DEFAULT_AXES


def test_preserves_axis_order_from_input() -> None:
    parent = [_trace("p1", [ATOM_EDIT])]
    child = [_trace("c1", [ATOM_EDIT])]
    diff = lineage_diff(parent, child, along=["entropy", "vocabulary"])
    assert [a.axis for a in diff.axes] == ["entropy", "vocabulary"]


def test_carries_labels_and_trajectory_counts() -> None:
    parent = [_trace(f"p{i}", [ATOM_EDIT]) for i in range(3)]
    child = [_trace(f"c{i}", [ATOM_EDIT]) for i in range(5)]
    diff = lineage_diff(parent, child, parent_label="Qwen-32B", child_label="SWE-LM-32B")
    assert diff.parent_label == "Qwen-32B"
    assert diff.child_label == "SWE-LM-32B"
    assert diff.n_parent == 3
    assert diff.n_child == 5


def test_unknown_axis_raises() -> None:
    parent = [_trace("p1", [ATOM_EDIT])]
    child = [_trace("c1", [ATOM_EDIT])]
    with pytest.raises(ValueError, match="unknown axis"):
        lineage_diff(parent, child, along=["nonexistent_axis"])


def test_outcome_quadrant_requires_outcome_field() -> None:
    parent = [_trace("p1", [ATOM_EDIT], resolved=True)]
    child = [_trace("c1", [ATOM_EDIT], resolved=False)]
    with pytest.raises(ValueError, match="outcome_field"):
        lineage_diff(parent, child, along=["outcome_quadrant"])


# Vocabulary axis ------------------------------------------------------------


def test_vocabulary_identical_sets_jaccard_1() -> None:
    parent = [_trace("p", [ATOM_EDIT, ATOM_RUN_TEST])]
    child = [_trace("c", [ATOM_RUN_TEST, ATOM_EDIT, ATOM_EDIT])]
    diff = lineage_diff(parent, child, along=["vocabulary"])
    axis = diff.axes[0]
    assert axis.axis == "vocabulary"
    assert axis.summary_value == pytest.approx(1.0)
    assert axis.detail["parent_only"] == []
    assert axis.detail["child_only"] == []


def test_vocabulary_disjoint_sets_jaccard_0() -> None:
    parent = [_trace("p", [ATOM_EDIT, ATOM_RUN_TEST])]
    child = [_trace("c", [ATOM_SEARCH_REPO, ATOM_READ_FILE])]
    diff = lineage_diff(parent, child, along=["vocabulary"])
    axis = diff.axes[0]
    assert axis.summary_value == pytest.approx(0.0)
    assert set(axis.detail["parent_only"]) == {ATOM_EDIT, ATOM_RUN_TEST}
    assert set(axis.detail["child_only"]) == {ATOM_SEARCH_REPO, ATOM_READ_FILE}


def test_vocabulary_partial_overlap() -> None:
    parent = [_trace("p", [ATOM_EDIT, ATOM_RUN_TEST, ATOM_SEARCH_REPO])]
    child = [_trace("c", [ATOM_EDIT, ATOM_SUBMIT])]
    diff = lineage_diff(parent, child, along=["vocabulary"])
    axis = diff.axes[0]
    # Shared: {EDIT}. Union: {EDIT, RUN_TEST, SEARCH_REPO, SUBMIT}. Jaccard = 1/4.
    assert axis.summary_value == pytest.approx(0.25)
    assert axis.detail["shared"] == [ATOM_EDIT]
    assert ATOM_SUBMIT in axis.detail["child_only"]
    assert ATOM_RUN_TEST in axis.detail["parent_only"]


def test_vocabulary_both_empty_jaccard_1() -> None:
    diff = lineage_diff([], [], along=["vocabulary"])
    axis = diff.axes[0]
    assert axis.summary_value == pytest.approx(1.0)


def test_vocabulary_reports_sizes() -> None:
    parent = [_trace("p", [ATOM_EDIT, ATOM_RUN_TEST, ATOM_SEARCH_REPO])]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(parent, child, along=["vocabulary"])
    axis = diff.axes[0]
    assert axis.detail["parent_vocab_size"] == 3
    assert axis.detail["child_vocab_size"] == 1


# Entropy axis ---------------------------------------------------------------


def test_entropy_identical_concentrated_traces_shift_zero() -> None:
    parent = [_trace("p", [ATOM_EDIT, ATOM_EDIT, ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT, ATOM_EDIT, ATOM_EDIT])]
    diff = lineage_diff(parent, child, along=["entropy"])
    axis = diff.axes[0]
    assert axis.summary_value == pytest.approx(0.0)
    assert axis.detail["parent_mean"] == pytest.approx(0.0)
    assert axis.detail["child_mean"] == pytest.approx(0.0)


def test_entropy_mode_collapse_positive_shift() -> None:
    # Parent: diverse 4-atom mix. Child: all-EDIT (mode collapse).
    parent = [_trace("p", [ATOM_EDIT, ATOM_RUN_TEST, ATOM_SEARCH_REPO, ATOM_READ_FILE])]
    child = [_trace("c", [ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT])]
    diff = lineage_diff(parent, child, along=["entropy"])
    axis = diff.axes[0]
    assert axis.summary_value > 0
    assert axis.detail["parent_mean"] == pytest.approx(math.log(4))
    assert axis.detail["child_mean"] == pytest.approx(0.0)


def test_entropy_child_more_diverse_negative_shift() -> None:
    parent = [_trace("p", [ATOM_EDIT, ATOM_EDIT, ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT, ATOM_RUN_TEST, ATOM_SEARCH_REPO])]
    diff = lineage_diff(parent, child, along=["entropy"])
    axis = diff.axes[0]
    assert axis.summary_value < 0


def test_entropy_handles_empty_traces() -> None:
    parent = [_trace("p", [])]
    child = [_trace("c", [])]
    diff = lineage_diff(parent, child, along=["entropy"])
    axis = diff.axes[0]
    assert axis.summary_value == pytest.approx(0.0)
    assert axis.detail["parent_mean"] == pytest.approx(0.0)


def test_entropy_handles_empty_corpora() -> None:
    diff = lineage_diff([], [], along=["entropy"])
    axis = diff.axes[0]
    assert axis.summary_value == pytest.approx(0.0)


def test_entropy_reports_median() -> None:
    parent = [
        _trace("p1", [ATOM_EDIT, ATOM_EDIT]),
        _trace("p2", [ATOM_EDIT, ATOM_RUN_TEST]),
        _trace("p3", [ATOM_EDIT, ATOM_RUN_TEST, ATOM_SEARCH_REPO]),
    ]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(parent, child, along=["entropy"])
    axis = diff.axes[0]
    assert "parent_median" in axis.detail
    assert "child_median" in axis.detail


# Outcome quadrant axis ------------------------------------------------------


def test_outcome_quadrant_basic() -> None:
    parent = [
        _trace("pp", [ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT], resolved=True),
        _trace("pf", [ATOM_EDIT, ATOM_EDIT], resolved=False),
    ]
    child = [
        _trace("cp", [ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT], resolved=True),
        _trace("cf", [ATOM_EDIT, ATOM_EDIT], resolved=False),
    ]
    diff = lineage_diff(parent, child, along=["outcome_quadrant"], outcome_field="resolved")
    axis = diff.axes[0]
    assert axis.axis == "outcome_quadrant"
    assert axis.detail["parent_pass_n"] == 1
    assert axis.detail["parent_fail_n"] == 1
    assert axis.detail["child_pass_n"] == 1
    assert axis.detail["child_fail_n"] == 1
    # Pass-side vocabularies are identical (EDIT, RUN_TEST, SUBMIT).
    assert axis.detail["pass_vocab_jaccard"] == pytest.approx(1.0)
    # Fail-side vocabularies are identical (just EDIT).
    assert axis.detail["fail_vocab_jaccard"] == pytest.approx(1.0)
    # Headline = pass-side jaccard.
    assert axis.summary_value == pytest.approx(1.0)


def test_outcome_quadrant_detects_pass_side_drift() -> None:
    parent = [_trace("pp", [ATOM_EDIT, ATOM_RUN_TEST], resolved=True)]
    child = [_trace("cp", [ATOM_SEARCH_REPO, ATOM_SUBMIT], resolved=True)]
    diff = lineage_diff(parent, child, along=["outcome_quadrant"], outcome_field="resolved")
    axis = diff.axes[0]
    assert axis.detail["pass_vocab_jaccard"] == pytest.approx(0.0)


def test_outcome_quadrant_with_missing_field_treats_as_fail() -> None:
    """Traces without the outcome_field land in the fail stratum (falsy default)."""
    parent = [_trace("p", [ATOM_EDIT])]  # no resolved field
    child = [_trace("c", [ATOM_EDIT], resolved=True)]
    diff = lineage_diff(parent, child, along=["outcome_quadrant"], outcome_field="resolved")
    axis = diff.axes[0]
    assert axis.detail["parent_fail_n"] == 1
    assert axis.detail["parent_pass_n"] == 0
    assert axis.detail["child_pass_n"] == 1


# LineageDiff methods --------------------------------------------------------


def test_summary_is_multiline_string() -> None:
    parent = [_trace("p", [ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(parent, child, parent_label="A", child_label="B")
    s = diff.summary()
    assert "A -> B" in s
    assert "n_parent=1" in s
    assert "vocabulary" in s
    assert "entropy" in s


def test_to_markdown_returns_markdown() -> None:
    parent = [_trace("p", [ATOM_EDIT, ATOM_RUN_TEST])]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(parent, child, parent_label="parent", child_label="child")
    md = diff.to_markdown()
    assert md.startswith("# Lineage diff")
    assert "## Axes" in md
    assert "### vocabulary" in md
    assert "### entropy" in md


def test_to_markdown_truncates_long_lists() -> None:
    """Lists with more than 10 entries get truncated with a +N more suffix."""
    many_atoms = [f"atom_{i}" for i in range(15)]
    parent = [_trace("p", many_atoms)]
    child = [_trace("c", [])]
    diff = lineage_diff(parent, child, along=["vocabulary"])
    md = diff.to_markdown()
    assert "more)" in md


def test_to_records_is_json_serializable() -> None:
    import json

    parent = [_trace("p", [ATOM_EDIT, ATOM_RUN_TEST])]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(parent, child, parent_label="A", child_label="B")
    records = diff.to_records()
    # Round-trip through JSON to assert serializability.
    payload = json.loads(json.dumps(records))
    assert payload["parent_label"] == "A"
    assert payload["child_label"] == "B"
    assert payload["n_parent"] == 1
    assert len(payload["axes"]) == len(DEFAULT_AXES)


# AxisResult -----------------------------------------------------------------


def test_axis_result_is_frozen_dataclass() -> None:
    import dataclasses

    axis = AxisResult(axis="x", summary_value=0.5, detail={"k": "v"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        axis.summary_value = 0.6  # type: ignore[misc]


def test_lineage_diff_is_frozen_dataclass() -> None:
    import dataclasses

    parent = [_trace("p", [ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(parent, child)
    with pytest.raises(dataclasses.FrozenInstanceError):
        diff.parent_label = "other"  # type: ignore[misc]
