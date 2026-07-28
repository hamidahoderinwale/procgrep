"""Tests for the lineage_diff primitive."""

from __future__ import annotations

import math
import warnings

import pytest

from procgrep.lineage_diff import (
    DEFAULT_AXES,
    AxisResult,
    LineageDiff,
    NoiseFloor,
    lineage_diff,
    noise_floor,
)
from procgrep.types import (
    ATOM_EDIT,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
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


# Top-level lineage_diff.


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


# Vocabulary axis.


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


# Entropy axis.


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


# Outcome quadrant axis.


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


# LineageDiff methods.


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


# AxisResult.


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


# Multi-resolution alphabets (Option C)


def test_default_alphabet_is_canonical() -> None:
    """Each axis defaults to alphabet='canonical' for backward compat."""
    parent = [_trace("p", [ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(parent, child)
    for axis in diff.axes:
        assert axis.alphabet == "canonical"


def test_single_alphabet_string_tags_axes() -> None:
    """A single alphabet string tags every axis with that label."""
    parent = [_trace("p", [ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(parent, child, alphabet="native")
    for axis in diff.axes:
        assert axis.alphabet == "native"


def test_multi_alphabet_runs_each_axis_per_alphabet() -> None:
    """A sequence of alphabets produces axes-x-alphabet results."""
    parent = [_trace("p", [ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(
        parent,
        child,
        along=["vocabulary", "entropy"],
        alphabet=["canonical", "native"],
    )
    # 2 axes x 2 alphabets = 4 AxisResults
    assert len(diff.axes) == 4
    # Order: canonical/vocab, canonical/entropy, native/vocab, native/entropy
    assert diff.axes[0].alphabet == "canonical"
    assert diff.axes[0].axis == "vocabulary"
    assert diff.axes[1].alphabet == "canonical"
    assert diff.axes[1].axis == "entropy"
    assert diff.axes[2].alphabet == "native"
    assert diff.axes[2].axis == "vocabulary"
    assert diff.axes[3].alphabet == "native"
    assert diff.axes[3].axis == "entropy"


def test_canonical_projection_applied_under_canonical_alphabet() -> None:
    """canonical_projection alters atoms only when alphabet='canonical'."""
    # Parent and child emit different native atoms that project to the
    # same canonical atom. Under canonical-with-projection, the
    # vocabulary axis should report Jaccard 1.0 (both collapse to one
    # canonical atom). Under native, vocabularies differ.
    parent = [_trace("p", ["str_replace"])]
    child = [_trace("c", ["str_replace_editor"])]

    def to_canonical(atom: str) -> str:
        if atom in {"str_replace", "str_replace_editor"}:
            return ATOM_EDIT
        return atom

    diff = lineage_diff(
        parent,
        child,
        along=["vocabulary"],
        alphabet=["canonical", "native"],
        canonical_projection=to_canonical,
    )
    canonical_voc = next(a for a in diff.axes if a.alphabet == "canonical")
    native_voc = next(a for a in diff.axes if a.alphabet == "native")
    assert canonical_voc.summary_value == pytest.approx(1.0)
    assert native_voc.summary_value == pytest.approx(0.0)


def test_canonical_projection_does_not_affect_native_alphabet() -> None:
    """canonical_projection is a no-op when alphabet='native' even if provided."""
    parent = [_trace("p", ["str_replace"])]
    child = [_trace("c", ["str_replace_editor"])]

    def collapse_all_to_edit(_atom: str) -> str:
        return ATOM_EDIT

    # Run native only with a projection that would collapse everything;
    # native results must reflect the original atoms, not the projection.
    diff = lineage_diff(
        parent,
        child,
        along=["vocabulary"],
        alphabet="native",
        canonical_projection=collapse_all_to_edit,
    )
    axis = diff.axes[0]
    assert axis.alphabet == "native"
    # Different native atoms => Jaccard 0
    assert axis.summary_value == pytest.approx(0.0)


def test_canonical_alphabet_without_projection_is_no_op() -> None:
    """alphabet='canonical' with projection=None passes atoms through unchanged."""
    parent = [_trace("p", [ATOM_EDIT, ATOM_RUN_TEST])]
    child = [_trace("c", [ATOM_EDIT, ATOM_RUN_TEST])]
    diff = lineage_diff(parent, child, alphabet="canonical")
    for axis in diff.axes:
        assert axis.alphabet == "canonical"
    # Same atoms on both sides, perfect preservation
    voc = next(a for a in diff.axes if a.axis == "vocabulary")
    assert voc.summary_value == pytest.approx(1.0)


def test_axis_result_alphabet_field_defaults_to_canonical() -> None:
    """AxisResult.alphabet has a default so existing constructors don't break."""
    axis = AxisResult(axis="vocabulary", summary_value=0.5, detail={})
    assert axis.alphabet == "canonical"


def test_axis_result_alphabet_is_settable() -> None:
    """AxisResult.alphabet accepts arbitrary string labels."""
    axis = AxisResult(
        axis="vocabulary",
        summary_value=0.5,
        detail={},
        alphabet="my-custom-vocab",
    )
    assert axis.alphabet == "my-custom-vocab"


def test_to_records_includes_alphabet_field() -> None:
    """LineageDiff.to_records preserves the alphabet on each axis."""
    parent = [_trace("p", [ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT])]
    diff = lineage_diff(
        parent,
        child,
        along=["vocabulary"],
        alphabet=["canonical", "native"],
    )
    records = diff.to_records()
    axes = records["axes"]
    assert isinstance(axes, list)
    assert axes[0]["alphabet"] == "canonical"
    assert axes[1]["alphabet"] == "native"


def test_outcome_quadrant_works_under_multiple_alphabets() -> None:
    """The outcome_quadrant axis is also multi-alphabet-aware."""
    parent = [
        _trace("pp", [ATOM_EDIT, ATOM_RUN_TEST], resolved=True),
        _trace("pf", [ATOM_EDIT], resolved=False),
    ]
    child = [
        _trace("cp", [ATOM_EDIT, ATOM_RUN_TEST], resolved=True),
        _trace("cf", [ATOM_EDIT], resolved=False),
    ]
    diff = lineage_diff(
        parent,
        child,
        along=["outcome_quadrant"],
        alphabet=["canonical", "native"],
        outcome_field="resolved",
    )
    assert len(diff.axes) == 2
    assert {a.alphabet for a in diff.axes} == {"canonical", "native"}


# Conditional axis.


def test_conditional_identical_corpora_zero_divergence() -> None:
    """Identical parent and child traces produce conditional JSD 0."""
    atoms = [ATOM_READ_FILE, ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT]
    parent = [_trace(f"p{i}", atoms) for i in range(3)]
    child = [_trace(f"c{i}", list(atoms)) for i in range(3)]
    diff = lineage_diff(parent, child, along=["conditional"])
    axis = diff.axes[0]
    assert axis.axis == "conditional"
    assert axis.summary_value == pytest.approx(0.0)
    assert axis.detail["shared_prefixes"] >= 1


def test_conditional_structural_drift_high_divergence() -> None:
    """Parent always edit->run_test, child always edit->edit => high JSD on ("edit",)."""
    parent = [
        _trace(f"p{i}", [ATOM_READ_FILE, ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT]) for i in range(5)
    ]
    child = [_trace(f"c{i}", [ATOM_READ_FILE, ATOM_EDIT, ATOM_EDIT, ATOM_SUBMIT]) for i in range(5)]
    diff = lineage_diff(parent, child, along=["conditional"])
    axis = diff.axes[0]
    # Both ("read_file",) and ("edit",) are shared; ("edit",) diverges fully
    # because parent always goes to RUN_TEST after edit while child always
    # goes to EDIT. The other prefixes contribute 0 or low JSD.
    assert axis.summary_value > 0.0
    edit_prefix_entry = next(
        e for e in axis.detail["top_divergent_prefixes"] if e["prefix"] == [ATOM_EDIT]
    )
    assert edit_prefix_entry["jsd"] == pytest.approx(1.0, abs=0.01)


def test_conditional_no_shared_prefixes_returns_zero_with_diagnostic() -> None:
    """Corpora with disjoint prefixes get summary 0 and explanatory detail."""
    parent = [_trace("p", [ATOM_READ_FILE, ATOM_EDIT])]
    child = [_trace("c", [ATOM_SEARCH_REPO, ATOM_RUN_TEST])]
    diff = lineage_diff(parent, child, along=["conditional"])
    axis = diff.axes[0]
    assert axis.summary_value == pytest.approx(0.0)
    assert axis.detail["shared_prefixes"] == 0
    assert axis.detail["parent_only_prefixes"] == 1
    assert axis.detail["child_only_prefixes"] == 1


def test_conditional_handles_short_traces_gracefully() -> None:
    """Traces shorter than context_k + 1 contribute no prefixes."""
    parent = [_trace("p1", [ATOM_EDIT]), _trace("p2", [ATOM_EDIT, ATOM_RUN_TEST])]
    child = [_trace("c1", [ATOM_EDIT]), _trace("c2", [ATOM_EDIT, ATOM_RUN_TEST])]
    diff = lineage_diff(parent, child, along=["conditional"], conditional_context_k=1)
    axis = diff.axes[0]
    # Only p2 / c2 contribute, with prefix ("edit",) -> "run_test" (JSD 0).
    assert axis.summary_value == pytest.approx(0.0)
    assert axis.detail["shared_prefixes"] == 1


def test_conditional_context_k_propagates() -> None:
    """conditional_context_k changes the prefix length used in the dispatch."""
    parent = [
        _trace(
            f"p{i}",
            [ATOM_READ_FILE, ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT],
        )
        for i in range(3)
    ]
    child = [
        _trace(
            f"c{i}",
            [ATOM_READ_FILE, ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT],
        )
        for i in range(3)
    ]
    diff_k1 = lineage_diff(parent, child, along=["conditional"], conditional_context_k=1)
    diff_k2 = lineage_diff(parent, child, along=["conditional"], conditional_context_k=2)
    assert diff_k1.axes[0].detail["context_k"] == 1
    assert diff_k2.axes[0].detail["context_k"] == 2
    # k=2 should generally have fewer shared prefixes than k=1 (sparser).
    assert diff_k2.axes[0].detail["shared_prefixes"] <= diff_k1.axes[0].detail["shared_prefixes"]


def test_conditional_invalid_context_k_raises() -> None:
    """context_k < 1 is rejected."""
    parent = [_trace("p", [ATOM_EDIT, ATOM_RUN_TEST])]
    child = [_trace("c", [ATOM_EDIT, ATOM_RUN_TEST])]
    with pytest.raises(ValueError, match="context_k must be >= 1"):
        lineage_diff(parent, child, along=["conditional"], conditional_context_k=0)


def test_conditional_works_under_multiple_alphabets() -> None:
    """The conditional axis composes with multi-resolution alphabets."""
    parent = [_trace(f"p{i}", [ATOM_EDIT, ATOM_RUN_TEST]) for i in range(3)]
    child = [_trace(f"c{i}", [ATOM_EDIT, ATOM_RUN_TEST]) for i in range(3)]
    diff = lineage_diff(
        parent,
        child,
        along=["conditional"],
        alphabet=["canonical", "native"],
    )
    assert len(diff.axes) == 2
    assert {a.alphabet for a in diff.axes} == {"canonical", "native"}
    assert all(a.axis == "conditional" for a in diff.axes)


def test_conditional_in_default_dispatch_error_message() -> None:
    """The unknown-axis error message now lists 'conditional' as available."""
    parent = [_trace("p", [ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT])]
    with pytest.raises(ValueError, match="conditional"):
        lineage_diff(parent, child, along=["definitely_not_a_real_axis"])


def _tagged(trace_id: str, atoms: list[str], scaffold: str) -> Trace:
    """A Trace tagged with a provenance/partition key for the new features."""
    return Trace(
        trace_id=trace_id, agent="agent", atoms=atoms, group=None, metadata={"scaffold": scaffold}
    )


# exclude_atoms.


def test_exclude_atoms_removes_them_from_vocabulary() -> None:
    """Dropping adapter-sensitive atoms can recover an otherwise shared vocabulary."""
    parent = [_trace("p", [ATOM_EDIT, ATOM_THINK, ATOM_EDIT])]
    child = [_trace("c", [ATOM_EDIT, ATOM_OTHER, ATOM_EDIT])]
    before = lineage_diff(parent, child, along=["vocabulary"]).axes[0].summary_value
    after = (
        lineage_diff(parent, child, along=["vocabulary"], exclude_atoms=[ATOM_THINK, ATOM_OTHER])
        .axes[0]
        .summary_value
    )
    assert before == pytest.approx(1 / 3)  # {edit} shared of {edit, think, other}
    assert after == pytest.approx(1.0)  # only edit remains on both sides


def test_exclude_atoms_none_is_noop() -> None:
    """The default leaves every atom in place."""
    parent = [_trace("p", [ATOM_EDIT, ATOM_THINK])]
    child = [_trace("c", [ATOM_EDIT, ATOM_THINK])]
    base = lineage_diff(parent, child, along=["vocabulary"]).axes[0].detail["shared"]
    assert ATOM_THINK in base


def test_exclude_atoms_applies_after_projection() -> None:
    """Exclusion is on the projected alphabet, not the raw atoms."""
    parent = [_trace("p", ["X", ATOM_EDIT])]
    child = [_trace("c", ["X", ATOM_EDIT])]
    # X projects to think, then think is excluded, leaving only edit.
    diff = lineage_diff(
        parent,
        child,
        along=["vocabulary"],
        canonical_projection=lambda a: ATOM_THINK if a == "X" else a,
        exclude_atoms=[ATOM_THINK],
    )
    assert diff.axes[0].detail["shared"] == [ATOM_EDIT]


def test_exclude_atoms_does_not_mutate_input() -> None:
    """Filtering builds new traces; caller atoms are untouched."""
    parent_atoms = [ATOM_EDIT, ATOM_THINK]
    parent = [_trace("p", parent_atoms)]
    child = [_trace("c", [ATOM_EDIT, ATOM_THINK])]
    lineage_diff(parent, child, along=["vocabulary"], exclude_atoms=[ATOM_THINK])
    assert parent_atoms == [ATOM_EDIT, ATOM_THINK]


# provenance_field guardrail.


def test_provenance_mismatch_warns() -> None:
    """Different scaffolds on the two sides trigger a warning."""
    parent = [_tagged("p", [ATOM_EDIT], "openhands")]
    child = [_tagged("c", [ATOM_EDIT], "sweagent")]
    with pytest.warns(UserWarning, match="provenance 'scaffold'"):
        lineage_diff(parent, child, along=["vocabulary"], provenance_field="scaffold")


def test_provenance_match_does_not_warn() -> None:
    """Matching scaffolds raise no warning."""
    parent = [_tagged("p", [ATOM_EDIT], "openhands")]
    child = [_tagged("c", [ATOM_EDIT], "openhands")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lineage_diff(parent, child, along=["vocabulary"], provenance_field="scaffold")


def test_provenance_field_none_never_warns() -> None:
    """Without the opt-in, mismatched scaffolds are not checked."""
    parent = [_tagged("p", [ATOM_EDIT], "openhands")]
    child = [_tagged("c", [ATOM_EDIT], "sweagent")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lineage_diff(parent, child, along=["vocabulary"])


def test_provenance_untagged_traces_ignored() -> None:
    """Traces missing the key contribute no provenance value, so no warning."""
    parent = [_trace("p", [ATOM_EDIT])]  # no scaffold metadata
    child = [_trace("c", [ATOM_EDIT])]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lineage_diff(parent, child, along=["vocabulary"], provenance_field="scaffold")


# noise_floor.


def test_noise_floor_identical_partitions_floor_is_clean() -> None:
    """Same atoms across scaffolds: vocabulary floor 1.0, conditional floor 0.0."""
    traces = [
        _tagged("a1", [ATOM_EDIT, ATOM_RUN_TEST], "openhands"),
        _tagged("b1", [ATOM_EDIT, ATOM_RUN_TEST], "sweagent"),
    ]
    nf = noise_floor(traces, by="scaffold", along=["vocabulary", "conditional"])
    assert isinstance(nf, NoiseFloor)
    assert nf.per_axis["vocabulary"]["max"] == pytest.approx(1.0)
    assert nf.per_axis["conditional"]["max"] == pytest.approx(0.0)


def test_noise_floor_counts_pairs_and_groups() -> None:
    """Three partitions give C(3, 2) = 3 pairs."""
    traces = [
        _tagged("a", [ATOM_EDIT], "x"),
        _tagged("b", [ATOM_EDIT], "y"),
        _tagged("c", [ATOM_EDIT], "z"),
    ]
    nf = noise_floor(traces, by="scaffold", along=["vocabulary"])
    assert nf.groups == ("x", "y", "z")
    assert nf.n_pairs == 3
    assert len(nf.pairs) == 3


def test_noise_floor_uses_magnitude_for_signed_axes() -> None:
    """Entropy shift sign must not cancel; the floor is its magnitude."""
    # 'lo' is concentrated (entropy 0), 'hi' is diverse (entropy > 0); the
    # pair's entropy summary is negative, but the floor reports the magnitude.
    traces = [
        _tagged("lo1", [ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT], "lo"),
        _tagged("hi1", [ATOM_EDIT, ATOM_READ_FILE, ATOM_RUN_TEST, ATOM_SEARCH_REPO], "hi"),
    ]
    nf = noise_floor(traces, by="scaffold", along=["entropy"])
    assert nf.per_axis["entropy"]["max"] > 0.5


def test_noise_floor_requires_two_partitions() -> None:
    """A single partition cannot define a floor."""
    traces = [_tagged("a", [ATOM_EDIT], "only")]
    with pytest.raises(ValueError, match=">= 2 partitions"):
        noise_floor(traces, by="scaffold")


def test_noise_floor_forwards_kwargs_to_lineage_diff() -> None:
    """exclude_atoms passed through changes the measured floor."""
    traces = [
        _tagged("a", [ATOM_EDIT, ATOM_THINK], "x"),
        _tagged("b", [ATOM_EDIT, ATOM_OTHER], "y"),
    ]
    floored = noise_floor(traces, by="scaffold", along=["vocabulary"]).per_axis["vocabulary"]["max"]
    cleaned = noise_floor(
        traces, by="scaffold", along=["vocabulary"], exclude_atoms=[ATOM_THINK, ATOM_OTHER]
    ).per_axis["vocabulary"]["max"]
    assert cleaned > floored  # dropping the adapter-only atoms recovers shared vocab


def test_noise_floor_summary_is_string() -> None:
    """summary() renders one line per axis."""
    traces = [_tagged("a", [ATOM_EDIT], "x"), _tagged("b", [ATOM_EDIT], "y")]
    text = noise_floor(traces, by="scaffold", along=["vocabulary"]).summary()
    assert "NoiseFloor over 'scaffold'" in text
    assert "vocabulary" in text
