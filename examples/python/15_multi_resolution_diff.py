"""Multi-resolution lineage diff in practice.

This example demonstrates running `lineage_diff` under multiple atom
alphabets in a single call, so a single audit produces both
cross-comparable canonical-level results AND scaffold-native richness.

We construct two small synthetic corpora that simulate SWE-agent
trajectories before and after a (made-up) fine-tune. Each trajectory
emits the scaffold's native action names (``str_replace``,
``str_replace_editor``, ``goto``, ``pytest``, etc.) rather than
already-canonicalized atoms. A simple projection callable maps each
native atom to its canonical equivalent at diff time, so we can
compute both views in one pass:

  - The CANONICAL view answers "did procedure categories shift?"
    (cross-comparable across scaffolds in a community catalog).
  - The NATIVE view answers "did scaffold-specific action choices
    shift?" (within-scaffold richness, not cross-comparable).

In the synthetic data below, the child uses ``str_replace_editor``
where the parent used ``str_replace`` -- procedurally equivalent at
the canonical level (both are EDITs) but different at the native
level. The canonical-level vocabulary axis preserves Jaccard 1.0;
the native-level axis reports drift. That's the multi-resolution
signal we get for free in one diff.

Run with::

    python examples/python/15_multi_resolution_diff.py
"""

from __future__ import annotations

from procgrep import lineage_diff
from procgrep.types import (
    ATOM_EDIT,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    Atom,
    Trace,
)

# ---------------------------------------------------------------------------
# Native-to-canonical projection.
#
# Maps SWE-agent native action names to procgrep's canonical 11-atom
# alphabet. In a real adapter this would live next to the adapter; here
# we surface it inline so the example is self-contained.
# ---------------------------------------------------------------------------

NATIVE_TO_CANONICAL: dict[str, Atom] = {
    # Edit family
    "str_replace": ATOM_EDIT,
    "str_replace_editor": ATOM_EDIT,
    # Read family
    "goto": ATOM_READ_FILE,
    "scroll_down": ATOM_READ_FILE,
    "scroll_up": ATOM_READ_FILE,
    "open": ATOM_READ_FILE,
    # Search family
    "find_file": ATOM_SEARCH_REPO,
    "search_dir": ATOM_SEARCH_REPO,
    # Test family
    "pytest": ATOM_RUN_TEST,
    # Submit family
    "submit": ATOM_SUBMIT,
}


def to_canonical(atom: Atom) -> Atom:
    """Project a native SWE-agent action name to its canonical atom."""
    return NATIVE_TO_CANONICAL.get(atom, ATOM_OTHER)


# ---------------------------------------------------------------------------
# Synthetic parent and child corpora.
#
# Each trace is one solved bug-fix attempt. The parent uses ``str_replace``
# for its edits; the child uses ``str_replace_editor``. Same canonical
# behavior (both are EDITs), different native action names. This is a
# common pattern when a fine-tune subtly shifts which tool variant the
# agent prefers without changing what it's doing categorically.
# ---------------------------------------------------------------------------


def _trace(trace_id: str, agent: str, atoms: list[str], resolved: bool) -> Trace:
    return Trace(
        trace_id=trace_id,
        agent=agent,
        atoms=atoms,
        group=agent,
        metadata={"resolved": resolved},
    )


parent_traces = [
    _trace(
        "p1",
        "claude-3-7-sonnet",
        ["find_file", "goto", "str_replace", "pytest", "submit"],
        resolved=True,
    ),
    _trace(
        "p2",
        "claude-3-7-sonnet",
        ["search_dir", "open", "str_replace", "str_replace", "pytest", "submit"],
        resolved=True,
    ),
    _trace(
        "p3",
        "claude-3-7-sonnet",
        ["find_file", "scroll_down", "str_replace", "pytest"],
        resolved=False,
    ),
]

child_traces = [
    _trace(
        "c1",
        "swe-agent-lm-32b",
        ["find_file", "goto", "str_replace_editor", "pytest", "submit"],
        resolved=True,
    ),
    _trace(
        "c2",
        "swe-agent-lm-32b",
        ["search_dir", "open", "str_replace_editor", "pytest", "submit"],
        resolved=True,
    ),
    _trace(
        "c3",
        "swe-agent-lm-32b",
        ["find_file", "str_replace_editor", "pytest"],
        resolved=False,
    ),
]


# ---------------------------------------------------------------------------
# Run the diff under both alphabets in a single call.
# ---------------------------------------------------------------------------

diff = lineage_diff(
    parent=parent_traces,
    child=child_traces,
    parent_label="Claude 3.7 Sonnet (synthetic)",
    child_label="SWE-agent-LM-32B (synthetic)",
    along=["vocabulary", "entropy", "outcome_quadrant"],
    outcome_field="resolved",
    alphabet=["canonical", "native"],
    canonical_projection=to_canonical,
)


# ---------------------------------------------------------------------------
# Show how the two alphabets differ at the axis level.
# ---------------------------------------------------------------------------

print("=" * 72)
print(f"LineageDiff: {diff.parent_label} -> {diff.child_label}")
print(f"  n_parent={diff.n_parent}, n_child={diff.n_child}")
print(f"  Total axis results: {len(diff.axes)}  (3 axes x 2 alphabets)")
print("=" * 72)

for alpha in ("canonical", "native"):
    print(f"\n--- alphabet = {alpha!r} ---")
    for axis in diff.axes:
        if axis.alphabet != alpha:
            continue
        print(f"  {axis.axis:18s}  summary={axis.summary_value:+.4f}")
        # Surface the key per-axis detail that explains the summary.
        if axis.axis == "vocabulary":
            shared = axis.detail.get("shared", [])
            parent_only = axis.detail.get("parent_only", [])
            child_only = axis.detail.get("child_only", [])
            print(f"    shared atoms     : {shared}")
            print(f"    parent-only atoms: {parent_only}")
            print(f"    child-only atoms : {child_only}")
        elif axis.axis == "entropy":
            print(
                f"    parent_mean={axis.detail['parent_mean']:.4f}  "
                f"child_mean={axis.detail['child_mean']:.4f}"
            )
        elif axis.axis == "outcome_quadrant":
            print(
                f"    pass_jaccard={axis.detail['pass_vocab_jaccard']:.4f}  "
                f"fail_jaccard={axis.detail['fail_vocab_jaccard']:.4f}"
            )

print("\n" + "=" * 72)
print("KEY OBSERVATION")
print("=" * 72)
print(
    """
At the CANONICAL level, vocabulary preservation is high -- both sides
emit EDITs, READ_FILEs, SEARCH_REPOs, RUN_TESTs, and SUBMITs. The
training procedure preserved the procedural categories perfectly.

At the NATIVE level, vocabulary preservation is LOWER -- the child
consistently chose `str_replace_editor` where the parent chose
`str_replace`. Both are valid EDIT actions, but the fine-tune shifted
the tool-variant preference. This is a scaffold-native shift that
would be invisible at the canonical level.

Single-alphabet audit would have missed this. Multi-alphabet audit
catches both the "categories preserved" finding (canonical) and the
"tool preference shifted" finding (native) in one call.
"""
)
