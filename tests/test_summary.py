"""Tests for `procgrep.summary` -- diffing trace groups by summary metadata."""

from __future__ import annotations

from procgrep.summary import SummaryDiff, summary_diff
from procgrep.types import Trace


def _trace(tid: str, meta: dict) -> Trace:  # type: ignore[type-arg]
    return Trace(trace_id=tid, agent="x", atoms=["edit"], metadata=meta)


def test_summary_diff_reports_numeric_deltas_and_tool_jsd() -> None:
    a = [_trace("a1", {"autonomy": 2.0, "prompt_words": 10, "tools": {"Bash": 8, "Edit": 2}})]
    b = [_trace("b1", {"autonomy": 8.0, "prompt_words": 40, "tools": {"Edit": 8, "Read": 2}})]
    diff = summary_diff(a, b, label_a="cursor", label_b="cc")
    assert isinstance(diff, SummaryDiff)
    assert diff.deltas["autonomy"] == 6.0
    assert diff.deltas["prompt_words"] == 30.0
    assert diff.categorical_jsd["tools"] > 0.0  # different tool mixes diverge


def test_summary_diff_ignores_keys_not_in_both_groups() -> None:
    a = [_trace("a1", {"autonomy": 2.0, "only_a": 5})]
    b = [_trace("b1", {"autonomy": 4.0})]
    diff = summary_diff(a, b)
    assert "autonomy" in diff.deltas
    assert "only_a" not in diff.deltas  # not shared, so not diffed


def test_summary_diff_handles_absent_categorical() -> None:
    a = [_trace("a1", {"autonomy": 1.0})]
    b = [_trace("b1", {"autonomy": 2.0})]
    diff = summary_diff(a, b)
    assert diff.categorical_jsd == {}  # no tool dicts present
    assert diff.deltas["autonomy"] == 1.0


def test_variance_decomposition_separates_explanatory_factor() -> None:
    from procgrep.bpe import fit_bpe
    from procgrep.summary import variance_decomposition

    traces = []
    for i in range(6):
        traces.append(Trace(trace_id=f"a{i}", agent="x", atoms=["edit", "edit", "run_test"], metadata={"real": "A", "noise": str(i % 2)}))
    for i in range(6):
        traces.append(Trace(trace_id=f"b{i}", agent="x", atoms=["read_file", "read_file", "search_repo"], metadata={"real": "B", "noise": str(i % 2)}))
    vocab = fit_bpe([t.atoms for t in traces], vocab_size=12, seed=0)
    r = variance_decomposition(traces, vocab, ["real", "noise"], sample=100)
    assert r["real"] > 0.8   # the real grouping explains almost all variation
    assert r["noise"] < 0.3  # the arbitrary split explains little
