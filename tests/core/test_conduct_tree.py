"""Conduct tree: distance, tree, poset, next-action forecast on controlled corpora."""

from __future__ import annotations

import random

import numpy as np
import pytest

from procgrep.conduct_tree import (
    build_conduct_tree,
    conditional_distance,
    encode_cells,
    next_action_scores,
    poset_distance,
    precedence_poset,
    split_half_floor,
)


def _runs(rng: random.Random, n: int, *, test_after_edit: float) -> list[list[str]]:
    """Synthetic agent: search, read, then edit loops that test with a given probability."""
    out = []
    for _ in range(n):
        run = ["search_repo", "read_file", "think"]
        for _ in range(rng.randint(2, 6)):
            run.append("edit")
            if rng.random() < test_after_edit:
                run.append("run_test")
        run.append("submit")
        out.append(run)
    return out


@pytest.fixture
def cells() -> dict[str, list[list[str]]]:
    rng = random.Random(0)
    return {
        "tester_a": _runs(rng, 300, test_after_edit=0.9),
        "tester_b": _runs(rng, 300, test_after_edit=0.9),
        "skipper": _runs(rng, 300, test_after_edit=0.1),
    }


def test_encode_drops_think_and_shares_alphabet(cells):
    encoded = encode_cells(cells)
    assert "think" not in encoded[0].actions
    assert all(c.actions == encoded[0].actions and c.states == encoded[0].states for c in encoded)
    assert encoded[0].counts.shape[0] == 300


def test_identical_policies_sit_at_the_floor_and_different_ones_do_not(cells):
    encoded = encode_cells(cells)
    same = conditional_distance(encoded[0].table(), encoded[1].table())[0]
    diff, contrib = conditional_distance(encoded[0].table(), encoded[2].table())
    floor = split_half_floor(encoded[0], reps=20)
    assert same < 2 * floor["p95"]
    assert diff > 10 * floor["p95"]
    # the split is after `edit`: that state must carry the distance
    assert encoded[0].states[int(np.argmax(contrib))] == "edit"


def test_tree_merges_the_twins_first_with_full_support(cells):
    tree = build_conduct_tree(encode_cells(cells), n_bootstrap=30, floor_reps=10)
    first = tree.merges[0]
    assert set(first.left) | set(first.right) == {"tester_a", "tester_b"}
    assert first.support == 1.0
    last = tree.merges[-1]
    assert last.top_states[0]["state"] == "edit"
    sides = {last.top_states[0]["left_next"]["action"], last.top_states[0]["right_next"]["action"]}
    assert sides == {"run_test", "edit"} or sides == {"run_test", "submit"}


def test_poset_recovers_the_phase_order(cells):
    poset = precedence_poset(cells["tester_a"])
    assert ("search_repo", "read_file") in {tuple(e) for e in poset["hasse"]}
    assert ("read_file", "edit") in {tuple(e) for e in poset["hasse"]}
    # transitive reduction: search -> edit is implied, not drawn
    assert ("search_repo", "edit") not in {tuple(e) for e in poset["hasse"]}
    assert poset_distance(poset, precedence_poset(cells["tester_b"])) < 0.05


def test_own_table_forecasts_better_than_unigram(cells):
    encoded = encode_cells(cells)
    pooled = sum(c.table() for c in encoded)
    scores = next_action_scores(encoded[2], pooled)
    assert scores["own"]["cross_entropy_bits"] < scores["unigram"]["cross_entropy_bits"]
    assert scores["own"]["cross_entropy_bits"] < scores["pooled"]["cross_entropy_bits"]
