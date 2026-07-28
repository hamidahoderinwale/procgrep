"""Tests for procgrep.cliff — online intent-cliff detection."""

from __future__ import annotations

from procgrep.cliff import (
    OnlineCliffDetector,
    detect_cliffs,
    enrich_panel_session,
    flat_action_stream,
    summarize_cliffs,
)


def test_flat_action_stream_prompt_boundaries() -> None:
    turns = [
        {"seq": ["read_file", "read_file"]},
        {"seq": ["edit", "run_test"]},
        {"seq": ["search_repo"]},
    ]
    stream, bounds = flat_action_stream(turns)
    assert stream == ["read_file", "read_file", "edit", "run_test", "search_repo"]
    assert bounds == [2, 4]


def test_detect_cliffs_fires_on_regime_switch() -> None:
    explore = ["read_file", "search_repo"] * 8
    implement = ["edit", "run_test"] * 8
    stream = explore + implement
    signals = detect_cliffs(stream, window=3, quantile=0.85, min_scores=4)
    assert signals
    assert any(abs(s.index - len(explore)) <= 4 for s in signals)


def test_summarize_cliffs_hidden_vs_prompt() -> None:
    turns = [
        {"seq": ["read_file"] * 10},
        {"seq": ["edit"] * 10},
    ]
    summary = summarize_cliffs(turns, window=3, quantile=0.8, min_scores=3)
    assert summary["n_actions"] == 20
    assert summary["n_cliffs"] >= 1
    assert summary["per_100_actions"] > 0


def test_enrich_panel_session_attaches_meta_and_turn_counts() -> None:
    panel = {
        "meta": {"name": "t"},
        "turns": [
            {"seq": ["read_file"] * 12, "prompt": "a"},
            {"seq": ["edit"] * 12, "prompt": "b"},
        ],
    }
    out = enrich_panel_session(panel, window=3, quantile=0.8)
    assert "cliffs" in out["meta"]
    assert out["meta"]["cliffs"]["n_actions"] == 24
    assert sum(t.get("cliff_count", 0) for t in out["turns"]) == out["meta"]["cliffs"]["n_cliffs"]


def test_online_detector_streams_incrementally() -> None:
    det = OnlineCliffDetector(window=2, quantile=0.75, min_scores=2)
    fired = []
    for atom in ["read_file", "read_file", "read_file", "edit", "edit", "edit"]:
        sig = det.append(atom)
        if sig:
            fired.append(sig.index)
    batch = [s.index for s in detect_cliffs(det.stream, window=2, quantile=0.75, min_scores=2)]
    assert fired == batch
