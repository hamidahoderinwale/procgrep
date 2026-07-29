"""Online intent-cliff detection over action-only atom streams.

A cliff is a sharp change in the local action-type distribution (JSD window).
Used by the live view panel to surface mid-run procedure shifts. Matches the
validated settings from plateau (window=5, session 90th-percentile threshold).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from procgrep.jsd import jsd


@dataclass(frozen=True)
class CliffSignal:
    index: int
    jsd: float
    threshold: float


@dataclass
class OnlineCliffDetector:
    """Sliding JSD-window detector; fires when score exceeds session quantile.

    Scores index ``i`` only after ``window`` trailing actions exist, so detection
    lags the true boundary by ``window`` actions.
    """

    window: int = 5
    quantile: float = 0.90
    min_scores: int = 5
    stream: list[str] = field(default_factory=list)
    jsd_scores: dict[int, float] = field(default_factory=dict)
    cliff_indices: list[int] = field(default_factory=list)

    def append(self, atom: str) -> CliffSignal | None:
        """Append one atom; score the position ``window`` steps behind when ready."""
        self.stream.append(atom)
        if len(self.stream) < 2 * self.window:
            return None
        idx = len(self.stream) - self.window - 1
        vocab = sorted(set(self.stream))
        score = jsd_window_at(self.stream, idx, self.window, vocab)
        if score is None:
            return None
        self.jsd_scores[idx] = score
        if len(self.jsd_scores) < self.min_scores:
            return None
        threshold = float(np.quantile(list(self.jsd_scores.values()), self.quantile))
        if score >= threshold:
            self.cliff_indices.append(idx)
            return CliffSignal(index=idx, jsd=score, threshold=threshold)
        return None


def jsd_window_at(stream: list[str], idx: int, window: int, vocab: list[str]) -> float | None:
    """JSD between action distributions in ``[idx-window, idx)`` vs ``[idx, idx+window)``."""
    if idx < window or idx > len(stream) - window:
        return None
    before = _freq(stream[idx - window : idx], vocab)
    after = _freq(stream[idx : idx + window], vocab)
    return jsd(before, after)


def _freq(sub: list[str], vocab: list[str]) -> np.ndarray[Any, np.dtype[np.float64]]:
    counts = Counter(sub)
    vec = np.array([counts.get(atom, 0) for atom in vocab], dtype=float)
    total = vec.sum()
    return vec / total if total else vec


def flat_action_stream(turns: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    """Concatenate turn ``seq`` atoms; return prompt boundary indices (action offsets)."""
    stream: list[str] = []
    bounds: list[int] = []
    for i, turn in enumerate(turns):
        seq = list(turn.get("seq") or [])
        if i > 0 and seq:
            bounds.append(len(stream))
        stream.extend(seq)
    return stream, bounds


def _action_index_to_turn(turns: list[dict[str, Any]], action_idx: int) -> int:
    pos = 0
    for i, turn in enumerate(turns):
        n = len(turn.get("seq") or [])
        if pos <= action_idx < pos + n:
            return i
        pos += n
    return max(0, len(turns) - 1)


def _at_prompt(action_idx: int, prompt_bounds: list[int], *, tol: int = 1) -> bool:
    return any(abs(action_idx - b) <= tol for b in prompt_bounds)


def detect_cliffs(
    stream: list[str],
    *,
    window: int = 5,
    quantile: float = 0.90,
    min_scores: int = 5,
) -> list[CliffSignal]:
    """Replay *stream* through ``OnlineCliffDetector``; return fired signals."""
    detector = OnlineCliffDetector(window=window, quantile=quantile, min_scores=min_scores)
    out: list[CliffSignal] = []
    for atom in stream:
        signal = detector.append(atom)
        if signal is not None:
            out.append(signal)
    return out


def summarize_cliffs(
    turns: list[dict[str, Any]],
    *,
    window: int = 5,
    quantile: float = 0.90,
    min_scores: int = 5,
) -> dict[str, Any]:
    """Cliff summary for a panel session (action stream + prompt boundaries)."""
    stream, prompt_bounds = flat_action_stream(turns)
    n_actions = len(stream)
    empty: dict[str, Any] = {
        "window": window,
        "quantile": quantile,
        "n_actions": n_actions,
        "n_cliffs": 0,
        "n_hidden": 0,
        "n_at_prompt": 0,
        "hidden_fraction": None,
        "per_100_actions": 0.0,
        "events": [],
    }
    if n_actions < 2 * window:
        return empty

    signals = detect_cliffs(stream, window=window, quantile=quantile, min_scores=min_scores)
    events: list[dict[str, Any]] = []
    n_hidden = 0
    for sig in signals:
        at_p = _at_prompt(sig.index, prompt_bounds)
        if not at_p:
            n_hidden += 1
        events.append(
            {
                "index": sig.index,
                "turn": _action_index_to_turn(turns, sig.index),
                "jsd": round(sig.jsd, 4),
                "at_prompt": at_p,
            }
        )

    n_cliffs = len(signals)
    return {
        "window": window,
        "quantile": quantile,
        "n_actions": n_actions,
        "n_cliffs": n_cliffs,
        "n_hidden": n_hidden,
        "n_at_prompt": n_cliffs - n_hidden,
        "hidden_fraction": round(n_hidden / n_cliffs, 3) if n_cliffs else None,
        "per_100_actions": round(100 * n_cliffs / n_actions, 2) if n_actions else 0.0,
        "events": events,
    }


def enrich_panel_session(
    panel: dict[str, Any],
    *,
    window: int = 5,
    quantile: float = 0.90,
) -> dict[str, Any]:
    """Attach ``meta.cliffs`` and per-turn ``cliff_count`` for the live view."""
    turns = list(panel.get("turns") or [])
    summary = summarize_cliffs(turns, window=window, quantile=quantile)
    turn_counts = [0] * len(turns)
    for ev in summary["events"]:
        t = int(ev["turn"])
        if 0 <= t < len(turn_counts):
            turn_counts[t] += 1
    for turn, count in zip(turns, turn_counts, strict=False):
        turn["cliff_count"] = count
    panel.setdefault("meta", {})["cliffs"] = summary
    return panel


__all__ = [
    "CliffSignal",
    "OnlineCliffDetector",
    "detect_cliffs",
    "enrich_panel_session",
    "flat_action_stream",
    "jsd_window_at",
    "summarize_cliffs",
]
