"""Conduct tree: which agents behave alike, where they part ways, what each does there.

Every agent is read as a policy on one shared directly-follows automaton whose
states are the last ``context_k`` actions. Two agents are compared state by
state -- the JSD between their next-action distributions, weighted by how often
the state is visited -- and the resulting distance matrix is clustered into a
dendrogram. Each merge in the tree is annotated with the states that contribute
most to the distance between its two sides and the most likely next action on
each side, so a split reads as "after X, side A does Y, side B does Z".

Companions: `precedence_poset` gives the order-of-operations skeleton (which
actions tend to happen first) as a Hasse diagram, and `next_action_scores`
asks whether an agent's own conduct table predicts its held-out runs better
than the pooled table -- past conduct forecasting future conduct.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt

from .jsd import jsd
from .types import Atom, AtomSequence

START: Atom = "<s>"
"""Synthetic state before the first action of a run."""

Matrix = npt.NDArray[np.float64]
SampleIndex = Sequence[int] | npt.NDArray[np.int64]


@dataclass(frozen=True)
class ConductCell:
    """One condition's runs encoded as per-run transition counts.

    ``counts`` has shape ``(n_runs, n_states, n_actions)``; ``states`` and
    ``actions`` name its axes. Built by `encode_cells` so every cell in a tree
    shares one alphabet.
    """

    name: str
    counts: npt.NDArray[np.int64]
    states: tuple[str, ...]
    actions: tuple[Atom, ...]

    @property
    def n_runs(self) -> int:
        return int(self.counts.shape[0])

    def table(self, index: SampleIndex | None = None) -> Matrix:
        """Summed transition counts over ``index`` (all runs when None)."""
        arr = self.counts if index is None else self.counts[np.asarray(index, dtype=np.intp)]
        return np.asarray(arr.sum(axis=0), dtype=np.float64)


@dataclass(frozen=True)
class Merge:
    """One internal node of the dendrogram."""

    left: tuple[str, ...]
    right: tuple[str, ...]
    height: float
    support: float | None
    top_states: list[dict[str, object]]


@dataclass(frozen=True)
class ConductTree:
    names: tuple[str, ...]
    distance: Matrix
    linkage: Matrix
    merges: list[Merge]
    floors: dict[str, dict[str, float]]
    context_k: int
    linkage_method: str
    n_bootstrap: int
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "distance": self.distance.tolist(),
            "linkage": self.linkage.tolist(),
            "merges": [
                {
                    "left": list(m.left),
                    "right": list(m.right),
                    "height": m.height,
                    "support": m.support,
                    "top_states": m.top_states,
                }
                for m in self.merges
            ],
            "floors": self.floors,
            "context_k": self.context_k,
            "linkage_method": self.linkage_method,
            "n_bootstrap": self.n_bootstrap,
            **self.extra,
        }


def encode_cells(
    cells: Mapping[str, Sequence[AtomSequence]],
    *,
    context_k: int = 1,
    exclude_atoms: Sequence[Atom] = ("think",),
) -> list[ConductCell]:
    """Encode every cell's runs as transition-count arrays over one shared alphabet.

    States are the last ``context_k`` actions joined with ``"|"``; runs shorter
    than the context start from `START` padding so the first actions still
    count. Atoms in ``exclude_atoms`` are dropped before encoding.
    """
    if context_k < 1:
        raise ValueError(f"context_k must be >= 1, got {context_k}")
    drop = set(exclude_atoms)
    cleaned = {
        name: [[a for a in run if a not in drop] for run in runs] for name, runs in cells.items()
    }
    actions = tuple(sorted({a for runs in cleaned.values() for run in runs for a in run}))
    if not actions:
        raise ValueError("no actions left after exclusion")
    state_set: set[str] = set()
    for runs in cleaned.values():
        for run in runs:
            for state, _ in _transitions(run, context_k):
                state_set.add(state)
    states = tuple(sorted(state_set))
    s_idx = {s: i for i, s in enumerate(states)}
    a_idx = {a: i for i, a in enumerate(actions)}
    out = []
    for name, runs in cleaned.items():
        counts = np.zeros((len(runs), len(states), len(actions)), dtype=np.int64)
        for r, run in enumerate(runs):
            for state, action in _transitions(run, context_k):
                counts[r, s_idx[state], a_idx[action]] += 1
        out.append(ConductCell(name=name, counts=counts, states=states, actions=actions))
    return out


def _transitions(run: Sequence[Atom], context_k: int) -> list[tuple[str, Atom]]:
    padded = [START] * context_k + list(run)
    return [("|".join(padded[i - context_k : i]), padded[i]) for i in range(context_k, len(padded))]


def conditional_distance(
    table_a: Matrix,
    table_b: Matrix,
) -> tuple[float, Matrix]:
    """Visit-weighted JSD between two agents' next-action distributions.

    Returns the distance and the per-state contribution vector (weight x JSD),
    which sums to the distance. Weight for a state is the mean of its visit
    share on the two sides, so the measure is symmetric. States visited by
    only one side are skipped (their share is reported by `unshared_mass`).
    Missing observations or no shared visited state raise ``ValueError``.
    """
    _validate_tables(table_a, table_b)
    visits_a = table_a.sum(axis=1)
    visits_b = table_b.sum(axis=1)
    if visits_a.sum() == 0 or visits_b.sum() == 0:
        raise ValueError("both conditions need observations to compare conduct")
    share_a = visits_a / visits_a.sum()
    share_b = visits_b / visits_b.sum()
    contrib = np.zeros(table_a.shape[0], dtype=np.float64)
    total_weight = 0.0
    for s in range(table_a.shape[0]):
        if visits_a[s] == 0 or visits_b[s] == 0:
            continue
        w = 0.5 * (share_a[s] + share_b[s])
        contrib[s] = w * jsd(table_a[s], table_b[s])
        total_weight += w
    if total_weight == 0:
        raise ValueError("conduct comparison needs a shared state with observations on both sides")
    contrib /= total_weight
    return float(contrib.sum()), contrib


def unshared_mass(table_a: Matrix, table_b: Matrix) -> float:
    """Share of visits (mean of both sides) in states only one side ever reaches."""
    _validate_tables(table_a, table_b)
    va, vb = table_a.sum(axis=1), table_b.sum(axis=1)
    if va.sum() == 0 or vb.sum() == 0:
        raise ValueError("both conditions need observations to measure unshared mass")
    only = (va == 0) | (vb == 0)
    return float(0.5 * (va[only].sum() / va.sum() + vb[only].sum() / vb.sum()))


def _validate_tables(table_a: Matrix, table_b: Matrix) -> None:
    if table_a.ndim != 2 or table_b.ndim != 2 or table_a.shape != table_b.shape:
        raise ValueError("conduct tables must have the same two-dimensional shape")
    if any(not np.isfinite(t).all() or (t < 0).any() for t in (table_a, table_b)):
        raise ValueError("conduct counts must be finite and non-negative")


def distance_matrix(
    cells: Sequence[ConductCell], index: Sequence[SampleIndex] | None = None
) -> Matrix:
    if cells and any(c.states != cells[0].states or c.actions != cells[0].actions for c in cells):
        raise ValueError("conduct cells must share the same ordered states and actions")
    if index is not None and len(index) != len(cells):
        raise ValueError("one resampling index is required per cell")
    tables = [c.table(None if index is None else index[i]) for i, c in enumerate(cells)]
    n = len(cells)
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = conditional_distance(tables[i], tables[j])[0]
    return m


def split_half_floor(
    cell: ConductCell,
    *,
    reps: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Distance between two random halves of one cell's runs: the same-condition noise.

    Reported at the half-n the split produces; this is a descriptive baseline,
    not a significance test. Both halves require observed transitions.
    """
    if cell.n_runs < 2:
        raise ValueError("a split-half floor needs at least two runs")
    if reps < 1:
        raise ValueError("floor repetitions must be positive")
    rng = np.random.default_rng(seed)
    n = cell.n_runs
    vals = []
    for _ in range(reps):
        perm = rng.permutation(n)
        vals.append(conditional_distance(cell.table(perm[: n // 2]), cell.table(perm[n // 2 :]))[0])
    arr = np.asarray(vals)
    return {"median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)), "n_half": n // 2}


def build_conduct_tree(
    cells: Sequence[ConductCell],
    *,
    linkage_method: str = "average",
    n_bootstrap: int = 200,
    seed: int = 0,
    floor_reps: int = 100,
    top_states: int = 3,
) -> ConductTree:
    """Cluster cells by conduct distance; annotate merges with support and deviation states.

    Support of a merge is the share of bootstrap trees (runs resampled with
    replacement within each cell) that contain the same clade. ``top_states``
    names the states contributing most to the distance between a merge's two
    sides, with each side's most likely next action there.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    if len(cells) < 2:
        raise ValueError("a tree needs at least two cells")
    if any(c.n_runs < 2 for c in cells):
        raise ValueError("each tree cell needs at least two runs for its split-half floor")
    if n_bootstrap < 0 or floor_reps < 1 or top_states < 0:
        raise ValueError(
            "bootstrap and top-state counts must be non-negative; floor repetitions must be positive"
        )
    names = tuple(c.name for c in cells)
    if len(set(names)) != len(names):
        raise ValueError("conduct cell names must be unique")
    dist = distance_matrix(cells)
    link = linkage(squareform(dist, checks=False), method=linkage_method)
    clades = _clades(link, len(cells))

    support: dict[frozenset[int], float] | None = None
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        hits: Counter[frozenset[int]] = Counter()
        for _ in range(n_bootstrap):
            idx = [rng.integers(0, c.n_runs, size=c.n_runs) for c in cells]
            boot_link = linkage(
                squareform(distance_matrix(cells, idx), checks=False), method=linkage_method
            )
            for clade in _clades(boot_link, len(cells)):
                hits[clade] += 1
        support = {c: hits[c] / n_bootstrap for c in clades}

    tables = [c.table() for c in cells]
    merges = []
    for row, (left, right) in zip(link, _merge_sides(link, len(cells)), strict=True):
        clade = frozenset(left | right)
        ta = np.stack([tables[i] for i in sorted(left)]).sum(axis=0)
        tb = np.stack([tables[i] for i in sorted(right)]).sum(axis=0)
        _, contrib = conditional_distance(ta, tb)
        order = np.argsort(-contrib)[:top_states]
        deviations = []
        for s in order:
            if contrib[s] <= 0:
                continue
            deviations.append(
                {
                    "state": cells[0].states[s],
                    "contribution": float(contrib[s]),
                    "left_next": _top_action(ta[s], cells[0].actions),
                    "right_next": _top_action(tb[s], cells[0].actions),
                }
            )
        merges.append(
            Merge(
                left=tuple(names[i] for i in sorted(left)),
                right=tuple(names[i] for i in sorted(right)),
                height=float(row[2]),
                support=None if support is None else support[clade],
                top_states=deviations,
            )
        )
    floors = {c.name: split_half_floor(c, reps=floor_reps, seed=seed) for c in cells}
    return ConductTree(
        names=names,
        distance=dist,
        linkage=link,
        merges=merges,
        floors=floors,
        context_k=len(cells[0].states[0].split("|")),
        linkage_method=linkage_method,
        n_bootstrap=n_bootstrap,
    )


def _top_action(row: Matrix, actions: Sequence[Atom]) -> dict[str, object]:
    total = row.sum()
    if total == 0:
        return {"action": None, "p": 0.0}
    i = int(np.argmax(row))
    return {"action": actions[i], "p": float(row[i] / total)}


def _merge_sides(link: Matrix, n: int) -> list[tuple[set[int], set[int]]]:
    members: dict[int, set[int]] = {i: {i} for i in range(n)}
    sides = []
    for k, row in enumerate(link):
        a, b = int(row[0]), int(row[1])
        sides.append((members[a], members[b]))
        members[n + k] = members[a] | members[b]
    return sides


def _clades(link: Matrix, n: int) -> list[frozenset[int]]:
    return [frozenset(a | b) for a, b in _merge_sides(link, n)]


def precedence_poset(
    runs: Sequence[AtomSequence],
    *,
    actions: Sequence[Atom] | None = None,
    exclude_atoms: Sequence[Atom] = ("think",),
    z: float = 1.96,
) -> dict[str, object]:
    """Order-of-operations skeleton from first-occurrence positions.

    For each ordered pair of actions, P(a before b) over runs containing both.
    A pair is kept when its Wilson interval excludes 0.5. These are unadjusted
    pairwise intervals, not simultaneous confidence for the graph. A cyclic
    relation raises ``ValueError`` instead of producing a Hasse diagram.
    """
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    drop = set(exclude_atoms)
    firsts = []
    for run in runs:
        seen: dict[Atom, int] = {}
        for i, a in enumerate(run):
            if a not in drop and a not in seen:
                seen[a] = i
        firsts.append(seen)
    acts = tuple(sorted(actions or {a for f in firsts for a in f}))
    pairs: dict[str, dict[str, float]] = {}
    relation: set[tuple[Atom, Atom]] = set()
    for a in acts:
        for b in acts:
            if a >= b:
                continue
            both = [(f[a], f[b]) for f in firsts if a in f and b in f]
            if not both:
                continue
            p = sum(1 for x, y in both if x < y) / len(both)
            lo, hi = _wilson_interval(p, len(both), z)
            pairs[f"{a}<{b}"] = {"p": p, "n": len(both), "lo": lo, "hi": hi}
            if lo > 0.5:
                relation.add((a, b))
            elif hi < 0.5:
                relation.add((b, a))
    hasse = _transitive_reduction(relation)
    return {"actions": list(acts), "pairs": pairs, "hasse": sorted(hasse), "n_runs": len(runs)}


def _wilson_interval(p: float, n: int, z: float) -> tuple[float, float]:
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def poset_distance(poset_a: Mapping[str, object], poset_b: Mapping[str, object]) -> float:
    """Mean absolute difference in P(a before b) over pairs both posets measured."""
    pa = cast(Mapping[str, Mapping[str, float]], poset_a["pairs"])
    pb = cast(Mapping[str, Mapping[str, float]], poset_b["pairs"])
    shared = set(pa) & set(pb)
    if not shared:
        raise ValueError("poset comparison needs shared observed action pairs")
    return float(np.mean([abs(pa[k]["p"] - pb[k]["p"]) for k in sorted(shared)]))


def _transitive_reduction(relation: set[tuple[Atom, Atom]]) -> set[tuple[Atom, Atom]]:
    succ: dict[Atom, set[Atom]] = {}
    for a, b in relation:
        succ.setdefault(a, set()).add(b)
    nodes = {a for edge in relation for a in edge}
    degree = Counter(b for _, b in relation)
    ready = [a for a in nodes if degree[a] == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for nxt in succ.get(node, ()):
            degree[nxt] -= 1
            if degree[nxt] == 0:
                ready.append(nxt)
    if visited != len(nodes):
        raise ValueError("estimated precedence relation contains a cycle; no partial order exists")

    def reaches(x: Atom, y: Atom, skip: tuple[Atom, Atom]) -> bool:
        stack = [x]
        seen = set()
        while stack:
            cur = stack.pop()
            for nxt in succ.get(cur, ()):
                if (cur, nxt) == skip or nxt in seen:
                    continue
                if nxt == y:
                    return True
                seen.add(nxt)
                stack.append(nxt)
        return False

    return {(a, b) for a, b in relation if not reaches(a, b, (a, b))}


def next_action_scores(
    cell: ConductCell,
    pooled_table: Matrix,
    *,
    seed: int = 0,
    alpha: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Does a cell's own conduct table forecast its held-out runs better than the pooled one?

    ``pooled_table`` must include all of this cell's transitions on the same
    ordered axes. This cell's held-out transitions are subtracted before
    fitting the pooled baseline. All models use add-``alpha`` smoothing.
    """
    if cell.n_runs < 2:
        raise ValueError("a forecast needs at least two runs")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    all_counts = cell.table()
    _validate_tables(all_counts, pooled_table)
    if (pooled_table < all_counts).any():
        raise ValueError("pooled_table must include all transitions from this cell")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(cell.n_runs)
    train, test = perm[: cell.n_runs // 2], perm[cell.n_runs // 2 :]
    own = cell.table(train)
    test_counts = cell.table(test)
    if own.sum() == 0 or test_counts.sum() == 0:
        raise ValueError("forecast training and test halves both need observations")
    pooled_train = pooled_table - test_counts
    unigram = np.tile(own.sum(axis=0), (own.shape[0], 1))
    out = {}
    for label, table in (("own", own), ("pooled", pooled_train), ("unigram", unigram)):
        probs = (table + alpha) / (table + alpha).sum(axis=1, keepdims=True)
        n = test_counts.sum()
        ce = float(-(test_counts * np.log2(probs)).sum() / n)
        top1 = float((test_counts[np.arange(len(probs)), probs.argmax(axis=1)]).sum() / n)
        out[label] = {"top1": top1, "cross_entropy_bits": ce}
    out["n_test_transitions"] = {"value": float(test_counts.sum())}
    return out
