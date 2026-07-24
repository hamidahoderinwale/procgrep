"""Leave-one-group-out predictive transfer probe.

Trains a classifier on all-but-one group's fingerprints and tests on
the held-out group. When labels are 1:1 with groups the held-out
label is absent from training, so the probe measures structural
novelty. When labels repeat across groups (controlled-eval arms with
shared targets), it's a standard OOD generalization test.

Defaults to multinomial L2-regularized logistic regression; pass any
sklearn-compatible factory to swap it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

from procgrep.encode import Fingerprint

ClassifierFactory = Callable[[int], Any]
"""``seed -> fresh sklearn estimator``."""


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a leave-one-group-out probe.

    Attributes:
        groups: Group names in sorted order.
        per_group_accuracy: Held-out group -> classifier accuracy.
        overall_accuracy: Mean of ``per_group_accuracy``.
        confusion: ``true_group -> predicted_label -> count``.
    """

    groups: tuple[str, ...]
    per_group_accuracy: dict[str, float]
    overall_accuracy: float
    confusion: dict[str, dict[str, int]]


def leave_one_group_out(
    fingerprints: Iterable[Fingerprint],
    *,
    label_field: str = "group",
    classifier: ClassifierFactory | None = None,
    seed: int = 0,
) -> ProbeResult:
    """Run the leave-one-group-out probe.

    Args:
        label_field: ``"group"`` predicts the group label;
            ``"agent"`` predicts the agent name.
        classifier: Factory ``seed -> sklearn estimator``. Defaults
            to multinomial L2 logistic regression.
    """
    fps = list(fingerprints)
    if not fps:
        raise ValueError("no fingerprints supplied to probe")

    x = np.stack([fp.distribution() for fp in fps], axis=0)
    groups = np.array([fp.group for fp in fps])
    labels = _extract_labels(fps, label_field)

    # Validate the design before sklearn so degenerate input fails with a
    # domain-specific message instead of an opaque sklearn error. The
    # split needs at least two groups to hold one out, and the classifier
    # needs at least two classes to learn a decision boundary.
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        raise ValueError(f"probe needs >=2 groups; got {n_groups}")
    n_classes = len(set(labels.tolist()))
    if n_classes < 2:
        raise ValueError(f"probe needs >=2 {label_field} classes; got {n_classes}")

    factory: ClassifierFactory = classifier if classifier is not None else _default_classifier

    confusion: dict[str, dict[str, int]] = {}
    per_group_accuracy: dict[str, float] = {}

    splitter = LeaveOneGroupOut()
    for train_idx, test_idx in splitter.split(x, labels, groups):
        held_out = groups[test_idx[0]]
        model = factory(seed)
        model.fit(x[train_idx], labels[train_idx])
        predictions = model.predict(x[test_idx])

        per_group_accuracy[held_out] = float(np.mean(predictions == labels[test_idx]))
        bucket = confusion.setdefault(held_out, {})
        for predicted in predictions:
            bucket[predicted] = bucket.get(predicted, 0) + 1

    sorted_groups = tuple(sorted(per_group_accuracy))
    overall = float(np.mean([per_group_accuracy[g] for g in sorted_groups]))
    return ProbeResult(
        groups=sorted_groups,
        per_group_accuracy=per_group_accuracy,
        overall_accuracy=overall,
        confusion=confusion,
    )


def _extract_labels(fps: list[Fingerprint], label_field: str) -> npt.NDArray[np.str_]:
    if label_field == "group":
        return np.array([fp.group for fp in fps])
    if label_field == "agent":
        return np.array([fp.agent for fp in fps])
    raise ValueError(f"label_field must be 'group' or 'agent', got {label_field!r}")


def _default_classifier(seed: int) -> LogisticRegression:
    """Default estimator: multinomial L2 logistic regression.

    Sklearn 1.7 removed ``multi_class``; the solver auto-selects
    multinomial behavior when there are more than two classes.
    """
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )


__all__ = ["ClassifierFactory", "ProbeResult", "leave_one_group_out"]
