"""Leave-one-group-out predictive transfer probe.

The probe answers: *given fingerprints labeled by group, can a
classifier trained on all-but-one group correctly identify the held-
out group's label?* When labels are tied 1:1 to groups (e.g., the
"group" is also the prediction target), the held-out label is by
construction absent from training, and the probe quantifies
structural novelty rather than accuracy.

When labels can repeat across groups (the common controlled-eval
case, where each arm is a group but the prediction target is shared
across arms), the probe is a standard out-of-distribution
generalization test.

The implementation uses scikit-learn's `LeaveOneGroupOut` for fold
construction and a multinomial logistic regression with L2
regularization as the default classifier; this matches the choices
in the originating paper. The classifier choice is exposed via a
hook so projects can swap in any sklearn-compatible estimator.
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
"""A callable that takes a seed and returns a fresh sklearn estimator."""


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a leave-one-group-out probe.

    Attributes:
        groups: Group names in canonical (sorted) order.
        per_group_accuracy: Mapping from held-out group name to the
            classifier's accuracy on that held-out group.
        overall_accuracy: Mean accuracy across all held-out groups.
        confusion: Nested mapping ``true_group -> predicted_label ->
            count``, giving the distribution of predictions made on
            each held-out group.
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
        fingerprints: Fingerprints to probe. Must carry consistent
            ``group`` labels.
        label_field: Either ``"group"`` (predict the group label) or
            ``"agent"`` (predict the agent name).
        classifier: Optional factory ``seed -> sklearn estimator``.
            Defaults to a multinomial logistic regression with L2
            regularization.
        seed: Random seed forwarded to the classifier factory.

    Returns:
        A `ProbeResult` with per-group accuracy and confusion.
    """
    fps = list(fingerprints)
    if not fps:
        raise ValueError("no fingerprints supplied to probe")

    x = np.stack([fp.distribution() for fp in fps], axis=0)
    groups = np.array([fp.group for fp in fps])
    labels = _extract_labels(fps, label_field)

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
    """Pull the prediction labels out of the fingerprints."""
    if label_field == "group":
        return np.array([fp.group for fp in fps])
    if label_field == "agent":
        return np.array([fp.agent for fp in fps])
    raise ValueError(f"label_field must be 'group' or 'agent', got {label_field!r}")


def _default_classifier(seed: int) -> LogisticRegression:
    """The probe's default estimator: multinomial logistic regression.

    Sklearn 1.7 removed the ``multi_class`` argument; multinomial
    behavior is selected automatically by the solver when the label
    set has more than two classes.
    """
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )


__all__ = ["ClassifierFactory", "ProbeResult", "leave_one_group_out"]
