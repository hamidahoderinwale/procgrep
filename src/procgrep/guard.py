"""Stateful runtime guard for execute-time procedure enforcement.

`enforce(spec, "decode")` and `enforce(spec, "guard")` return stateless
prefix functions. `ProcedureGuard` wraps them into a stateful checker a host
calls before each action: it maintains the running atom prefix, classifies a
proposed action to an atom, and reports whether taking it keeps the trajectory
legal under the spec. The host enforces the decision (block / steer / warn).

procgrep stays model-free: this never runs the agent. It is the preventive,
execute-time counterpart to the model-free `enforce` artifacts, and it hard-
blocks only what the spec forbids (an edit-streak past its cap, a forbidden
sequence). Phases are reward signals, not hard constraints, so a spec with only
phases never blocks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from procgrep.program import enforce
from procgrep.reward import ProcedureSpec
from procgrep.types import CANONICAL_ATOMS, Atom
from dataclasses import dataclass

OnViolation = Literal["block", "steer", "warn"]
Directive = Literal["allow", "block", "steer", "warn"]


@dataclass(frozen=True)
class GuardDecision:
    """One check of a proposed action against the spec.

    Attributes:
        allowed: True when the proposed atom keeps the trajectory legal.
        atom: The proposed action's classified canonical atom.
        directive: What the host should do: ``"allow"`` when legal, else
            the guard's ``on_violation`` policy (``"block"`` / ``"steer"``
            / ``"warn"``).
        reason: Why the action was blocked; empty when allowed.
        allowed_atoms: Atoms that would keep the trajectory legal next.
            Never empty, so the host can always offer a legal move.
        steer_message: Nudge text to inject, populated only when
            ``directive == "steer"``.
        violated_guards: Names of must-hold-false guards the action would
            newly trigger.
    """

    allowed: bool
    atom: Atom
    directive: Directive
    reason: str
    allowed_atoms: tuple[Atom, ...]
    steer_message: str | None
    violated_guards: tuple[str, ...]


def _default_classifier(action: Any) -> Atom:
    """Classify a shell-command-shaped action to an atom (lazy import)."""
    from procgrep.ingest.adapters.mini_swe_agent import _classify_command

    return _classify_command(str(action))


class ProcedureGuard:
    """Stateful, execute-time guard built from a `ProcedureSpec`.

    Call `check(action)` before executing an action to get a `GuardDecision`
    without advancing state, or `step(action)` to check and advance the prefix
    in one call (the prefix advances unless the action was hard-blocked). An
    action is either a canonical atom (used directly) or anything else, which is
    passed to `classifier` (default: the shared bash-command classifier).
    """

    def __init__(
        self,
        spec: ProcedureSpec,
        *,
        on_violation: OnViolation = "block",
        classifier: Callable[[Any], Atom] | None = None,
        steer_message: str | None = None,
    ) -> None:
        self._spec = spec
        self._decode = enforce(spec, "decode")
        self._guard = enforce(spec, "guard")
        self._on_violation: OnViolation = on_violation
        self._classifier = classifier or _default_classifier
        self._steer = steer_message if steer_message is not None else spec.to_prompt()
        self._prefix: list[Atom] = []

    @property
    def prefix(self) -> tuple[Atom, ...]:
        """The atoms committed so far."""
        return tuple(self._prefix)

    def reset(self) -> None:
        """Clear the prefix to start a fresh trajectory."""
        self._prefix = []

    def commit(self, action: Any) -> Atom:
        """Advance the prefix by the action's atom, with no checking. Returns the atom.

        Use when the host has already decided to run an action and just needs
        the guard's state to track it.
        """
        atom = self._to_atom(action)
        self._prefix.append(atom)
        return atom

    def _to_atom(self, action: Any) -> Atom:
        if isinstance(action, str) and action in CANONICAL_ATOMS:
            return action
        return self._classifier(action)

    def check(self, action: Any) -> GuardDecision:
        """Decide on a proposed action without committing it to the prefix."""
        atom = self._to_atom(action)
        allowed_set = self._decode.allowed(self._prefix)
        before = set(self._guard.check(self._prefix))
        after = set(self._guard.check([*self._prefix, atom]))
        newly_violated = tuple(sorted(after - before))
        legal = atom in allowed_set and not newly_violated
        allowed_atoms = tuple(sorted(allowed_set))
        if legal:
            return GuardDecision(True, atom, "allow", "", allowed_atoms, None, ())
        return GuardDecision(
            allowed=False,
            atom=atom,
            directive=self._on_violation,
            reason=_reason(atom, allowed_set, newly_violated),
            allowed_atoms=allowed_atoms,
            steer_message=self._steer if self._on_violation == "steer" else None,
            violated_guards=newly_violated,
        )

    def step(self, action: Any) -> GuardDecision:
        """Check, then advance the prefix unless the action was hard-blocked.

        Under ``on_violation="block"`` a blocked action does not run, so its
        atom is not committed. Under ``"steer"`` / ``"warn"`` the action still
        runs (the host redirects or logs), so the atom is committed.
        """
        decision = self.check(action)
        if decision.allowed or self._on_violation != "block":
            self._prefix.append(decision.atom)
        return decision


def _reason(atom: Atom, allowed_set: set[Atom], violated: Sequence[str]) -> str:
    """Explain why an atom was blocked."""
    parts: list[str] = []
    if atom not in allowed_set:
        parts.append(f"{atom!r} would commit a forbidden step (edit-streak cap or forbidden sequence)")
    if violated:
        parts.append(f"would trigger guard(s): {', '.join(violated)}")
    return "; ".join(parts) or f"{atom!r} not permitted after the current prefix"


__all__ = ["Directive", "GuardDecision", "OnViolation", "ProcedureGuard"]
