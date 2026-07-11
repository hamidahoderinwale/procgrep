"""Guard-mode enforcement host: a mini-swe-agent agent that consults a `ProcedureGuard`.

Intent: the runtime half of guard enforcement. The core emits the guard
(`ProcedureGuard` wraps `enforce(spec, "guard")` + `enforce(spec, "decode")`);
this `DefaultAgent` subclass is the scaffold-side host that checks every
action before it runs. Read this when changing how a violation is applied at
execute time.

Design decisions:

1. A hard-blocked action is substituted with a harmless echo notice, not
   skipped. Benefit: the model receives an observation with the normal
   schema and adapts on its own. Price: the blocked attempt still spends a
   step, and the trajectory's assistant message keeps the attempted command
   (measurement sees attempts; `guard_events` records what was blocked).
2. The guard classifies actions itself (`GuardDecision.atom`), so this module
   never imports an atom classifier. Benefit: one classifier, owned by the
   core adapter. Price: the guard must be given the raw command string.
3. The guard prefix advances only for actions that actually run; a blocked
   action's atom is never committed (mirrors `ProcedureGuard.step`).
"""

from __future__ import annotations

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import Submitted

from procgrep.guard import ProcedureGuard

_BLOCK_NOTICE = (
    "echo '[procedure-guard] action blocked by the procedure rule; choose a different action.'"
)


class GuardedAgent(DefaultAgent):
    """`DefaultAgent` that applies a `ProcedureGuard` before each action.

    With ``guard=None`` it behaves exactly like `DefaultAgent` while still
    recording ``action_count``, so both arms of a paired run share one agent
    class and differ only in configuration.
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        guard: ProcedureGuard | None = None,
        config_class: type = AgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.guard = guard
        self.guard_events: list[dict] = []
        self.action_index = 0
        if self.guard is not None:
            self.guard.reset()

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs: list[dict] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            steer_after = self._apply_guard(action)
            run = action
            if steer_after == "__blocked__":
                run = {**action, "command": _BLOCK_NOTICE}
                steer_after = None
            try:
                outputs.append(self.env.execute(run))
            except Submitted:
                self.action_index += 1
                raise
            self.action_index += 1
            if steer_after:
                self.add_messages(
                    self.model.format_message(
                        role="user", content=steer_after, extra={"procgrep_guard_steer": True}
                    )
                )
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )

    def _apply_guard(self, action: dict) -> str | None:
        """Check one action; return a steer message, ``"__blocked__"``, or None.

        The prefix advances only for actions that actually run (allowed,
        steered, or warned), never for a hard-blocked one.
        """
        if self.guard is None:
            return None
        decision = self.guard.check(str(action.get("command", "")))
        self.guard_events.append(
            {
                "action_index": self.action_index,
                "atom": decision.atom,
                "allowed": decision.allowed,
                "directive": decision.directive,
                "reason": decision.reason,
            }
        )
        if not decision.allowed and decision.directive == "block":
            return "__blocked__"
        self.guard.commit(decision.atom)
        if not decision.allowed and decision.directive == "steer":
            return decision.steer_message
        return None

    def serialize(self, *extra_dicts) -> dict:
        data = super().serialize(*extra_dicts)
        runner: dict = {"action_count": self.action_index}
        if self.guard is not None:
            events = self.guard_events
            runner["guard"] = {
                # Private guard attributes: ProcedureGuard exposes no metadata
                # accessors yet; recorded here so the trajectory is self-describing.
                "spec": self.guard._spec.name,
                "on_violation": self.guard._on_violation,
                "checks": len(events),
                "blocked": sum(1 for e in events if not e["allowed"] and e["directive"] == "block"),
                "steered": sum(1 for e in events if not e["allowed"] and e["directive"] == "steer"),
                "events": events,
            }
        data["procgrep_runner"] = runner
        return data
