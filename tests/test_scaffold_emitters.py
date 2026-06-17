"""Tests for `procgrep.scaffolds` and the `enforce(..., scaffold=...)` dispatch.

Covers that each scaffold-native emitter produces non-empty, well-formed output
that carries the spec's rule content, that the rule prose round-trips through
the envelope, and that `enforce` dispatches the prompt mode to the right
emitter for each scaffold.
"""

from __future__ import annotations

import pytest
import yaml

from procgrep.program import enforce
from procgrep.reward import Penalty, Phase, ProcedureSpec
from procgrep.scaffolds import to_openhands_skill, to_swe_agent_config
from procgrep.types import ATOM_EDIT, ATOM_READ_FILE, ATOM_RUN_TEST


def _spec() -> ProcedureSpec:
    """A spec with both a phase and a penalty so rule prose is non-trivial."""
    return ProcedureSpec(
        phases=(
            Phase(name="explore", reward=0.3, require_any=(ATOM_READ_FILE,)),
            Phase(name="verify", reward=0.3, require_any=(ATOM_RUN_TEST,)),
        ),
        penalties=(Penalty(name="edit_streak", reward=0.15, max_run=2),),
        name="my_procedure",
    )


def _rule_lines(spec: ProcedureSpec) -> list[str]:
    """Non-blank lines of the generic prompt: the rule content to look for."""
    return [line.strip() for line in spec.to_prompt().split("\n") if line.strip()]


# --- SWE-agent emitter ------------------------------------------------------


def test_swe_agent_config_is_non_empty_and_parses_as_yaml() -> None:
    spec = _spec()
    out = to_swe_agent_config(spec)
    assert out.strip()
    parsed = yaml.safe_load(out)
    assert isinstance(parsed, dict)
    assert "agent" in parsed


def test_swe_agent_config_carries_rules_in_system_template() -> None:
    spec = _spec()
    parsed = yaml.safe_load(to_swe_agent_config(spec))
    system_template = parsed["agent"]["templates"]["system_template"]
    assert isinstance(system_template, str)
    for line in _rule_lines(spec):
        assert line in system_template


def test_swe_agent_config_round_trips_prompt_body() -> None:
    """The literal block scalar reproduces the generic prompt verbatim."""
    spec = _spec()
    parsed = yaml.safe_load(to_swe_agent_config(spec))
    assert parsed["agent"]["templates"]["system_template"] == spec.to_prompt()


# --- OpenHands emitter ------------------------------------------------------


def test_openhands_skill_is_non_empty_with_frontmatter() -> None:
    spec = _spec()
    out = to_openhands_skill(spec)
    assert out.strip()
    assert out.startswith("---\n")
    assert out.count("---\n") >= 2  # opening and closing frontmatter fences


def test_openhands_skill_frontmatter_has_name_and_description() -> None:
    spec = _spec()
    out = to_openhands_skill(spec)
    _, frontmatter, _body = out.split("---\n", 2)
    meta = yaml.safe_load(frontmatter)
    assert meta["name"] == spec.name
    assert isinstance(meta["description"], str)
    assert meta["description"].strip()


def test_openhands_skill_body_carries_rules() -> None:
    spec = _spec()
    out = to_openhands_skill(spec)
    _, _frontmatter, body = out.split("---\n", 2)
    for line in _rule_lines(spec):
        assert line in body


def test_openhands_skill_round_trips_prompt_body() -> None:
    spec = _spec()
    out = to_openhands_skill(spec)
    _, _frontmatter, body = out.split("---\n", 2)
    assert spec.to_prompt() in body


# --- enforce dispatch -------------------------------------------------------


def test_enforce_scaffold_none_is_generic_prompt() -> None:
    spec = _spec()
    assert enforce(spec, mode="prompt", scaffold=None) == spec.to_prompt()


def test_enforce_scaffold_swe_agent_dispatches_to_emitter() -> None:
    spec = _spec()
    assert enforce(spec, mode="prompt", scaffold="swe-agent") == to_swe_agent_config(spec)


def test_enforce_scaffold_openhands_dispatches_to_emitter() -> None:
    spec = _spec()
    assert enforce(spec, mode="prompt", scaffold="openhands") == to_openhands_skill(spec)


def test_enforce_unknown_scaffold_raises() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="unknown scaffold"):
        enforce(spec, mode="prompt", scaffold="cursor")  # type: ignore[arg-type]


def test_emitters_differ_across_scaffolds() -> None:
    """Each scaffold wraps the rules in its own distinct envelope."""
    spec = _spec()
    generic = enforce(spec, mode="prompt")
    swe = enforce(spec, mode="prompt", scaffold="swe-agent")
    openhands = enforce(spec, mode="prompt", scaffold="openhands")
    assert generic != swe != openhands
    assert generic != openhands


def test_emitters_handle_empty_spec() -> None:
    """An empty spec still yields well-formed, non-empty scaffold output."""
    spec = ProcedureSpec(name="empty_proc")
    swe = yaml.safe_load(to_swe_agent_config(spec))
    assert swe["agent"]["templates"]["system_template"].strip()
    out = to_openhands_skill(spec)
    _, frontmatter, body = out.split("---\n", 2)
    assert yaml.safe_load(frontmatter)["name"] == "empty_proc"
    assert body.strip()


def test_edit_atom_referenced_in_penalty_prose() -> None:
    """A real spec's penalty prose reaches the scaffold output unchanged."""
    spec = ProcedureSpec(
        penalties=(Penalty(name="streak", reward=0.15, max_run=3),),
        name="p",
    )
    assert ATOM_EDIT  # atom constant exists and is non-empty
    sentence = spec.to_prompt()
    assert "3 edits in a row" in sentence
    assert "3 edits in a row" in to_swe_agent_config(spec)
    assert "3 edits in a row" in to_openhands_skill(spec)


__all__: list[str] = []
