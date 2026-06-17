"""Scaffold-native emitters: render a `ProcedureSpec` into a coding agent
harness's own customization format.

`reward.ProcedureSpec.to_prompt` produces generic system-prompt text. That
text is portable but not first-class to any one harness: a user still has to
know where their scaffold wants prompt rules and paste it by hand into the
right field. These emitters close that gap by rendering the same spec into the
exact file a given scaffold consumes.

procgrep stays model-free. These are pure string renderers: they take a spec
and return text the caller drops into their scaffold's config tree. They never
run, wrap, or call an agent, and they hold no model.

Supported targets:

* SWE-agent (`to_swe_agent_config`): a YAML config fragment whose
  ``agent.templates.system_template`` carries the procedural rules. The user
  merges it into their SWE-agent config with ``--config``.
* OpenHands (`to_openhands_skill`): a Skill markdown file for
  ``.openhands/skills/<name>/SKILL.md``: YAML frontmatter (``name``,
  ``description``) plus a markdown body that augments the system prompt.

Design decisions:

* Reuse `spec.to_prompt` as the single source of the rule prose, then wrap it
  in each scaffold's envelope. Benefit: the procedural content stays identical
  across the generic prompt and every scaffold rendering, so a spec change
  propagates everywhere without per-target editing. Price: each emitter is a
  thin presentation layer and cannot express scaffold-specific knobs the prompt
  prose does not already carry.
* Emit text, not a parsed config object or a written file. Benefit: the caller
  owns where it lands and how it merges into an existing config, and the
  emitter has no filesystem side effects. Price: the caller does the paste or
  the file write.
* Hand-render the YAML fragment rather than dumping a dict through PyYAML.
  Benefit: a stable, block-scalar layout the user can read and diff, with no
  dependency on PyYAML being installed for emission. Price: one small escaping
  rule (indent the prompt body under a literal block scalar) we own here.

Guard-mode mapping (see `to_swe_agent_config` for the per-scaffold note):
the `program.GuardArtifact` patterns are control-flow assertions over the atom
stream, so they map to a scaffold's history-processing or step hook rather than
to a prompt template. procgrep emits the generic `GuardArtifact`; wiring it
into a specific hook is left to the scaffold integration and documented, not
built here.
"""

from __future__ import annotations

from procgrep.reward import ProcedureSpec

_SWE_AGENT_INDENT = "      "


def to_swe_agent_config(spec: ProcedureSpec) -> str:
    """Render the spec as a SWE-agent config fragment.

    Returns a YAML fragment carrying the spec's procedural rules in
    ``agent.templates.system_template`` as a literal block scalar. The user
    merges it into their SWE-agent config by passing it with ``--config``;
    SWE-agent shows ``system_template`` to the agent once at the start of a
    trajectory, which is where these standing procedural rules belong.

    Guard-mode mapping: the spec's guard patterns (`spec.to_patterns`, surfaced
    as `program.GuardArtifact`) are not prompt rules. They are assertions over
    the running atom stream, so in SWE-agent they map to a history processor
    under ``agent.history_processors`` (or an equivalent per-step control-flow
    hook): run the `GuardArtifact.check` callable against the atoms decoded so
    far and act on a returned violation, for example by injecting a corrective
    observation or truncating history. This emitter renders the prompt envelope
    only; wiring `GuardArtifact` into a history processor is left to the
    scaffold integration and is documented here rather than built.
    """
    body = _indent_block(spec.to_prompt(), _SWE_AGENT_INDENT)
    return (
        f"# SWE-agent config fragment derived from procgrep spec {spec.name!r}.\n"
        "# Merge into your SWE-agent config with --config; the procedural rules\n"
        "# are injected once via agent.templates.system_template.\n"
        "agent:\n"
        "  templates:\n"
        "    system_template: |-\n"
        f"{body}\n"
    )


def to_openhands_skill(spec: ProcedureSpec) -> str:
    """Render the spec as an OpenHands Skill markdown file.

    Returns the full contents for ``.openhands/skills/<name>/SKILL.md``: YAML
    frontmatter (``name``, ``description``) followed by a markdown body holding
    the spec's procedural rules. OpenHands loads the Skill into the system
    prompt, so the rules augment the agent's standing instructions.

    The frontmatter ``name`` is the spec name; OpenHands matches the Skill's
    name to its parent folder, so place this file at
    ``.openhands/skills/<spec.name>/SKILL.md``.
    """
    description = (
        f"Procedural guidance derived from procgrep spec {spec.name!r}: "
        "ordered phases to follow and failure patterns to avoid when solving a "
        "coding task."
    )
    body = spec.to_prompt()
    return (
        "---\n"
        f"name: {_yaml_scalar(spec.name)}\n"
        f"description: {_yaml_scalar(description)}\n"
        "---\n"
        f"# {spec.name}\n"
        "\n"
        f"{body}\n"
    )


def _yaml_scalar(value: str) -> str:
    """Render a string as a safe single-line YAML scalar.

    The spec name and the derived description can contain a colon, which a
    bare YAML scalar would misread as a mapping. Double-quoting with the two
    in-quote escapes YAML needs keeps the value a single plain string.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _indent_block(text: str, indent: str) -> str:
    """Indent every line of ``text`` by ``indent`` for a YAML block scalar.

    Blank lines stay blank so the literal block scalar keeps clean line breaks
    rather than carrying trailing whitespace.
    """
    return "\n".join(indent + line if line else line for line in text.split("\n"))


__all__ = [
    "to_openhands_skill",
    "to_swe_agent_config",
]
