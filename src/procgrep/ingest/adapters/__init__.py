"""Built-in trace adapters.

Importing this package registers each built-in adapter under its
canonical name in `procgrep.canonicalize`.

Registered: ``swe-agent``, ``swe-smith`` and ``swe-smith-native`` (share the
swe-agent atom map), ``mini-swe-agent``, ``openhands``, ``react-text``,
``agentless``, ``dars``, ``moatless``, ``prompt-completion`` (OpenAI-style
prompt/completion SFT tool-use corpora), ``gumtree`` (fine-grained
node-typed AST atoms), ``cursor-companion`` (human+AI sessions from Cursor
IDE via the cursor-telemetry companion service), ``claude-code``
(Claude Code session transcripts from ``~/.claude/projects/``), the other
SWE-chat terminal-agent formats ``opencode``, ``codex``, and ``gemini-cli``
(each mapping its own tool/event vocabulary into the same atom alphabet), and
``spine`` (precomputed space-joined atom strings, e.g. midah/procgrep-spines).

The interactive adapters (``cursor-companion``, ``claude-code``, ``opencode``,
``codex``, ``gemini-cli``) extend the atom alphabet with ``prompt_ai`` to mark
human-to-AI handoffs -- an event type absent from autonomous-agent traces. All identifying fields (workspace
paths, session ids, file names, prompt text) are hashed or dropped before
they reach the atom sequence; only action structure crosses the boundary.

Design decisions (benefit / price):

1. Each adapter registers itself on import, and this package imports them all.
   Benefit: one import wires up the whole registry; the atom-only adapters
   (openhands, react-text) reuse mini-swe-agent's bash classifier rather than
   forking it. Price: a new adapter is not discovered until it is added to the
   imports below (explicit registration, no plugin autodiscovery by design).
"""

from __future__ import annotations

from procgrep.ingest.adapters import (
    agentless,
    claude_code,
    codex,
    cursor_companion,
    dars,
    function_xml,
    gemini_cli,
    gumtree,
    mini_swe_agent,
    moatless,
    opencode,
    openhands,
    prompt_completion,
    react_text,
    spine,
    swe_agent,
    swe_agent_traj,
    swe_smith,
)

__all__ = [
    "agentless",
    "claude_code",
    "codex",
    "cursor_companion",
    "dars",
    "function_xml",
    "gemini_cli",
    "gumtree",
    "mini_swe_agent",
    "moatless",
    "opencode",
    "openhands",
    "prompt_completion",
    "react_text",
    "spine",
    "swe_agent",
    "swe_agent_traj",
    "swe_smith",
]
