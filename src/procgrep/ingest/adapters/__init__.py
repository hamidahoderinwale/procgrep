"""Built-in trace adapters.

Importing this package registers each built-in adapter under its
canonical name in `procgrep.canonicalize`.

Registered: ``swe-agent``, ``swe-smith`` and ``swe-smith-native`` (share the
swe-agent atom map), ``mini-swe-agent``, ``openhands``, ``react-text``,
``agentless``, ``dars``, ``moatless``, and ``gumtree`` (fine-grained
node-typed AST atoms).

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
    dars,
    gumtree,
    mini_swe_agent,
    moatless,
    openhands,
    react_text,
    swe_agent,
    swe_smith,
)

__all__ = [
    "agentless",
    "dars",
    "gumtree",
    "mini_swe_agent",
    "moatless",
    "openhands",
    "react_text",
    "swe_agent",
    "swe_smith",
]
