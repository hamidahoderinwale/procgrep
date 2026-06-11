"""Built-in trace adapters.

Importing this package registers each built-in adapter under its
canonical name in `procgrep.canonicalize`.

Registered: ``swe-agent``, ``swe-smith`` (shares the swe-agent atom
map), ``mini-swe-agent``, ``openhands``, ``agentless``, ``dars``,
``moatless``, and ``gumtree`` (fine-grained node-typed AST atoms).
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
