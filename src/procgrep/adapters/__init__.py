"""Built-in trace adapters.

Importing this package side-effect-registers every built-in adapter
under its canonical name in `procgrep.canonicalize`. Each adapter
module is small and self-contained: it builds a `TraceAdapter` and
registers it at import time. Custom-scaffold adapters can follow the
same pattern and live alongside, or stay external and register via
`procgrep.register_adapter`.

Adapters registered here:

* ``swe-agent``   -- SWE-agent action traces.
* ``agentless``   -- Agentless phase traces.
* ``dars``        -- DARS tool traces.
* ``moatless``    -- Moatless action traces.
* ``gumtree``     -- Gumtree AST edit-script traces (fine-grained
  node-typed atoms).
"""

from __future__ import annotations

from procgrep.adapters import agentless, dars, gumtree, moatless, swe_agent

__all__ = ["agentless", "dars", "gumtree", "moatless", "swe_agent"]
