"""SWE-smith chat-format trajectory adapter.

The SWE-smith-trajectories dataset stores a JSON-string ``messages`` list
per row in chat format. Each assistant turn carries the raw shell command
the agent executed in the ``action`` field (not a discrete tool name) and
a ``thought`` field carrying its reasoning.

Two adapters are exposed:

- :func:`swe_smith_adapter` returns the canonical atom alphabet
  (``ATOM_EDIT``, ``ATOM_READ_FILE`` etc.), suitable for cross-scaffold
  comparison.
- :func:`swe_smith_native_adapter` returns a richer native alphabet
  (``str_replace_editor:view``, ``python:script``, ``find`` etc.), suitable
  for within-scaffold depth.

Both share :func:`classify_swe_smith_action`, which inspects the bash
command string and returns ``(canonical, native)`` in one pass.
:func:`swe_smith_canonical_projection` maps a native atom to its canonical,
matching the projection callable shape that ``lineage_diff`` expects.

See https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from procgrep.adapters.swe_agent import ATOM_MAP as _BARE_ACTION_MAP
from procgrep.canonicalize import register_adapter
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_DELETE_FILE,
    ATOM_EDIT,
    ATOM_ERROR,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
    Atom,
    AtomSequence,
)


def _parse_messages(raw: Any) -> list[Any]:
    """Return the messages list; parses a JSON string if needed."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _strip_leading_cd(cmd: str) -> str:
    """Strip a leading ``cd <path> &&`` wrapper. Returns the remaining command."""
    s = cmd.strip()
    if not s.startswith("cd "):
        return s
    sep = s.find("&&")
    if sep == -1:
        return s
    return s[sep + 2 :].strip()


def _first_in_chain(cmd: str) -> str:
    """First command in a ``&&`` or pipe chain."""
    s = cmd.strip()
    for sep in (" && ", " | ", "|"):
        idx = s.find(sep)
        if idx != -1:
            return s[:idx].strip()
    return s


def _python_invocation(cmd: str) -> tuple[Atom, str]:
    """Classify a ``python ...`` command into (canonical, native)."""
    rest = cmd[len("python") :].lstrip()
    if rest.startswith(("-m pytest", "-m unittest")):
        return ATOM_RUN_TEST, "python:pytest"
    if rest.startswith(("pytest", "unittest")):
        return ATOM_RUN_TEST, "python:pytest"
    return ATOM_RUN_TEST, "python:script"


def _classify_simple(cmd: str) -> tuple[Atom, str]:
    """Classify a single (non-compound) bash command. Empty input is caller's job."""
    s = cmd.strip()

    # Bare classic SWE-agent action names like ``"edit"`` or ``"open"``
    # (no arguments). Used by traces in the classic SWE-agent format that
    # happen to flow through this adapter.
    if s in _BARE_ACTION_MAP:
        return _BARE_ACTION_MAP[s], s

    # Submit is the one true canonical action name in the SWE-smith dataset.
    if s == "submit":
        return ATOM_SUBMIT, "submit"

    # str_replace_editor is the dominant file-manipulation tool. Subcommand matters.
    if s.startswith("str_replace_editor "):
        sub = s[len("str_replace_editor ") :].lstrip().split(None, 1)[0]
        if sub == "view":
            return ATOM_READ_FILE, "str_replace_editor:view"
        if sub == "create":
            return ATOM_CREATE_FILE, "str_replace_editor:create"
        if sub == "str_replace":
            return ATOM_EDIT, "str_replace_editor:str_replace"
        if sub == "insert":
            return ATOM_EDIT, "str_replace_editor:insert"
        if sub == "undo_edit":
            return ATOM_EDIT, "str_replace_editor:undo"
        return ATOM_OTHER, f"str_replace_editor:{sub}"

    # python: differentiate test runners from script invocations.
    if s.startswith(("python ", "python3 ")) or s == "python":
        normalized = "python " + s[len("python3 ") :] if s.startswith("python3 ") else s
        return _python_invocation(normalized)
    if s.startswith("pytest"):
        return ATOM_RUN_TEST, "python:pytest"

    # Search tools.
    if s.startswith("find ") or s == "find":
        return ATOM_SEARCH_REPO, "find"
    if s.startswith(("grep ", "rg ", "ag ")) or s in {"grep", "rg", "ag"}:
        return ATOM_SEARCH_REPO, "grep"

    # Read-only inspection.
    if s.startswith("ls ") or s == "ls":
        return ATOM_READ_FILE, "ls"
    if s.startswith(("cat ", "head ", "tail ", "less ", "more ")):
        return ATOM_READ_FILE, "cat"

    # Delete.
    if s.startswith("rm ") or s == "rm":
        return ATOM_DELETE_FILE, "rm"

    # Create / scaffold.
    if s.startswith(("mkdir ", "touch ")):
        return ATOM_CREATE_FILE, "mkdir_touch"

    # File ops we don't claim are either edit or read.
    if s.startswith(("mv ", "cp ")):
        return ATOM_OTHER, "mv_cp"
    if s.startswith(("chmod ", "chown ")):
        return ATOM_OTHER, "permissions"

    # `echo "..." > file` is creation-by-redirect; the simpler form catches most cases.
    if s.startswith("echo ") and ">" in s:
        return ATOM_CREATE_FILE, "echo_redirect"

    # Fallback: first token, lowercased, as a bucket tag.
    first = s.split(None, 1)[0].lower() if s else "empty"
    return ATOM_OTHER, f"other:{first}"


def classify_swe_smith_action(action: str) -> tuple[Atom, str]:
    """Classify a raw SWE-smith ``action`` string into (canonical, native).

    Returns ``(ATOM_ERROR, "empty_or_error")`` for empty / whitespace
    actions; these are scaffold failures (context limits, API errors).
    Strips a leading ``cd <path> &&`` wrapper, then dispatches by the
    first command in any ``&&`` or pipe chain.
    """
    if not isinstance(action, str) or not action.strip():
        return ATOM_ERROR, "empty_or_error"

    inner = _strip_leading_cd(action)
    head = _first_in_chain(inner)
    canonical, native = _classify_simple(head)

    # Tag compound shape on the native side so within-scaffold analysis
    # can tell `find` from `find | xargs grep` even though both project
    # to ATOM_SEARCH_REPO at canonical.
    if " && " in inner:
        native = f"chain:{native}"
    elif " | " in inner or "|" in inner.replace("||", ""):
        native = f"pipe:{native}"

    return canonical, native


# Static native -> canonical table for use as a `canonical_projection`
# callable with `lineage_diff(alphabet=["canonical","native"], ...)`.
_NATIVE_TO_CANONICAL: dict[str, Atom] = {
    "submit": ATOM_SUBMIT,
    "empty_or_error": ATOM_ERROR,
    "str_replace_editor:view": ATOM_READ_FILE,
    "str_replace_editor:create": ATOM_CREATE_FILE,
    "str_replace_editor:str_replace": ATOM_EDIT,
    "str_replace_editor:insert": ATOM_EDIT,
    "str_replace_editor:undo": ATOM_EDIT,
    "python:pytest": ATOM_RUN_TEST,
    "python:script": ATOM_RUN_TEST,
    "find": ATOM_SEARCH_REPO,
    "grep": ATOM_SEARCH_REPO,
    "ls": ATOM_READ_FILE,
    "cat": ATOM_READ_FILE,
    "rm": ATOM_DELETE_FILE,
    "mkdir_touch": ATOM_CREATE_FILE,
    "mv_cp": ATOM_OTHER,
    "permissions": ATOM_OTHER,
    "echo_redirect": ATOM_CREATE_FILE,
}


def swe_smith_canonical_projection(native: Atom) -> Atom:
    """Project a SWE-smith native atom to its canonical atom.

    Strips ``chain:`` / ``pipe:`` wrappers, drops the ``str_replace_editor:``
    or ``python:`` namespace if needed, and looks up the static table.
    Unknown natives (``other:*`` and ``str_replace_editor:<unseen>``) fall
    through to ``ATOM_OTHER``.
    """
    n = native
    if n.startswith("chain:"):
        n = n[len("chain:") :]
    elif n.startswith("pipe:"):
        n = n[len("pipe:") :]
    return _NATIVE_TO_CANONICAL.get(n, ATOM_OTHER)


def _tool_calls_action_name(msg: Mapping[str, Any]) -> str | None:
    """Pull the first tool-call function name from a chat-format turn."""
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0]
        if isinstance(first, Mapping):
            fn = first.get("function")
            if isinstance(fn, Mapping):
                name = fn.get("name")
                if isinstance(name, str) and name:
                    return name
    return None


def _atoms_from_messages(messages: list[Any], *, layer: str) -> AtomSequence:
    """Walk a messages list, emit one canonical or native atom per assistant turn.

    Prepends ``ATOM_THINK`` when the turn carries non-empty ``thought`` text.
    When the ``action`` field is missing, falls back to
    ``tool_calls[0].function.name`` and emits nothing if neither is present.
    Empty ``action`` strings (scaffold errors like context-limit hits) become
    ``ATOM_ERROR``.
    """
    atoms: AtomSequence = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        if msg.get("role") != "assistant":
            continue

        thought = msg.get("thought")
        if isinstance(thought, str) and thought.strip():
            atoms.append(ATOM_THINK)

        action = msg.get("action")
        if isinstance(action, str) and action.strip():
            canonical, native = classify_swe_smith_action(action)
            atoms.append(canonical if layer == "canonical" else native)
        elif isinstance(action, str):  # present but empty -> scaffold error
            atoms.append(ATOM_ERROR if layer == "canonical" else "empty_or_error")
        else:  # action key absent -> try tool_calls
            name = _tool_calls_action_name(msg)
            if name is not None:
                canonical, native = classify_swe_smith_action(name)
                atoms.append(canonical if layer == "canonical" else native)

    return atoms


def swe_smith_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert one SWE-smith-trajectories row into canonical atoms.

    Each assistant turn yields one canonical atom from
    :func:`classify_swe_smith_action`. Non-empty ``thought`` text prepends
    ``ATOM_THINK``. Scaffold errors (empty action) become ``ATOM_ERROR``.
    """
    messages = _parse_messages(record.get("messages"))
    return _atoms_from_messages(messages, layer="canonical")


def swe_smith_native_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert one SWE-smith-trajectories row into native atoms.

    Same shape as :func:`swe_smith_adapter` but emits the richer native
    alphabet (``str_replace_editor:view`` etc.). Use together with
    :func:`swe_smith_canonical_projection` for hierarchical
    ``lineage_diff(alphabet=["canonical","native"], ...)`` analyses.
    """
    messages = _parse_messages(record.get("messages"))
    return _atoms_from_messages(messages, layer="native")


register_adapter("swe-smith", swe_smith_adapter, overwrite=True)
register_adapter("swe-smith-native", swe_smith_native_adapter, overwrite=True)


__all__ = [
    "classify_swe_smith_action",
    "swe_smith_adapter",
    "swe_smith_canonical_projection",
    "swe_smith_native_adapter",
]
