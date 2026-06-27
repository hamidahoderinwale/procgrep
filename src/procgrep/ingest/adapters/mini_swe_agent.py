"""mini-swe-agent trajectory adapter.

mini-swe-agent writes each run to one ``<task_id>.traj.json``
(format ``mini-swe-agent-1.1``). Bash is the only tool, so every
assistant action is a command. Each command's first significant
token (after stripping ``cd ... &&`` prefixes) maps to a canonical
atom. See https://github.com/SWE-agent/mini-swe-agent.

Trace shape::

    {
      "info": {...},
      "messages": [
        {"role": "assistant", "content": "...", "extra": {
            "actions": [{"command": "cat src/foo.py", ...}]
        }},
        ...
      ]
    }
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import register_adapter
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_DELETE_FILE,
    ATOM_EDIT,
    ATOM_ERROR,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
    Atom,
    AtomSequence,
)

_BASH_COMMAND_ATOM: dict[str, Atom] = {
    "cat": ATOM_READ_FILE,
    "head": ATOM_READ_FILE,
    "tail": ATOM_READ_FILE,
    "less": ATOM_READ_FILE,
    "more": ATOM_READ_FILE,
    "view": ATOM_READ_FILE,
    "open": ATOM_READ_FILE,
    "bat": ATOM_READ_FILE,
    "grep": ATOM_SEARCH_REPO,
    "rg": ATOM_SEARCH_REPO,
    "ripgrep": ATOM_SEARCH_REPO,
    "ag": ATOM_SEARCH_REPO,
    "ack": ATOM_SEARCH_REPO,
    "find": ATOM_SEARCH_REPO,
    "locate": ATOM_SEARCH_REPO,
    "fd": ATOM_SEARCH_REPO,
    "sed": ATOM_EDIT,
    "awk": ATOM_EDIT,
    "patch": ATOM_EDIT,
    "ed": ATOM_EDIT,
    "pytest": ATOM_RUN_TEST,
    "unittest": ATOM_RUN_TEST,
    "tox": ATOM_RUN_TEST,
    "nose2": ATOM_RUN_TEST,
    "django-admin": ATOM_RUN_TEST,
    # Running a script or inline snippet (test runners are matched earlier, so
    # only non-test python reaches here): the agent's repro / debug loop.
    "python": ATOM_RUN_CODE,
    "python3": ATOM_RUN_CODE,
    "mkdir": ATOM_CREATE_FILE,
    "touch": ATOM_CREATE_FILE,
    "rm": ATOM_DELETE_FILE,
    "rmdir": ATOM_DELETE_FILE,
    "submit": ATOM_SUBMIT,
    "MINI_SWE_AGENT_FINAL_OUTPUT": ATOM_SUBMIT,
}

_PYTHON_TEST_RUNNERS = (
    re.compile(r"\bpython\s+-m\s+(pytest|unittest|nose2|tox)\b"),
    re.compile(r"\bpython\s+-m\s+django\s+test\b"),
)
_PYTHON_EDIT_PATTERNS = (
    # Heredocs (python <<EOF ... EOF) and redirected writes
    # (echo "..." > file, cat <<EOF > file).
    re.compile(r"<<\s*['\"]?(\w+)['\"]?"),
    re.compile(r">\s*[\w/.\-]+"),
)
_GIT_SUBMIT_PATTERNS = (re.compile(r"\bgit\s+(diff|apply|format-patch)\b"),)


def _strip_chain_prefix(cmd: str) -> str:
    """Strip leading ``cd X && `` / ``cd X ;`` chains.

    DOTALL so the trailing command survives embedded newlines (agents
    write ``cd repo && python -c "<multiline snippet>"``); without it the
    whole command would keep its ``cd`` prefix and misclassify as other.
    """
    out = cmd.strip()
    while True:
        match = re.match(r"^cd\s+\S+\s*(&&|;)\s*(.+)$", out, re.DOTALL)
        if not match:
            return out
        out = match.group(2).strip()


def _classify_command(command: str) -> Atom:
    """Map one bash command string to a canonical atom."""
    if not command or not command.strip():
        return ATOM_OTHER
    stripped = _strip_chain_prefix(command)

    for pattern in _PYTHON_TEST_RUNNERS:
        if pattern.search(stripped):
            return ATOM_RUN_TEST
    for pattern in _PYTHON_EDIT_PATTERNS:
        if pattern.search(stripped):
            return ATOM_EDIT
    for pattern in _GIT_SUBMIT_PATTERNS:
        if pattern.search(stripped):
            return ATOM_SUBMIT

    first = stripped.split()[0] if stripped.split() else ""
    return _BASH_COMMAND_ATOM.get(first, ATOM_OTHER)


def _iter_actions(messages: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Flatten assistant-message actions into one ordered list."""
    out: list[Mapping[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        if msg.get("role") != "assistant":
            continue
        extra = msg.get("extra")
        if not isinstance(extra, Mapping):
            continue
        actions = extra.get("actions") or []
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, Mapping):
                out.append(action)
    return out


def _action_carried_thought(msg: Mapping[str, Any]) -> bool:
    """True iff the assistant message has non-empty reasoning content."""
    content = msg.get("content")
    return isinstance(content, str) and content.strip() != ""


def mini_swe_agent_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert a mini-swe-agent ``.traj.json`` record into atoms.

    Each assistant bash action becomes one atom via
    :func:`_classify_command`. Assistant messages with non-empty
    reasoning emit ``ATOM_THINK`` before that turn's first action.
    Non-normal ``info.exit_status`` values (anything outside
    ``submitted``, ``completed``, ``ok``, ``success``) append
    ``ATOM_ERROR``. The adapter degrades gracefully on missing or
    malformed fields.
    """
    messages = record.get("messages") or []
    if not isinstance(messages, list):
        return []

    atoms: AtomSequence = []
    for msg in messages:
        if not isinstance(msg, Mapping) or msg.get("role") != "assistant":
            continue
        extra = msg.get("extra")
        actions = (extra or {}).get("actions") if isinstance(extra, Mapping) else None
        if not isinstance(actions, list) or not actions:
            # Thought without a tool call still emits THINK.
            if _action_carried_thought(msg):
                atoms.append(ATOM_THINK)
            continue
        if _action_carried_thought(msg):
            atoms.append(ATOM_THINK)
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            command = str(action.get("command", ""))
            atoms.append(_classify_command(command))

    info = record.get("info")
    if isinstance(info, Mapping):
        exit_status = str(info.get("exit_status", "") or "").lower()
        if exit_status and exit_status not in {"submitted", "completed", "ok", "success"}:
            atoms.append(ATOM_ERROR)

    return atoms


def _register() -> None:
    """Register under the name ``"mini-swe-agent"``."""
    register_adapter("mini-swe-agent", mini_swe_agent_adapter, overwrite=True)


_register()


__all__ = ["mini_swe_agent_adapter"]
