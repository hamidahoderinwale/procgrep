"""mini-swe-agent trajectory adapter.

mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent) writes
each run to a single `<task_id>.traj.json` file with format version
``mini-swe-agent-1.1``. The agent has one tool — bash — so every
action is a bash command. The adapter classifies each command's
first significant token (after stripping ``cd`` prefixes and shell
chaining) and maps it to a procgrep canonical atom.

Trace shape recap:

    {
      "info": {...},
      "messages": [
        {"role": "system",    "content": "...", "extra": {...}},
        {"role": "user",      "content": "...", "extra": {...}},
        {"role": "assistant", "content": "...", "extra": {
            "actions": [{"command": "cat src/foo.py", ...}, ...]
        }},
        {"role": "tool",      "content": "<bash output>", "extra": {...}},
        ...
      ],
      "trajectory_format": "mini-swe-agent-1.1"
    }

Each assistant turn carries zero or more parsed bash actions under
``extra.actions``. We treat each action as one atom.
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
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
    Atom,
    AtomSequence,
)

# First-token classification. The agent issues bash; we look at the
# command head after stripping leading `cd ... &&` chaining.
_BASH_COMMAND_ATOM: dict[str, Atom] = {
    # Read-file family
    "cat": ATOM_READ_FILE,
    "head": ATOM_READ_FILE,
    "tail": ATOM_READ_FILE,
    "less": ATOM_READ_FILE,
    "more": ATOM_READ_FILE,
    "view": ATOM_READ_FILE,
    "open": ATOM_READ_FILE,
    "bat": ATOM_READ_FILE,
    # Search family
    "grep": ATOM_SEARCH_REPO,
    "rg": ATOM_SEARCH_REPO,
    "ripgrep": ATOM_SEARCH_REPO,
    "ag": ATOM_SEARCH_REPO,
    "ack": ATOM_SEARCH_REPO,
    "find": ATOM_SEARCH_REPO,
    "locate": ATOM_SEARCH_REPO,
    "fd": ATOM_SEARCH_REPO,
    # Edit family (in-place file edits + writes via redirection)
    "sed": ATOM_EDIT,
    "awk": ATOM_EDIT,
    "patch": ATOM_EDIT,
    "ed": ATOM_EDIT,
    # Test family
    "pytest": ATOM_RUN_TEST,
    "unittest": ATOM_RUN_TEST,
    "tox": ATOM_RUN_TEST,
    "nose2": ATOM_RUN_TEST,
    "django-admin": ATOM_RUN_TEST,  # commonly used to run Django tests
    # Create-file family
    "mkdir": ATOM_CREATE_FILE,
    "touch": ATOM_CREATE_FILE,
    # Delete-file family
    "rm": ATOM_DELETE_FILE,
    "rmdir": ATOM_DELETE_FILE,
    # Submit family
    "submit": ATOM_SUBMIT,
    "MINI_SWE_AGENT_FINAL_OUTPUT": ATOM_SUBMIT,
}

# Commands that may carry richer semantics depending on flags or
# arguments. We delegate to small classifiers below.
_PYTHON_TEST_RUNNERS = (
    re.compile(r"\bpython\s+-m\s+(pytest|unittest|nose2|tox)\b"),
    re.compile(r"\bpython\s+-m\s+django\s+test\b"),
)
_PYTHON_EDIT_PATTERNS = (
    # Heredocs writing files: python <<EOF ... EOF
    re.compile(r"<<\s*['\"]?(\w+)['\"]?"),
    # echo "..." > file or cat <<EOF > file
    re.compile(r">\s*[\w/.\-]+"),
)
_GIT_SUBMIT_PATTERNS = (re.compile(r"\bgit\s+(diff|apply|format-patch)\b"),)


def _strip_chain_prefix(cmd: str) -> str:
    """Drop leading ``cd X && `` / ``cd X ;`` so we classify the user-visible command."""
    out = cmd.strip()
    while True:
        match = re.match(r"^cd\s+\S+\s*(&&|;)\s*(.+)$", out)
        if not match:
            return out
        out = match.group(2).strip()


def _classify_command(command: str) -> Atom:
    """Map one bash command string to a canonical atom."""
    if not command or not command.strip():
        return ATOM_OTHER
    stripped = _strip_chain_prefix(command)

    # Python-runner specials.
    for pattern in _PYTHON_TEST_RUNNERS:
        if pattern.search(stripped):
            return ATOM_RUN_TEST
    # echo "..." > file, cat <<EOF > file, heredoc edits.
    for pattern in _PYTHON_EDIT_PATTERNS:
        if pattern.search(stripped):
            return ATOM_EDIT
    # git diff / git apply -> patch submission flow
    for pattern in _GIT_SUBMIT_PATTERNS:
        if pattern.search(stripped):
            return ATOM_SUBMIT

    # Standard first-token lookup.
    first = stripped.split()[0] if stripped.split() else ""
    return _BASH_COMMAND_ATOM.get(first, ATOM_OTHER)


def _iter_actions(messages: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Flatten assistant-message actions into a single ordered list."""
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
    """Did this assistant message include non-empty 'reasoning' content alongside the action?"""
    content = msg.get("content")
    return isinstance(content, str) and content.strip() != ""


def mini_swe_agent_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert a mini-swe-agent .traj.json record into procgrep atoms.

    Args:
        record: Parsed contents of a mini-swe-agent trajectory file.
            Expected keys at top level: ``messages`` (list), optionally
            ``info``, ``trajectory_format``. The adapter is permissive
            and degrades gracefully on missing or malformed fields.

    Returns:
        Ordered atom sequence. Each assistant-emitted bash action
        becomes one atom (classified via :func:`_classify_command`).
        Assistant messages whose ``content`` contains non-empty
        reasoning text emit an ``ATOM_THINK`` atom before the first
        action in that turn.

        If ``info.exit_status`` indicates an error (anything other
        than ``"Submitted"`` and other normal-completion strings), the
        sequence is suffixed with ``ATOM_ERROR``.
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
            # Thought without a corresponding tool call still counts.
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

    # Reflect non-normal exit in the trace.
    info = record.get("info")
    if isinstance(info, Mapping):
        exit_status = str(info.get("exit_status", "") or "").lower()
        if exit_status and exit_status not in {"submitted", "completed", "ok", "success"}:
            atoms.append(ATOM_ERROR)

    return atoms


def _register() -> None:
    """Register the mini-swe-agent adapter under the name ``"mini-swe-agent"``."""
    register_adapter("mini-swe-agent", mini_swe_agent_adapter, overwrite=True)


_register()


__all__ = ["mini_swe_agent_adapter"]
