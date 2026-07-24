"""Claude Code transcript adapter.

Converts Claude Code session transcripts into procgrep atoms. Claude Code is a
separate tool woven in only as an ingest adapter -- an exemplar trace source,
not part of procgrep's core. Unlike the cursor companion, transcripts are plain
local files at ``~/.claude/projects/<project>/<session>.jsonl``, so ingesting
them needs no running service.

A transcript is a JSONL stream where each line is one event and a single
assistant line may carry several ``tool_use`` blocks. ``_flatten`` first
explodes the stream into atomic, prompt-anchored actions -- one event per human
prompt, per assistant tool call, and per file-history snapshot -- which the
shared feature-based `make_event_adapter` then maps. Reusing that builder is the
point: a structurally different source mapped by the same machinery.

Expected record shape (one record == one session)::

    {"trace_id": "<session>", "agent": "<workspace>", "events": [<raw lines>]}

Use `load_claude_transcript(path)` to build one from a ``.jsonl`` file.

Atom mapping (by flattened action kind):

    human prompt (a user turn, not a tool result)      -> prompt_ai
    Edit / Write / NotebookEdit / file-history snapshot -> edit
    Read / NotebookRead                                 -> read_file
    Grep / Glob                                         -> search_repo
    WebSearch / WebFetch                                -> search_repo (search-class)
    Bash, test/build command                            -> run_test
    Bash, git / gh / hg / svn                            -> version_control
    Bash, pip / uv / npm / cargo ... install            -> package
    Bash, ruff / mypy / eslint / prettier ...           -> lint
    Bash, grep / rg / find / fd                          -> search_repo
    Bash, python / node / make / ./script               -> run_code
    Bash (other), Agent, Task*, MCP tools, ...          -> other

Design decisions:

- Only the action *structure* is read -- tool names, a Bash command's leading
  verb, and event types. Message text is never inspected, so a transcript can be
  fingerprinted without exposing its content.

- ``prompt_ai`` is the human->AI turn, matching the cursor-companion atom so the
  two sources share one alphabet. Claude Code is agent-dense (one prompt, then a
  long tool run) where Cursor is interleaved; the shared atoms make that
  contrast measurable rather than hiding it.

- Bash is sub-classified by its leading verb into a small set of safe category
  atoms (test, version_control, package, lint, search, run_code); the command
  string is classified and then *discarded* in `_flatten`, so only the category
  -- never the command -- reaches the fingerprint. Anything unrecognized stays
  ``other`` rather than inflating a specific signal.
"""

from __future__ import annotations

import collections
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from procgrep.canonicalize import EventRule, field_in, make_event_adapter, register_adapter
from procgrep.types import (
    ATOM_EDIT,
    ATOM_LINT,
    ATOM_OTHER,
    ATOM_PACKAGE,
    ATOM_PROMPT_AI,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_THINK,
    ATOM_VERSION_CONTROL,
    Atom,
    AtomSequence,
)

_EDIT_TOOLS = {"edit", "write", "notebookedit", "multiedit"}
_READ_TOOLS = {"read", "notebookread"}
_SEARCH_TOOLS = {"grep", "glob"}
_WEB_TOOLS = {"websearch", "webfetch"}
# Plan-management tools map to think, matching the other interface adapters
# (opencode todowrite, codex update_plan, gemini write_todos / exit_plan_mode) so
# planning is one shared atom across interfaces rather than CC-only ``other``.
_THINK_TOOLS = {"todowrite", "exitplanmode"}
_TEST_CMD = re.compile(
    r"(pytest|(^|\s|/)tests?\b|unittest|jest|mocha|vitest|tox|"
    r"go test|cargo test|npm (run )?test|yarn test|make test|gradle test)",
    re.IGNORECASE,
)
_VCS_CMD = re.compile(r"(^|\s|;|&|\|)(git|gh|hg|svn)\b", re.IGNORECASE)
_PKG_CMD = re.compile(
    r"\b(pip3?|uv|poetry|pipenv|conda|npm|yarn|pnpm|bundle|gem|cargo|go|apt|apt-get|brew)"
    r"\b[^|;&]*\b(install|add|sync|update|upgrade)\b",
    re.IGNORECASE,
)
_LINT_CMD = re.compile(
    r"\b(ruff|black|isort|flake8|pylint|mypy|pyright|eslint|prettier|tsc|golangci-lint|gofmt|clippy|rubocop)\b",
    re.IGNORECASE,
)
_SEARCH_CMD = re.compile(r"(^|\s|;|&|\|)(grep|rg|ag|ack|find|fd)\b", re.IGNORECASE)
_RUN_CMD = re.compile(
    r"(^|\s|;|&|\|)(python3?|node|deno|bun|ruby|go run|cargo run|make|\./)", re.IGNORECASE
)

# A Bash command -> a privacy-safe sub-kind, ordered most-specific first. The
# command is classified by its leading verb/keywords and then DISCARDED in
# `_flatten`; only the category reaches the event, so the command text never
# leaves -- the reduction to a category is itself the obfuscation. A command
# matching nothing stays the generic "bash" (which falls through to `other`),
# so unknown commands never inflate a specific signal.
_BASH_SUBKIND: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bash_test", _TEST_CMD),
    ("bash_vcs", _VCS_CMD),
    ("bash_package", _PKG_CMD),
    ("bash_lint", _LINT_CMD),
    ("bash_search", _SEARCH_CMD),
    ("bash_run", _RUN_CMD),
)


def _classify_bash(command: str) -> str:
    """Classify a Bash command into a sub-kind by its leading verb only."""
    for subkind, pattern in _BASH_SUBKIND:
        if pattern.search(command):
            return subkind
    return "bash"


# A file read or directory listing done through the shell, by leading verb. This
# is split out from `_classify_bash` because Claude Code rarely shells out to
# read a file (it has Read/Glob tools), but the other terminal agents routinely
# do -- Codex reads files with ``sed -n``/``cat``/``nl``, lists with ``ls``. The
# terminal-agent adapters layer this on so that idiom is recovered as read_file /
# search_repo instead of collapsing to ``other``; claude-code's own mapping is
# left unchanged. ``Select-String`` is the PowerShell grep, ``Get-ChildItem`` its
# directory listing, ``Get-Content`` its cat.
_READ_CMD = re.compile(
    r"(^|\s|;|&|\|)(cat|sed|head|tail|nl|less|more|bat|Get-Content)\b", re.IGNORECASE
)
_LIST_CMD = re.compile(r"(^|\s|;|&|\|)(ls|tree|Get-ChildItem|Select-String)\b", re.IGNORECASE)


def _classify_terminal_command(command: str) -> str:
    """Classify a terminal command, extending `_classify_bash` with read / list.

    Tries the shared `_classify_bash` verbs first (test, vcs, package, lint,
    search, run); then the read and directory-listing idioms the non-Claude-Code
    terminal agents lean on. Like `_classify_bash`, the command is classified by
    its leading verb only and then discarded -- only the category crosses the
    boundary.
    """
    subkind = _classify_bash(command)
    if subkind != "bash":
        return subkind
    if _READ_CMD.search(command):
        return "bash_read"
    if _LIST_CMD.search(command):
        return "bash_search"
    return "bash"


CLAUDE_RULES: tuple[EventRule, ...] = (
    EventRule(field_in("kind", {"user_prompt"}), (ATOM_PROMPT_AI,)),
    EventRule(field_in("kind", {*_EDIT_TOOLS, "file_snapshot"}), (ATOM_EDIT,)),
    EventRule(field_in("kind", _READ_TOOLS), (ATOM_READ_FILE,)),
    EventRule(field_in("kind", _SEARCH_TOOLS), (ATOM_SEARCH_REPO,)),
    EventRule(field_in("kind", _WEB_TOOLS), (ATOM_SEARCH_REPO,)),
    EventRule(field_in("kind", _THINK_TOOLS), (ATOM_THINK,)),
    EventRule(field_in("kind", {"bash_test"}), (ATOM_RUN_TEST,)),
    EventRule(field_in("kind", {"bash_search"}), (ATOM_SEARCH_REPO,)),
    EventRule(field_in("kind", {"bash_vcs"}), (ATOM_VERSION_CONTROL,)),
    EventRule(field_in("kind", {"bash_package"}), (ATOM_PACKAGE,)),
    EventRule(field_in("kind", {"bash_lint"}), (ATOM_LINT,)),
    EventRule(field_in("kind", {"bash_run"}), (ATOM_RUN_CODE,)),
)

_MAP = make_event_adapter(rules=CLAUDE_RULES)


# Harness-injected user turns that are not the human typing: task notifications,
# system reminders, slash-command echoes, captured command output. They arrive as
# role=user text, so without this guard they're miscounted as human prompts and
# become false intent-cliff boundaries (~15% of turns in local transcripts).
_INJECTED_PREFIXES = (
    "<task-notification>",
    "<system-reminder>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<bash-stdout>",
    "<bash-stderr>",
)


def _is_human_prompt(content: Any) -> bool:
    """A user turn typed by the human, not a tool-result or harness-injected event."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        has_text = any(isinstance(b, Mapping) and b.get("type") == "text" for b in content)
        has_tool_result = any(
            isinstance(b, Mapping) and b.get("type") == "tool_result" for b in content
        )
        if not (has_text and not has_tool_result):
            return False
        text = _prompt_text(content)
    else:
        return False
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith(_INJECTED_PREFIXES)


def _flatten(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Explode raw transcript lines into atomic, prompt-anchored action events.

    One human prompt, one assistant ``tool_use`` block, or one file-history
    snapshot each become a single ``{"kind": ...}`` event. Tool-result user
    events (the agent's own outputs) are dropped -- they are not human turns.
    """
    events: list[dict[str, Any]] = []
    lines = record.get("events")
    if not isinstance(lines, list):
        return events
    for line in lines:
        if not isinstance(line, Mapping):
            continue
        kind = line.get("type")
        message = line.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if kind == "user":
            if _is_human_prompt(content):
                events.append({"kind": "user_prompt"})
        elif kind == "assistant" and isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "tool_use":
                    name = str(block.get("name") or "")
                    if name.lower() == "bash":
                        tool_input = block.get("input")
                        command = (
                            tool_input.get("command", "") if isinstance(tool_input, Mapping) else ""
                        )
                        events.append({"kind": _classify_bash(str(command))})
                    else:
                        events.append({"kind": name})
        elif kind == "file-history-snapshot":
            events.append({"kind": "file_snapshot"})
    return events


def claude_code_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert one Claude Code session record into an atom sequence."""
    return _MAP({"events": _flatten(record)})


def _words_chars(text: str) -> tuple[int, int]:
    return len(text.split()), len(text)


def summarize_transcript(record: Mapping[str, Any]) -> dict[str, Any]:
    """Cheaply-gleanable per-session metadata, counts only -- never the text.

    Reads message text solely to *count* it (words/chars of human prompts and of
    assistant reasoning), then discards it, so a session can be summarized
    without retaining its content. Also surfaces turn counts, per-tool call
    counts, file-snapshot count, models used, and verbosity-per-turn -- the
    raw material for verbosity/groundedness and autonomy reads.
    """
    lines = record.get("events")
    if not isinstance(lines, list):
        return {}
    out: dict[str, Any] = {
        "human_turns": 0,
        "assistant_turns": 0,
        "prompt_words": 0,
        "prompt_chars": 0,
        "reasoning_words": 0,
        "reasoning_chars": 0,
        "tool_calls": 0,
        "file_snapshots": 0,
    }
    tools: collections.Counter[str] = collections.Counter()
    models: set[str] = set()
    for line in lines:
        if not isinstance(line, Mapping):
            continue
        kind = line.get("type")
        message = line.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        model = message.get("model") if isinstance(message, Mapping) else None
        if model:
            models.add(str(model))
        if kind == "user" and _is_human_prompt(content):
            out["human_turns"] += 1
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(str(b.get("text", "")) for b in content if isinstance(b, Mapping))
            else:
                text = ""
            words, chars = _words_chars(text)
            out["prompt_words"] += words
            out["prompt_chars"] += chars
        elif kind == "assistant" and isinstance(content, list):
            out["assistant_turns"] += 1
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                btype = block.get("type")
                if btype in ("text", "thinking"):
                    words, chars = _words_chars(
                        str(block.get("text") or block.get("thinking") or "")
                    )
                    out["reasoning_words"] += words
                    out["reasoning_chars"] += chars
                elif btype == "tool_use":
                    out["tool_calls"] += 1
                    tools[str(block.get("name") or "")] += 1
        elif kind == "file-history-snapshot":
            out["file_snapshots"] += 1
    out["tools"] = dict(tools)
    out["models"] = sorted(models)
    if out["human_turns"]:
        out["prompt_words_per_turn"] = round(out["prompt_words"] / out["human_turns"], 1)
        out["autonomy"] = round(out["assistant_turns"] / out["human_turns"], 1)
        out["actions_per_human_turn"] = round(out["tool_calls"] / out["human_turns"], 1)
    if out["assistant_turns"]:
        out["reasoning_words_per_turn"] = round(out["reasoning_words"] / out["assistant_turns"], 1)
    return out


# Tool name / Bash sub-kind -> atom, mirroring CLAUDE_RULES. build_panel_session
# needs the atom for each tool call in turn order; the rule adapter returns a
# flat sequence and drops the prompt text and timestamps the panel needs.
_BASH_ATOM: dict[str, Atom] = {
    "bash_test": ATOM_RUN_TEST,
    "bash_search": ATOM_SEARCH_REPO,
    "bash_vcs": ATOM_VERSION_CONTROL,
    "bash_package": ATOM_PACKAGE,
    "bash_lint": ATOM_LINT,
    "bash_run": ATOM_RUN_CODE,
    "bash": ATOM_OTHER,
}


def _tool_atom(name: str) -> Atom:
    """Map a non-Bash tool name to its atom (mirrors CLAUDE_RULES)."""
    lowered = name.lower()
    if lowered in _EDIT_TOOLS:
        return ATOM_EDIT
    if lowered in _READ_TOOLS:
        return ATOM_READ_FILE
    if lowered in _SEARCH_TOOLS or lowered in _WEB_TOOLS:
        return ATOM_SEARCH_REPO
    return ATOM_OTHER


def _clock(ts: object) -> str:
    """``HH:MM`` from an ISO timestamp, or ``''``."""
    if not isinstance(ts, str):
        return ""
    from datetime import datetime

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M")
    except ValueError:
        return ""


def _day(ts: object) -> str:
    """``Mon DD`` from an ISO timestamp, or ``''``."""
    if not isinstance(ts, str):
        return ""
    from datetime import datetime

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%b %d")
    except ValueError:
        return ""


def _iso_dt(ts: object):
    """Parse an ISO timestamp to a ``datetime``, or ``None``."""
    if not isinstance(ts, str):
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _prompt_text(content: Any) -> str:
    """The human's typed text from a user turn (text blocks only)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(b.get("text", ""))
            for b in content
            if isinstance(b, Mapping) and b.get("type") == "text"
        )
    return ""


# Tool names whose input carries an edited file path, for the per-cascade module
# rollup (where a goal landed). Lower-cased for matching.
_EDIT_TOOLS = frozenset(
    {
        "edit",
        "write",
        "multiedit",
        "notebookedit",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "create_file",
    }
)


def _module_dir(fp: str, root: str) -> str:
    """Directory of an edited file, made relative to the session root when possible.

    Keeps the rollup module-relative (e.g. ``src/license``) instead of an absolute
    home path, so the per-cascade rollup can be surfaced without leaking the
    machine's filesystem layout. Falls back to the trailing two path components.
    """
    p = Path(fp)
    if root:
        try:
            return str(p.relative_to(root).parent)
        except ValueError:
            pass
    parts = p.parent.parts
    return str(Path(*parts[-2:])) if len(parts) >= 2 else (p.parent.name or ".")


def to_shareable(record: Mapping[str, Any]) -> dict[str, Any]:
    """The one sanctioned export payload: atoms + hashed id + counts-only metadata.

    Drops the raw ``events`` (the transcript) entirely, so a system handed a
    ``to_shareable`` result is never given the originals -- local atomization is
    the privacy boundary. Pass an anonymized record (``load_claude_transcript``'s
    default) so ``trace_id``/``agent`` are already hashed.
    """
    metadata = record.get("metadata")
    counts = metadata if isinstance(metadata, dict) else summarize_transcript(record)
    return {
        "trace_id": str(record.get("trace_id", "")),
        "agent": str(record.get("agent", "")),
        "atoms": list(claude_code_adapter(record)),
        "metadata": counts,
    }


def build_panel_session(
    record: Mapping[str, Any], *, paraphrase: Callable[[str], str] | None = None
) -> dict[str, Any]:
    """Reduce one transcript to the live panel's per-session shape, for LOCAL view.

    Prompt-anchored: each human prompt opens a turn whose atom sequence is the
    agent's following tool calls (the same mapping as the fingerprint), tagged
    with the turn's clock time and model. The conversation (prompt text) is
    sampled for this LOCAL view; a ``paraphrase`` callable, when supplied,
    rewrites each prompt (style + identifiers stripped) instead of showing it
    raw -- the opt-in for sharing or screenshots. Either way this output is
    LOCAL; use ``to_shareable`` for anything that leaves the machine (it carries
    atoms and counts only, never prompt text).
    """
    lines = [line for line in record.get("events", []) if isinstance(line, Mapping)]
    sid = next((str(line["sessionId"]) for line in lines if line.get("sessionId")), "session")
    cwd = next((str(line["cwd"]) for line in lines if line.get("cwd")), "")
    workspace = Path(cwd).name or "local"
    # Real session timing from ISO timestamps -- correct across multi-day, resumed
    # sessions, unlike the per-turn HH:MM clock. `ended` is the recency sort key,
    # `date` the last-activity day so the rail reads in the same order it sorts.
    stamps = sorted(d for d in (_iso_dt(line.get("timestamp")) for line in lines) if d)
    ended = stamps[-1].isoformat() if stamps else ""
    date = _day(ended) if ended else ""
    duration_min = round((stamps[-1] - stamps[0]).total_seconds() / 60) if stamps else None
    models: set[str] = set()
    turns: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for line in lines:
        kind = line.get("type")
        message = line.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if kind == "user" and _is_human_prompt(content):
            if cur is not None and cur["seq"]:
                turns.append(cur)
            text = _prompt_text(content)
            prompt = paraphrase(text) if (paraphrase is not None and text) else text
            cur = {
                "t": _clock(line.get("timestamp")),
                "model": "",
                "prompt": prompt,
                "plan": "",
                "seq": [],
                "edits": {},
            }
        elif kind == "assistant" and isinstance(content, list):
            if cur is None:
                continue
            model = message.get("model") if isinstance(message, Mapping) else None
            if model:
                models.add(str(model))
                if not cur["model"]:
                    cur["model"] = str(model)
            for block in content:
                if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                    continue
                name = str(block.get("name") or "")
                if name.lower() == "bash":
                    tool_input = block.get("input")
                    command = (
                        tool_input.get("command", "") if isinstance(tool_input, Mapping) else ""
                    )
                    cur["seq"].append(_BASH_ATOM[_classify_bash(str(command))])
                else:
                    cur["seq"].append(_tool_atom(name))
                    if name.lower() in _EDIT_TOOLS:
                        tool_input = block.get("input")
                        fp = (
                            tool_input.get("file_path")
                            or tool_input.get("path")
                            or tool_input.get("notebook_path")
                            if isinstance(tool_input, Mapping)
                            else None
                        )
                        if fp:
                            d = _module_dir(str(fp), cwd)
                            cur["edits"][d] = cur["edits"].get(d, 0) + 1
        elif kind == "file-history-snapshot" and cur is not None:
            cur["seq"].append(ATOM_EDIT)
    if cur is not None and cur["seq"]:
        turns.append(cur)
    fallback_model = next(iter(sorted(models)), "")
    for turn in turns:
        if not turn["model"]:
            turn["model"] = fallback_model
    meta: dict[str, Any] = {
        "name": _anon_id(sid),
        "client": "Claude Code",
        "project": workspace,
        "id": _anon_id(sid),
        "date": date,
        "ended": ended,
        "durationMin": duration_min,
        "intent": "",
        "illustrative": False,
        "promptsParaphrased": paraphrase is not None,
        "models": [{"name": m} for m in sorted(models)],
    }
    return {"meta": meta, "turns": turns}


def _anon_id(value: str, length: int = 8) -> str:
    """Stable truncated SHA-256 of *value*. Not reversible."""
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()[:length]


def load_claude_transcript(path: str | Path, *, anonymize: bool = True) -> dict[str, Any]:
    """Read a ``.jsonl`` transcript into a single session record.

    Args:
        anonymize: When True (default), replace the session id and workspace
            path with stable hashes so neither appears in exported records or
            published outputs. Set False only for local debugging — never
            commit or share the result.
    """
    lines: list[dict[str, Any]] = []
    with open(path) as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                lines.append(parsed)
    raw_id = next(
        (str(line["sessionId"]) for line in lines if line.get("sessionId")), Path(path).stem
    )
    cwd = next((str(line["cwd"]) for line in lines if line.get("cwd")), "")
    if anonymize:
        trace_id = _anon_id(raw_id)
        agent = _anon_id(cwd or raw_id)
    else:
        trace_id = raw_id
        agent = Path(cwd).name or "unknown"
    record: dict[str, Any] = {"trace_id": trace_id, "agent": agent, "events": lines}
    record["metadata"] = summarize_transcript(record)
    return record


register_adapter("claude-code", claude_code_adapter, overwrite=True)

__all__ = [
    "ATOM_PROMPT_AI",
    "CLAUDE_RULES",
    "build_panel_session",
    "claude_code_adapter",
    "load_claude_transcript",
    "summarize_transcript",
    "to_shareable",
]
