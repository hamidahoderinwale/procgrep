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
    Bash with a test/build command                      -> run_test
    Bash (other), Agent, Task*, MCP tools, ...          -> other

Design decisions:

- Only the action *structure* is read -- tool names, a Bash command string, and
  event types. Message text is never inspected, so a transcript can be
  fingerprinted without exposing its content.

- ``prompt_ai`` is the human->AI turn, matching the cursor-companion atom so the
  two sources share one alphabet. Claude Code is agent-dense (one prompt, then a
  long tool run) where Cursor is interleaved; the shared atoms make that
  contrast measurable rather than hiding it.

- Bash is classified by its command: test/build commands map to ``run_test``
  (the canonical proxy), everything else falls through to ``other`` rather than
  inflating the test signal -- a one-size ``Bash -> run_test`` rule would.
"""

from __future__ import annotations

import collections
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from procgrep.canonicalize import EventRule, field_in, make_event_adapter, register_adapter
from procgrep.types import (
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    Atom,
    AtomSequence,
)

# Matches the cursor-companion atom so both human+AI sources share one alphabet.
ATOM_PROMPT_AI: Atom = "prompt_ai"

_EDIT_TOOLS = {"edit", "write", "notebookedit", "multiedit"}
_READ_TOOLS = {"read", "notebookread"}
_SEARCH_TOOLS = {"grep", "glob"}
_WEB_TOOLS = {"websearch", "webfetch"}
_TEST_CMD = re.compile(
    r"(pytest|(^|\s|/)tests?\b|unittest|jest|mocha|vitest|tox|"
    r"go test|cargo test|npm (run )?test|yarn test|make test|gradle test)",
    re.IGNORECASE,
)


def _is_test_bash(event: Mapping[str, Any]) -> bool:
    """A Bash action whose command looks like running tests or a build."""
    if str(event.get("kind", "")).lower() != "bash":
        return False
    return bool(_TEST_CMD.search(str(event.get("command", ""))))


CLAUDE_RULES: tuple[EventRule, ...] = (
    EventRule(field_in("kind", {"user_prompt"}), (ATOM_PROMPT_AI,)),
    EventRule(field_in("kind", {*_EDIT_TOOLS, "file_snapshot"}), (ATOM_EDIT,)),
    EventRule(field_in("kind", _READ_TOOLS), (ATOM_READ_FILE,)),
    EventRule(field_in("kind", _SEARCH_TOOLS), (ATOM_SEARCH_REPO,)),
    EventRule(field_in("kind", _WEB_TOOLS), (ATOM_SEARCH_REPO,)),
    EventRule(_is_test_bash, (ATOM_RUN_TEST,)),
)

_MAP = make_event_adapter(rules=CLAUDE_RULES)


def _is_human_prompt(content: Any) -> bool:
    """A user turn typed by the human, not a tool-result echoed back as a user event."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        has_text = any(isinstance(b, Mapping) and b.get("type") == "text" for b in content)
        has_tool_result = any(
            isinstance(b, Mapping) and b.get("type") == "tool_result" for b in content
        )
        return has_text and not has_tool_result
    return False


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
                    tool_input = block.get("input")
                    command = (
                        tool_input.get("command", "") if isinstance(tool_input, Mapping) else ""
                    )
                    events.append({"kind": str(block.get("name") or ""), "command": command})
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
                    words, chars = _words_chars(str(block.get("text") or block.get("thinking") or ""))
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


def load_claude_transcript(path: str | Path) -> dict[str, Any]:
    """Read a ``.jsonl`` transcript into a single session record.

    The session id and the working directory's basename (a workspace/developer
    proxy) are pulled from the first line that carries them.
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
    trace_id = next((str(line["sessionId"]) for line in lines if line.get("sessionId")), Path(path).stem)
    cwd = next((str(line["cwd"]) for line in lines if line.get("cwd")), "")
    agent = Path(cwd).name or "unknown"
    record: dict[str, Any] = {"trace_id": trace_id, "agent": agent, "events": lines}
    record["metadata"] = summarize_transcript(record)
    return record


register_adapter("claude-code", claude_code_adapter, overwrite=True)

__all__ = [
    "ATOM_PROMPT_AI",
    "CLAUDE_RULES",
    "claude_code_adapter",
    "load_claude_transcript",
    "summarize_transcript",
]
