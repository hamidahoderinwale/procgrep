"""Tests for `procgrep.ingest.adapters.claude_code`.

Treats Claude Code transcripts as a standard ingest adapter: a separate tool's
trace source woven in via the shared feature-based mapper. Covers the
line-to-action flatten (including multi-tool assistant lines), the human-prompt
vs tool-result distinction, Bash command classification, and registration.
"""

from __future__ import annotations

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.claude_code import (
    ATOM_PROMPT_AI,
    claude_code_adapter,
    summarize_transcript,
)
from procgrep.types import (
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
)


def _assistant(*tools: tuple[str, str]) -> dict:  # type: ignore[type-arg]
    """An assistant line carrying tool_use blocks: (name, command)."""
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": name, "input": {"command": cmd}}
                for name, cmd in tools
            ]
        },
    }


def _user_prompt(text: str = "do the thing") -> dict:  # type: ignore[type-arg]
    return {"type": "user", "message": {"content": text}}


def _tool_result() -> dict:  # type: ignore[type-arg]
    return {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}}


def test_human_prompt_maps_to_prompt_ai() -> None:
    record = {"events": [_user_prompt()]}
    assert claude_code_adapter(record) == [ATOM_PROMPT_AI]


def test_tool_result_user_event_is_not_a_prompt() -> None:
    # A tool result echoed back as a user event is the agent's output, not a turn.
    record = {"events": [_tool_result()]}
    assert claude_code_adapter(record) == []


def test_assistant_tools_map_to_atoms() -> None:
    record = {"events": [_assistant(("Read", ""), ("Edit", ""), ("Grep", ""))]}
    assert claude_code_adapter(record) == [ATOM_READ_FILE, ATOM_EDIT, ATOM_SEARCH_REPO]


def test_multi_tool_line_explodes_into_several_atoms() -> None:
    record = {"events": [_assistant(("Write", ""), ("Write", ""))]}
    assert claude_code_adapter(record) == [ATOM_EDIT, ATOM_EDIT]


def test_bash_test_command_is_run_test_other_bash_is_other() -> None:
    record = {"events": [_assistant(("Bash", "pytest -q"), ("Bash", "git status"), ("Bash", "ls -la"))]}
    # test -> run_test, git -> version_control, unrecognized -> other
    assert claude_code_adapter(record) == [ATOM_RUN_TEST, "version_control", "other"]


def test_file_history_snapshot_is_an_edit() -> None:
    record = {"events": [{"type": "file-history-snapshot"}]}
    assert claude_code_adapter(record) == [ATOM_EDIT]


def test_prompt_anchored_session_orders_correctly() -> None:
    record = {
        "events": [
            _user_prompt("fix the bug"),
            _assistant(("Grep", ""), ("Read", "")),
            _assistant(("Edit", ""), ("Bash", "pytest")),
            _tool_result(),
        ]
    }
    assert claude_code_adapter(record) == [
        ATOM_PROMPT_AI,
        ATOM_SEARCH_REPO,
        ATOM_READ_FILE,
        ATOM_EDIT,
        ATOM_RUN_TEST,
    ]


def test_empty_and_malformed_records_are_lenient() -> None:
    assert claude_code_adapter({}) == []
    assert claude_code_adapter({"events": "not a list"}) == []
    assert claude_code_adapter({"events": [None, 42, "x"]}) == []


def test_adapter_is_registered_under_claude_code() -> None:
    assert get_adapter("claude-code") is claude_code_adapter


def test_summarize_counts_turns_words_tools_without_storing_text() -> None:
    record = {
        "events": [
            _user_prompt("fix the bug now"),  # 4 words
            _assistant(("Read", ""), ("Edit", "")),
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "I will look"}]}},
            {"type": "file-history-snapshot"},
            _tool_result(),  # not a human turn
        ]
    }
    s = summarize_transcript(record)
    assert s["human_turns"] == 1
    assert s["prompt_words"] == 4
    assert s["tool_calls"] == 2
    assert s["tools"] == {"Read": 1, "Edit": 1}
    assert s["reasoning_words"] == 3
    assert s["file_snapshots"] == 1
    assert s["prompt_words_per_turn"] == 4.0
    # only counts are retained -- no message text leaks into the summary
    assert "fix the bug now" not in str(s)


def test_bash_subclassified_into_safe_categories() -> None:
    """Bash commands map to category atoms by leading verb; command is discarded."""
    from procgrep.ingest.adapters.claude_code import _classify_bash, _flatten, claude_code_adapter

    assert _classify_bash("pytest -q") == "bash_test"
    assert _classify_bash("git commit -m x") == "bash_vcs"
    assert _classify_bash("pip install ruff") == "bash_package"  # install wins over the lint word
    assert _classify_bash("ruff check .") == "bash_lint"
    assert _classify_bash("rg pattern") == "bash_search"
    assert _classify_bash("python app.py") == "bash_run"
    assert _classify_bash("ls -la") == "bash"  # unknown stays generic -> other

    record = {"trace_id": "t", "agent": "a", "events": [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "git status"}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "echo hi"}},
        ]}},
    ]}
    assert claude_code_adapter(record) == ["version_control", "other"]
    # privacy: the raw command never reaches the flattened event
    flat = _flatten(record)
    assert all("command" not in event for event in flat)


def test_to_shareable_emits_atoms_only_no_transcript() -> None:
    from procgrep.ingest.adapters.claude_code import to_shareable

    record = {"trace_id": "h", "agent": "w", "events": [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Edit", "input": {}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "git status"}},
        ]}},
    ]}
    sh = to_shareable(record)
    assert sh["atoms"] == ["edit", "version_control"]
    assert "events" not in sh  # the raw transcript never rides along
    assert isinstance(sh["metadata"], dict)


def test_build_panel_session_is_prompt_anchored_and_local() -> None:
    from procgrep.ingest.adapters.claude_code import build_panel_session

    def _u(text: str, ts: str, **kw: object) -> dict:  # type: ignore[type-arg]
        return {"type": "user", "timestamp": ts, "message": {"content": text}, **kw}

    def _a(blocks: list, ts: str) -> dict:  # type: ignore[type-arg]
        return {"type": "assistant", "timestamp": ts, "message": {"model": "claude-opus-4-8", "content": blocks}}

    record = {"events": [
        _u("scaffold it", "2026-06-17T14:08:00Z", sessionId="sess-1", cwd="/home/u/learning-from-dev"),
        _a([{"type": "text", "text": "read then edit"},
            {"type": "tool_use", "name": "Read", "input": {}},
            {"type": "tool_use", "name": "Edit", "input": {}}], "2026-06-17T14:08:30Z"),
        _u("run the tests", "2026-06-17T14:10:00Z"),
        _a([{"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}}], "2026-06-17T14:10:20Z"),
    ]}
    ps = build_panel_session(record)  # no paraphraser -> conversation sampled raw (local)
    assert ps["meta"]["project"] == "learning-from-dev"
    assert ps["meta"]["client"] == "Claude Code"
    assert [t["seq"] for t in ps["turns"]] == [["read_file", "edit"], ["run_test"]]
    assert ps["turns"][0]["t"] == "14:08"
    assert ps["turns"][0]["prompt"] == "scaffold it"  # sampled for the local view
    assert ps["meta"]["models"] == [{"name": "claude-opus-4-8"}]

    ps2 = build_panel_session(record, paraphrase=lambda s: s.upper())
    assert ps2["turns"][0]["prompt"] == "SCAFFOLD IT"  # shown, rewritten
    assert ps2["meta"]["promptsParaphrased"] is True
