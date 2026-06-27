"""Tests for `procgrep.ingest.adapters.function_xml`.

R2E-Gym and similar agents embed tool calls as ``<function=...><parameter=...>``
tags in the assistant's text. Covers execute_bash command classification,
file_editor operation mapping, finish -> submit, the think prefix for reasoning
prose, multiple functions per turn, and registration.
"""

from __future__ import annotations

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.function_xml import function_xml_adapter
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
)


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _bash(cmd: str) -> str:
    return f"<function=execute_bash>\n  <parameter=cmd>{cmd}</parameter>\n</function>"


def _editor(command: str) -> str:
    return f"<function=file_editor>\n  <parameter=command>{command}</parameter>\n  <parameter=path>/x.py</parameter>\n</function>"


def test_execute_bash_classified_via_command() -> None:
    rec = {"messages": [_assistant("Run the tests.\n" + _bash("pytest tests/"))]}
    assert function_xml_adapter(rec) == [ATOM_THINK, ATOM_RUN_TEST]


def test_file_editor_operations_map() -> None:
    rec = {
        "messages": [
            _assistant(_editor("view")),
            _assistant(_editor("create")),
            _assistant(_editor("str_replace")),
        ]
    }
    # No prose outside the tags here, so no think steps.
    assert function_xml_adapter(rec) == [ATOM_READ_FILE, ATOM_CREATE_FILE, ATOM_EDIT]


def test_finish_is_submit() -> None:
    rec = {"messages": [_assistant("<function=finish>\n  <parameter=command>submit</parameter>\n</function>")]}
    assert function_xml_adapter(rec) == [ATOM_SUBMIT]


def test_reasoning_prose_emits_one_think() -> None:
    rec = {"messages": [_assistant("Let me explore the repo first.\n" + _bash("cd /repo && python repro.py"))]}
    # Prose -> think; the cd-chained python script -> run_code (shared classifier).
    assert function_xml_adapter(rec) == [ATOM_THINK, ATOM_RUN_CODE]


def test_multiple_functions_in_one_turn() -> None:
    rec = {"messages": [_assistant(_bash("grep -r foo .") + "\n" + _bash("cat a.py"))]}
    # Two functions, no surrounding prose -> no think, two action atoms.
    assert function_xml_adapter(rec) == [ATOM_SEARCH_REPO, ATOM_READ_FILE]


def test_non_assistant_and_empty_turns_skipped() -> None:
    rec = {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": ""},
        ]
    }
    assert function_xml_adapter(rec) == []


def test_degrades_on_missing_messages() -> None:
    assert function_xml_adapter({}) == []
    assert function_xml_adapter({"messages": "not a list"}) == []


def test_registered_under_name() -> None:
    assert get_adapter("function-xml") is function_xml_adapter
