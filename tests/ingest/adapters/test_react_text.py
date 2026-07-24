"""Tests for `procgrep.ingest.adapters.react_text`."""

from __future__ import annotations

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.react_text import (
    _commands,
    _turn_text,
    react_text_adapter,
)
from procgrep.types import (
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_THINK,
)

# --- text + fenced-block extraction -----------------------------------------


def test_turn_text_prefers_content_then_text() -> None:
    assert _turn_text({"content": "hi", "text": "other"}) == "hi"
    assert _turn_text({"text": "from text"}) == "from text"
    assert _turn_text({"content": "   "}) == ""
    assert _turn_text({}) == ""


def test_commands_extracts_leading_line_of_each_fence() -> None:
    text = "THOUGHT: look\n```bash\nfind . -name '*.py'\ngrep foo\n```"
    assert _commands(text) == ["find . -name '*.py'"]


def test_commands_handles_multiple_blocks_and_skips_comments() -> None:
    text = "```bash\n# a comment\npytest tests/\n```\nthen\n```\nsed -i s/a/b/ x.py\n```"
    assert _commands(text) == ["pytest tests/", "sed -i s/a/b/ x.py"]


def test_commands_empty_when_no_fence() -> None:
    assert _commands("just prose, no code") == []


# --- full adapter -----------------------------------------------------------


def test_adapter_assistant_role_with_content_field() -> None:
    record = {
        "messages": [
            {"role": "assistant", "content": "explore\n```bash\ngrep -r foo src/\n```"},
        ]
    }
    assert react_text_adapter(record) == [ATOM_THINK, ATOM_SEARCH_REPO]


def test_adapter_ai_role_with_text_field() -> None:
    record = {
        "messages": [
            {"role": "ai", "text": "run it\n```bash\npytest\n```"},
        ]
    }
    assert react_text_adapter(record) == [ATOM_THINK, ATOM_RUN_TEST]


def test_adapter_multi_turn_and_skips_non_assistant() -> None:
    record = {
        "messages": [
            {"role": "system", "text": "sys"},
            {"role": "user", "content": "fix it"},
            {"role": "ai", "text": "look\n```bash\ncat a.py\n```"},
            {"role": "ai", "text": "patch\n```bash\nsed -i s/a/b/ a.py\n```"},
        ]
    }
    assert react_text_adapter(record) == [ATOM_THINK, ATOM_READ_FILE, ATOM_THINK, ATOM_EDIT]


def test_adapter_degrades_on_bad_input() -> None:
    assert react_text_adapter({}) == []
    assert react_text_adapter({"messages": None}) == []
    assert react_text_adapter({"messages": [{"role": "ai", "text": "no commands here"}]}) == [
        ATOM_THINK
    ]


def test_registered_under_react_text() -> None:
    record = {"messages": [{"role": "ai", "text": "go\n```bash\npytest\n```"}]}
    assert get_adapter("react-text")(record) == [ATOM_THINK, ATOM_RUN_TEST]
