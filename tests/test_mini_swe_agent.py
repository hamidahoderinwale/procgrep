"""Tests for `procgrep.ingest.adapters.mini_swe_agent`."""

from __future__ import annotations

from procgrep.canonicalize import canonicalize, get_adapter
from procgrep.ingest.adapters.mini_swe_agent import (
    _classify_command,
    _strip_chain_prefix,
    mini_swe_agent_adapter,
)
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
)

# --- chain stripping --------------------------------------------------------


def test_strip_chain_prefix_drops_cd_and_chain() -> None:
    assert _strip_chain_prefix("cd /repo && pytest") == "pytest"
    assert _strip_chain_prefix("cd /repo ; pytest") == "pytest"
    assert _strip_chain_prefix("cd /repo && cd sub && pytest") == "pytest"


def test_strip_chain_prefix_passes_through_when_no_cd() -> None:
    assert _strip_chain_prefix("pytest tests/") == "pytest tests/"
    assert _strip_chain_prefix("  cat foo  ") == "cat foo"


# --- single-command classification -----------------------------------------


def test_classify_read_family() -> None:
    assert _classify_command("cat src/foo.py") == ATOM_READ_FILE
    assert _classify_command("head -100 file") == ATOM_READ_FILE
    assert _classify_command("less file.log") == ATOM_READ_FILE


def test_classify_search_family() -> None:
    assert _classify_command("grep -r 'foo' src/") == ATOM_SEARCH_REPO
    assert _classify_command("rg pattern") == ATOM_SEARCH_REPO
    assert _classify_command("find . -name '*.py'") == ATOM_SEARCH_REPO


def test_classify_edit_family_direct() -> None:
    assert _classify_command("sed -i s/foo/bar/g file.py") == ATOM_EDIT
    assert _classify_command("patch -p1 < fix.diff") == ATOM_EDIT


def test_classify_edit_family_via_redirection() -> None:
    assert _classify_command("echo 'x = 1' > x.py") == ATOM_EDIT
    assert _classify_command("cat <<EOF > new.py") == ATOM_EDIT


def test_classify_run_test_family() -> None:
    assert _classify_command("pytest tests/") == ATOM_RUN_TEST
    assert _classify_command("python -m pytest tests/") == ATOM_RUN_TEST
    assert _classify_command("python -m unittest discover") == ATOM_RUN_TEST
    assert _classify_command("python -m django test") == ATOM_RUN_TEST


def test_classify_run_test_via_cd_chain() -> None:
    assert _classify_command("cd /repo && pytest tests/foo.py") == ATOM_RUN_TEST


def test_classify_create_and_delete_file() -> None:
    assert _classify_command("mkdir -p src/new") == ATOM_CREATE_FILE
    assert _classify_command("touch __init__.py") == ATOM_CREATE_FILE
    assert _classify_command("rm -rf build/") == ATOM_DELETE_FILE
    assert _classify_command("rmdir empty") == ATOM_DELETE_FILE


def test_classify_submit_family() -> None:
    assert _classify_command("git diff") == ATOM_SUBMIT
    assert _classify_command("git apply patch.diff") == ATOM_SUBMIT


def test_classify_unknown_command_yields_other() -> None:
    assert _classify_command("weird_unknown_cmd --foo") == ATOM_OTHER
    assert _classify_command("") == ATOM_OTHER
    assert _classify_command("   ") == ATOM_OTHER


# --- full-trace traversal --------------------------------------------------


def _make_trace(messages: list[dict], exit_status: str | None = "Submitted") -> dict:
    """Build a minimal mini-swe-agent .traj.json-shaped record."""
    record: dict = {
        "messages": messages,
        "trajectory_format": "mini-swe-agent-1.1",
    }
    if exit_status is not None:
        record["info"] = {"exit_status": exit_status}
    return record


def test_adapter_emits_one_atom_per_action() -> None:
    record = _make_trace(
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": "I'll look at the file first.",
                "extra": {
                    "actions": [
                        {"command": "cat src/foo.py"},
                        {"command": "grep 'def bar' src/"},
                    ]
                },
            },
            {"role": "tool", "content": "<output>"},
            {
                "role": "assistant",
                "content": "Now edit it.",
                "extra": {"actions": [{"command": "sed -i s/x/y/g src/foo.py"}]},
            },
            {"role": "tool", "content": "<output>"},
            {
                "role": "assistant",
                "content": "Run tests.",
                "extra": {"actions": [{"command": "pytest tests/"}]},
            },
        ]
    )
    atoms = mini_swe_agent_adapter(record)
    # Three assistant turns, each with a non-empty thought; three turns of
    # bash actions (2 + 1 + 1). Expect: think, read, search, think, edit,
    # think, run_test.
    assert atoms == [
        ATOM_THINK,
        ATOM_READ_FILE,
        ATOM_SEARCH_REPO,
        ATOM_THINK,
        ATOM_EDIT,
        ATOM_THINK,
        ATOM_RUN_TEST,
    ]


def test_adapter_appends_error_on_non_submitted_exit() -> None:
    record = _make_trace(
        [
            {
                "role": "assistant",
                "content": "",
                "extra": {"actions": [{"command": "cat foo"}]},
            },
        ],
        exit_status="LimitsExceeded",
    )
    atoms = mini_swe_agent_adapter(record)
    assert atoms[-1] == ATOM_ERROR
    assert atoms[:-1] == [ATOM_READ_FILE]


def test_adapter_ignores_non_assistant_messages() -> None:
    record = _make_trace(
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "tool", "content": "..."},
        ]
    )
    assert mini_swe_agent_adapter(record) == []


def test_adapter_skips_assistant_messages_with_no_actions_or_thought() -> None:
    record = _make_trace(
        [
            {"role": "assistant", "content": "", "extra": {"actions": []}},
        ]
    )
    assert mini_swe_agent_adapter(record) == []


def test_adapter_emits_think_when_thought_but_no_actions() -> None:
    record = _make_trace(
        [
            {"role": "assistant", "content": "I am thinking aloud.", "extra": {}},
        ]
    )
    assert mini_swe_agent_adapter(record) == [ATOM_THINK]


def test_adapter_degrades_on_missing_messages_key() -> None:
    assert mini_swe_agent_adapter({}) == []
    assert mini_swe_agent_adapter({"messages": "not a list"}) == []


def test_adapter_degrades_on_malformed_action_entry() -> None:
    record = _make_trace(
        [
            {
                "role": "assistant",
                "content": "",
                "extra": {
                    "actions": [
                        "not a dict",
                        {"command": "cat foo"},
                        42,
                    ]
                },
            }
        ]
    )
    atoms = mini_swe_agent_adapter(record)
    # Only the valid action contributes; non-dict entries are skipped.
    assert atoms == [ATOM_READ_FILE]


# --- adapter registration --------------------------------------------------


def test_adapter_registered_under_name_mini_swe_agent() -> None:
    adapter = get_adapter("mini-swe-agent")
    record = _make_trace(
        [
            {
                "role": "assistant",
                "content": "",
                "extra": {"actions": [{"command": "pytest tests/"}]},
            }
        ]
    )
    assert adapter(record) == [ATOM_RUN_TEST]


def test_canonicalize_via_adapter_round_trip() -> None:
    record = {
        "trace_id": "mini-001",
        "agent": "deepseek-coder-6.7b",
        "group": "open-mid",
        "messages": [
            {
                "role": "assistant",
                "content": "I'll search for the bug.",
                "extra": {
                    "actions": [
                        {"command": "rg 'def fetch' src/"},
                        {"command": "cat src/api.py"},
                    ]
                },
            }
        ],
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {"exit_status": "Submitted"},
    }
    traces = canonicalize([record], adapter="mini-swe-agent")
    assert len(traces) == 1
    t = traces[0]
    assert t.trace_id == "mini-001"
    assert t.agent == "deepseek-coder-6.7b"
    assert t.group == "open-mid"
    assert t.atoms == [ATOM_THINK, ATOM_SEARCH_REPO, ATOM_READ_FILE]
