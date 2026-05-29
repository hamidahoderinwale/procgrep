"""Tests for the SWE-smith shell-action classifier.

These cover :func:`classify_swe_smith_action` (raw bash command strings to
canonical/native atom pairs) and the round-trip via
:func:`swe_smith_canonical_projection`.
"""

from __future__ import annotations

import pytest

from procgrep.adapters.swe_smith import (
    classify_swe_smith_action,
    swe_smith_canonical_projection,
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
)


def test_empty_action_is_scaffold_error() -> None:
    assert classify_swe_smith_action("") == (ATOM_ERROR, "empty_or_error")


def test_whitespace_only_is_scaffold_error() -> None:
    assert classify_swe_smith_action("   \n  ") == (ATOM_ERROR, "empty_or_error")


def test_submit_is_canonical() -> None:
    assert classify_swe_smith_action("submit") == (ATOM_SUBMIT, "submit")


def test_str_replace_editor_view() -> None:
    canonical, native = classify_swe_smith_action("str_replace_editor view /testbed/foo.py")
    assert canonical == ATOM_READ_FILE
    assert native == "str_replace_editor:view"


def test_str_replace_editor_create() -> None:
    canonical, native = classify_swe_smith_action(
        "str_replace_editor create /testbed/new.py --file_text 'x = 1'"
    )
    assert canonical == ATOM_CREATE_FILE
    assert native == "str_replace_editor:create"


def test_str_replace_editor_str_replace() -> None:
    canonical, native = classify_swe_smith_action(
        "str_replace_editor str_replace /testbed/foo.py --old_str 'a' --new_str 'b'"
    )
    assert canonical == ATOM_EDIT
    assert native == "str_replace_editor:str_replace"


def test_str_replace_editor_insert() -> None:
    canonical, native = classify_swe_smith_action(
        "str_replace_editor insert /testbed/foo.py --insert_line 10 --new_str 'x'"
    )
    assert canonical == ATOM_EDIT
    assert native == "str_replace_editor:insert"


def test_strips_leading_cd_chain() -> None:
    # ``cd /testbed && X`` is structural plumbing -- the cd is the
    # wrapper, not a real chain operation -- so native is just X's tag.
    canonical, native = classify_swe_smith_action("cd /testbed && python reproduce_bug.py")
    assert canonical == ATOM_RUN_TEST
    assert native == "python:script"


def test_real_chain_keeps_chain_prefix() -> None:
    # When stripping the cd still leaves an ``&&``, that's a real chain.
    canonical, native = classify_swe_smith_action("cd /testbed && find . -name foo && grep bar baz")
    assert canonical == ATOM_SEARCH_REPO
    assert native == "chain:find"


def test_python_pytest_invocation() -> None:
    canonical, native = classify_swe_smith_action("python -m pytest tests/")
    assert canonical == ATOM_RUN_TEST
    assert native == "python:pytest"


def test_python_script_invocation() -> None:
    canonical, native = classify_swe_smith_action("python /testbed/test_edge_cases.py")
    assert canonical == ATOM_RUN_TEST
    assert native == "python:script"


def test_bare_pytest_invocation() -> None:
    # Bare `pytest` is also a known SWE-agent action name, so the bare
    # ATOM_MAP entry wins.
    canonical, _native = classify_swe_smith_action("pytest")
    assert canonical == ATOM_RUN_TEST


def test_find_search() -> None:
    canonical, native = classify_swe_smith_action('find /testbed -type f -name "*.py"')
    assert canonical == ATOM_SEARCH_REPO
    assert native == "find"


def test_find_pipe_grep_is_pipe_compound() -> None:
    canonical, native = classify_swe_smith_action(
        'find /testbed -name "*.py" | xargs grep -l "foo"'
    )
    assert canonical == ATOM_SEARCH_REPO
    assert native == "pipe:find"


def test_grep_search() -> None:
    canonical, native = classify_swe_smith_action('grep -rn "MyClass" /testbed')
    assert canonical == ATOM_SEARCH_REPO
    # Bare "grep" is a known classic action name; rg/ag are not so bare
    # "grep" lands via the bare-action path with native==`grep`.
    assert native == "grep"


def test_rm_delete() -> None:
    canonical, native = classify_swe_smith_action("rm /testbed/reproduce_bug.py")
    assert canonical == ATOM_DELETE_FILE
    assert native == "rm"


def test_ls_read() -> None:
    canonical, native = classify_swe_smith_action("ls /testbed")
    assert canonical == ATOM_READ_FILE
    assert native == "ls"


def test_cat_read() -> None:
    canonical, native = classify_swe_smith_action("cat /testbed/foo.py")
    assert canonical == ATOM_READ_FILE
    assert native == "cat"


def test_mkdir_creates() -> None:
    canonical, native = classify_swe_smith_action("mkdir /testbed/newdir")
    assert canonical == ATOM_CREATE_FILE
    assert native == "mkdir_touch"


def test_chmod_other() -> None:
    canonical, native = classify_swe_smith_action("chmod +x /testbed/script.sh")
    assert canonical == ATOM_OTHER
    assert native == "permissions"


def test_echo_redirect_creates() -> None:
    canonical, native = classify_swe_smith_action('echo "x = 1" > /testbed/config.py')
    assert canonical == ATOM_CREATE_FILE
    assert native == "echo_redirect"


def test_unknown_command_falls_through() -> None:
    canonical, native = classify_swe_smith_action("weird_unknown_command arg1 arg2")
    assert canonical == ATOM_OTHER
    assert native == "other:weird_unknown_command"


def test_cd_chain_with_str_replace_editor() -> None:
    # cd is the wrapper, not a chain operation.
    canonical, native = classify_swe_smith_action(
        "cd /testbed && str_replace_editor view /testbed/foo.py"
    )
    assert canonical == ATOM_READ_FILE
    assert native == "str_replace_editor:view"


def test_canonical_projection_known_native() -> None:
    assert swe_smith_canonical_projection("str_replace_editor:view") == ATOM_READ_FILE
    assert swe_smith_canonical_projection("python:pytest") == ATOM_RUN_TEST
    assert swe_smith_canonical_projection("submit") == ATOM_SUBMIT
    assert swe_smith_canonical_projection("empty_or_error") == ATOM_ERROR


def test_canonical_projection_strips_chain_prefix() -> None:
    assert swe_smith_canonical_projection("chain:python:script") == ATOM_RUN_TEST
    assert swe_smith_canonical_projection("chain:find") == ATOM_SEARCH_REPO


def test_canonical_projection_strips_pipe_prefix() -> None:
    assert swe_smith_canonical_projection("pipe:find") == ATOM_SEARCH_REPO


def test_canonical_projection_unknown_native_falls_back() -> None:
    assert swe_smith_canonical_projection("other:unknown_cmd") == ATOM_OTHER
    assert swe_smith_canonical_projection("str_replace_editor:never_seen") == ATOM_OTHER


@pytest.mark.parametrize(
    ("action", "expected_canonical"),
    [
        # Patterns observed in the real 200-trace sample
        ("cd /testbed && python reproduce_error.py", ATOM_RUN_TEST),
        ('find /testbed -type f -name "*.py" | grep -v "__pycache__" | sort', ATOM_SEARCH_REPO),
        ("cd /testbed && python test_edge_cases.py", ATOM_RUN_TEST),
        ("python /testbed/reproduce_error.py", ATOM_RUN_TEST),
        ("rm /testbed/reproduce_error.py /testbed/test_edge_cases.py", ATOM_DELETE_FILE),
        ("cd /testbed && python -m pytest", ATOM_RUN_TEST),
        ("cd /testbed && python -m unittest discover", ATOM_RUN_TEST),
        ("submit", ATOM_SUBMIT),
    ],
)
def test_real_corpus_patterns(action: str, expected_canonical: str) -> None:
    canonical, _ = classify_swe_smith_action(action)
    assert canonical == expected_canonical, f"failed: {action!r} -> {canonical}"


# --- Classic SWE-agent verbs with arguments (older submissions) ---


def test_bare_verb_with_args_edit() -> None:
    canonical, native = classify_swe_smith_action("edit /path/to/file 100:120\nnew_content")
    assert canonical == ATOM_EDIT
    assert native == "edit"


def test_bare_verb_with_args_open() -> None:
    canonical, native = classify_swe_smith_action("open /testbed/foo.py")
    assert canonical == ATOM_READ_FILE
    assert native == "open"


def test_bare_verb_with_args_scroll_down() -> None:
    canonical, native = classify_swe_smith_action("scroll_down")
    assert canonical == ATOM_READ_FILE
    assert native == "scroll_down"


def test_bare_verb_with_args_goto() -> None:
    canonical, native = classify_swe_smith_action("goto 50")
    assert canonical == ATOM_READ_FILE
    assert native == "goto"


def test_bare_verb_with_args_search_dir() -> None:
    canonical, native = classify_swe_smith_action("search_dir 'class ParkingLot' /testbed")
    assert canonical == ATOM_SEARCH_REPO
    assert native == "search_dir"


def test_bash_rules_still_win_over_bare_verb_lookup() -> None:
    """``find /testbed -name foo`` must classify via the bash-find rule
    (which preserves the compound shape), not via the bare-verb fallback,
    even though ``find`` is not in ``_BARE_ACTION_MAP`` (only ``find_file``).
    """
    canonical, native = classify_swe_smith_action("find /testbed -name foo")
    assert canonical == ATOM_SEARCH_REPO
    assert native == "find"
