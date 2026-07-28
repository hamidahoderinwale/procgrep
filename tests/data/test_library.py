"""Tests for `procgrep.library` and ProcedureSpec YAML round-tripping."""

from __future__ import annotations

import pytest

from procgrep.library import ProcedureLibrary
from procgrep.reward import Penalty, Phase, ProcedureSpec


def _spec(name: str = "test_after_edit") -> ProcedureSpec:
    return ProcedureSpec(
        name=name,
        phases=(
            Phase(
                name="verify",
                reward=0.5,
                require_any=("run_test",),
                before_first="submit",
                min_count=2,
            ),
        ),
        penalties=(Penalty(name="streak", reward=0.25, max_run=5),),
    )


def test_spec_yaml_round_trips(tmp_path):  # type: ignore[no-untyped-def]
    spec = _spec()
    path = tmp_path / "s.yaml"
    spec.to_yaml(path)
    back = ProcedureSpec.from_yaml(path)
    assert back.name == spec.name
    assert [
        (p.name, p.reward, p.require_any, p.before_first, p.min_count) for p in back.phases
    ] == [("verify", 0.5, ("run_test",), "submit", 2)]
    assert [(p.name, p.reward, p.max_run, p.forbid_sequence) for p in back.penalties] == [
        ("streak", 0.25, 5, ())
    ]
    assert back.floor == 0.0
    assert back.ceiling == 1.0


def test_library_save_load_names(tmp_path):  # type: ignore[no-untyped-def]
    lib = ProcedureLibrary(tmp_path / "lib")
    assert lib.names() == []
    lib.save("test_after_edit", _spec())
    lib.save(
        "explore first", ProcedureSpec(name="x", phases=(Phase("explore", 0.1, ("search_repo",)),))
    )
    assert "test_after_edit" in lib
    assert "test_after_edit" in lib.names()
    assert len(lib) == 2
    loaded = lib.load("test_after_edit")
    assert loaded.name == "test_after_edit"
    # save stamps the library name onto the spec even if it differed
    assert lib.load("explore first").name == "explore first"
    assert all(isinstance(spec, ProcedureSpec) for _, spec in lib)


def test_library_missing_raises(tmp_path):  # type: ignore[no-untyped-def]
    with pytest.raises(KeyError, match="no procedure"):
        ProcedureLibrary(tmp_path).load("nope")
