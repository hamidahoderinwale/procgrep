"""Offline tests for the NL->regex compiler: the API is always mocked."""

from __future__ import annotations

import json
from typing import Any

import pytest

from procgrep import ask
from procgrep.ask import AskError, CompiledQuery, compile_query


def _api_response(parsed: dict[str, Any], stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": json.dumps(parsed)}],
    }


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Route _post_messages to a canned response, recording the payload."""
    seen: dict[str, Any] = {"response": _api_response({})}

    def fake_post(payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
        seen["payload"] = payload
        seen["api_key"] = api_key
        response: dict[str, Any] = seen["response"]
        return response

    monkeypatch.setattr(ask, "_post_messages", fake_post)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return seen


def test_expressible_question_returns_compiled_regex(capture: dict[str, Any]) -> None:
    capture["response"] = _api_response(
        {
            "expressible": True,
            "regex": "^(?:(?!run_test).)*submit",
            "paraphrase": "submit appears with no run_test anywhere before it",
            "reason": None,
        }
    )
    out = compile_query("did it submit without testing?")
    assert isinstance(out, CompiledQuery)
    assert out.expressible
    assert out.regex == "^(?:(?!run_test).)*submit"
    assert out.model == ask.DEFAULT_MODEL


def test_inexpressible_question_carries_reason_not_regex(capture: dict[str, Any]) -> None:
    capture["response"] = _api_response(
        {
            "expressible": False,
            "regex": None,
            "paraphrase": None,
            "reason": "needs a step window; nearest expressible: (edit ){2,}",
        }
    )
    out = compile_query("edited the same file twice within five steps?")
    assert not out.expressible
    assert out.regex is None
    assert "window" in (out.reason or "")


def test_uncompilable_regex_from_model_raises(capture: dict[str, Any]) -> None:
    capture["response"] = _api_response(
        {"expressible": True, "regex": "(edit {5,}", "paraphrase": "x", "reason": None}
    )
    with pytest.raises(AskError, match="does not compile"):
        compile_query("edit streak?")


def test_refusal_raises(capture: dict[str, Any]) -> None:
    capture["response"] = _api_response({}, stop_reason="refusal")
    with pytest.raises(AskError, match="declined"):
        compile_query("anything")


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AskError, match="ANTHROPIC_API_KEY"):
        compile_query("anything")


def test_prompt_carries_alphabet_contract_and_limits(capture: dict[str, Any]) -> None:
    capture["response"] = _api_response(
        {"expressible": True, "regex": "edit", "paraphrase": "x", "reason": None}
    )
    compile_query("q")
    system = capture["payload"]["system"]
    for atom in ("edit", "run_test", "prompt_ai"):
        assert atom in system
    assert "trailing" in system
    assert "variable binding" in system
    for _, gold_regex in ask.GOLD_PAIRS:
        assert gold_regex in system


def test_request_shape_is_schema_constrained(capture: dict[str, Any]) -> None:
    capture["response"] = _api_response(
        {"expressible": True, "regex": "edit", "paraphrase": "x", "reason": None}
    )
    compile_query("q", model="claude-opus-5")
    payload = capture["payload"]
    assert payload["model"] == "claude-opus-5"
    fmt = payload["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert capture["api_key"] == "test-key"


def test_gold_pairs_all_compile() -> None:
    import re

    for _, gold_regex in ask.GOLD_PAIRS:
        re.compile(gold_regex)
