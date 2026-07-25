"""Tests for the prompt/completion chat-format adapter."""

from __future__ import annotations

import json

from procgrep.ingest.adapters.prompt_completion import prompt_completion_adapter
from procgrep.ingest.core import DatasetSchema, sniff

# shaped like a PrimeIntellect/INTELLECT-3-SFT toucan_tool row
TOUCAN_ROW = {
    "source": "toucan",
    "tools": "[]",
    "prompt": [{"role": "user", "content": "Fetch the ETH price and make a round."}],
    "completion": [
        {
            "role": "assistant",
            "content": "I'll fetch the price first.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "",
                    "function": {"name": "coin-price-getTokenPrice", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "4043.72", "tool_calls": None},
        {
            "role": "assistant",
            "content": "Creating the round.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "",
                    "function": {"name": "game-create_round", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "Error: round limit reached", "tool_calls": None},
        {"role": "assistant", "content": "Done!", "tool_calls": None},
    ],
}


def test_toucan_row_parses_to_expected_atoms() -> None:
    atoms = prompt_completion_adapter(TOUCAN_ROW)
    assert atoms == [
        "prompt_ai",  # user setup
        "think",  # assistant text
        "read_file",  # getTokenPrice: get verb
        "think",
        "create_file",  # create_round: create verb
        "error",  # failing tool result
        "think",
    ]


def test_json_string_turns_decode() -> None:
    row = {
        "prompt": json.dumps(TOUCAN_ROW["prompt"]),
        "completion": json.dumps(TOUCAN_ROW["completion"]),
    }
    assert prompt_completion_adapter(row) == prompt_completion_adapter(TOUCAN_ROW)


def test_unknown_tool_and_missing_fields_degrade_gracefully() -> None:
    row = {
        "completion": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "frobnicate"}}],
            },
            {"role": "tool", "content": "ok"},
            "not-a-mapping",
        ]
    }
    assert prompt_completion_adapter(row) == ["other"]
    assert prompt_completion_adapter({}) == []


def test_sniffer_prefers_prompt_completion_over_openhands_fallback() -> None:
    schema = DatasetSchema(
        dataset="unit",
        config="default",
        split="train",
        columns=("source", "prompt", "completion", "tools"),
        sample_rows=(TOUCAN_ROW,),
    )
    ranked = sniff(schema)
    assert ranked[0].adapter == "prompt-completion"
    assert ranked[0].confidence == 0.9
