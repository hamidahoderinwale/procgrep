"""Evaluate the NL->regex compiler against the gold pairs it ships with.

Scores by match-set equivalence over the bundled synthetic corpus, not
string equality: two different regexes that select the same traces count
as agreeing. Needs ANTHROPIC_API_KEY; costs a few cents (one small call
per gold question).

Run:
    python examples/python/17_ask_gold_eval.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from procgrep import canonicalize
from procgrep.ask import GOLD_PAIRS, AskError, compile_query
from procgrep.io import read_jsonl

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_traces.jsonl"


def match_set(pattern: str, spines: list[str]) -> frozenset[int]:
    rx = re.compile(pattern)
    return frozenset(i for i, s in enumerate(spines) if rx.search(s))


def main() -> int:
    traces = canonicalize(list(read_jsonl(DATA)), adapter="swe-agent")
    spines = [" ".join(t.atoms) + " " for t in traces]

    agree = 0
    for question, gold in GOLD_PAIRS:
        try:
            out = compile_query(question)
        except AskError as exc:
            print(f"ERROR  {question!r}: {exc}")
            continue
        if not out.expressible or out.regex is None:
            print(f"REFUSED  {question!r}: {out.reason}")
            continue
        same = match_set(out.regex, spines) == match_set(gold, spines)
        agree += same
        flag = "ok " if same else "DIFF"
        print(f"{flag}  {question!r}\n      model {out.regex}\n      gold  {gold}")

    print(f"\nmatch-set agreement: {agree}/{len(GOLD_PAIRS)}")
    return 0 if agree == len(GOLD_PAIRS) else 1


if __name__ == "__main__":
    sys.exit(main())
