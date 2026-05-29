"""Cross-language attribution: does an agent's procedural fingerprint transfer?

The strongest form of the attribution baseline:

    If we train a classifier on (agent_label) using fingerprints
    from a subset of languages, can it still identify the agent
    when shown fingerprints from a *held-out* language?

This is what gumtree atoms uniquely let us ask. Other agent traces
(SWE-agent, Agentless, ...) bake one language's tool surface into
the action layer; gumtree atoms are AST-level and language-neutral
at the *operation* layer (insert / delete / update / move),
though they remain language-specific at the *node-type* layer
(``Name`` in Python vs ``Identifier`` in JS vs ``SimpleName`` in
Java). High cross-language attribution accuracy therefore implies
the agent's procedure has a structural signature visible above the
language-specific node-type noise.

Setup: LOGO probe with ``label_field="agent"``, ``group="language"``.
Three folds: hold out Python, JavaScript, Java in turn.

Falsifier: held-out accuracy at chance (1/N_agents). Either the
agents are too similar at the AST level, or the language-specific
node-type vocabulary swamps the agent signal.

Run from the repository root:

    python examples/python/13_cross_language_attribution.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from procgrep import (
    canonicalize,
    encode,
    fit_bpe,
    leave_one_group_out,
)
from procgrep.io import read_jsonl
from procgrep.stats import discriminative_procedures

warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = ROOT / "examples" / "synthetic_gumtree_traces.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--traces",
        type=Path,
        default=DEFAULT_TRACES,
        help="JSONL trace file (default: bundled gumtree fixture)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50,
        help="BPE target vocabulary size (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for the LOGO classifier (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = list(read_jsonl(args.traces))
    traces = canonicalize(raw, adapter="gumtree")

    if any(t.group is None for t in traces):
        raise SystemExit("this analysis requires every trace to carry a `group` label (language).")

    agents = sorted({t.agent for t in traces})
    languages = sorted({t.group or "" for t in traces})
    chance = 1.0 / len(agents)
    print(f"loaded {len(traces)} traces; {len(agents)} agents x {len(languages)} languages")
    print(f"  agents:    {agents}")
    print(f"  languages: {languages}")
    print(f"  chance accuracy: 1/{len(agents)} = {chance:.2f}")

    # Fit one shared BPE vocabulary across all languages — refitting per
    # language would make the vocabulary itself language-specific and
    # confound the comparison.
    vocab = fit_bpe((t.atoms for t in traces), vocab_size=args.vocab_size, seed=args.seed)
    fps = encode(traces, vocab=vocab)

    result = leave_one_group_out(fps, label_field="agent", seed=args.seed)

    print(
        "\nleave-one-language-out attribution accuracy "
        "(predict agent label when language is held out):"
    )
    print(f"  overall: {result.overall_accuracy:.2f}")
    for lang in languages:
        if lang in result.per_group_accuracy:
            acc = result.per_group_accuracy[lang]
            verdict = "  (above chance)" if acc > chance else ""
            print(f"  held-out {lang:>10s}: {acc:.2f}{verdict}")

    print("\nconfusion (true language -> predicted agent counts):")
    for true_lang in sorted(result.confusion):
        bucket = result.confusion[true_lang]
        rendered = ", ".join(f"{a}={c}" for a, c in sorted(bucket.items()))
        print(f"  {true_lang:>10s}: {rendered}")

    # Surface the procedures that most distinguish the two agents -- these are
    # the structural signatures that need to be language-invariant for the
    # cross-language story to hold up.
    if len(agents) == 2:
        a, b = agents
        ranked = discriminative_procedures(
            fps, vocab, group_a=a, group_b=b, k=10, ranking="log_odds", group_by="agent"
        )
        if ranked:
            print(f"\ntop discriminative procedures ({a} vs {b}, log-odds-ranked):")
            for m in ranked:
                print(
                    f"  {m.procedure:50s}  log_odds={m.log_odds:+.3f}  p_a={m.p_a:.3f}  p_b={m.p_b:.3f}"
                )


if __name__ == "__main__":
    main()
