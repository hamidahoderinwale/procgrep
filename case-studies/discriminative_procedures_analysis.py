"""Run discriminative_procedures on the key comparative pairs.

Three comparisons:
  1. Claude-4 Sonnet vs old-cluster agents (Claude-3 Opus + GPT-4 + GPT-4o)
  2. SWE-agent-LM-32B vs old-cluster agents
  3. SWE-agent-LM-32B vs Claude-4 Sonnet  (mutual nearest neighbours)
  4. Pass vs fail on SWE-agent-LM-32B (resolve labels available)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_PROCGREP_SRC = Path(__file__).resolve().parent.parent / "procgrep" / "src"
if _PROCGREP_SRC.exists() and str(_PROCGREP_SRC) not in sys.path:
    sys.path.insert(0, str(_PROCGREP_SRC))

from procgrep.bpe import fit_bpe
from procgrep.encode import encode
from procgrep.stats import discriminative_procedures
from procgrep.types import PROCEDURE_SEPARATOR, Trace

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT_DIR = RESULTS / "discriminative_procedures_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = [
    ("Claude-3 Opus", "fingerprints_claude3opus_n500.jsonl"),
    ("Claude-3.5 Sonnet", "fingerprints_claude3.5sonnet_n500.jsonl"),
    ("Claude-4 Sonnet", "fingerprints_claude4sonnet_n500.jsonl"),
    ("SWE-agent-LM-32B", "fingerprints_child_n500.jsonl"),
    ("GPT-4", "fingerprints_gpt4_n500.jsonl"),
    ("GPT-4o", "fingerprints_gpt4o_n500.jsonl"),
]

OLD_CLUSTER = {"Claude-3 Opus", "Claude-3.5 Sonnet", "GPT-4", "GPT-4o"}


def load_traces(layer: str) -> list[Trace]:
    traces: list[Trace] = []
    per_agent: dict[str, list[Trace]] = defaultdict(list)
    for name, fname in AGENTS:
        path = RESULTS / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found, skipping {name}")
            continue
        for line in path.read_text().splitlines():
            d = json.loads(line)
            atoms = d.get(f"atoms_{layer}", [])
            if atoms:
                per_agent[name].append(
                    Trace(
                        trace_id=d["instance_id"],
                        agent=name,
                        atoms=atoms,
                        metadata={"resolved": d.get("resolved")},
                    )
                )
    for name, ts in per_agent.items():
        traces.extend(ts)
        print(f"  loaded {len(ts):>4d} traces for {name}")
    return traces


def fmt_proc(procedure: str) -> str:
    """Make a procedure readable: edit▁run_test▁edit → [edit → run_test → edit]."""
    parts = [p.strip() for p in procedure.split(PROCEDURE_SEPARATOR) if p.strip()]
    if len(parts) == 1:
        return parts[0]
    return " → ".join(parts)


def run_comparison(
    fps,
    vocab,
    label_a: str,
    label_b: str,
    traces: list[Trace],
    k: int = 8,
    ranking: str = "log_odds",
) -> dict:
    """Return structured dict with top procedures for both sides."""
    top_a = discriminative_procedures(
        fps, vocab, group_a=label_a, group_b=label_b, k=k, ranking=ranking
    )
    top_b = discriminative_procedures(
        fps, vocab, group_a=label_b, group_b=label_a, k=k, ranking=ranking
    )

    print(f"\n  TOP PROCEDURES IN '{label_a}':")
    for m in top_a:
        if m.log_odds > 0:
            print(
                f"    {m.log_odds:+.3f}  p_a={m.p_a:.3f}  p_b={m.p_b:.3f}  [{fmt_proc(m.procedure)}]"
            )
    print(f"\n  TOP PROCEDURES IN '{label_b}':")
    for m in top_b:
        if m.log_odds > 0:
            print(
                f"    {m.log_odds:+.3f}  p_a={m.p_a:.3f}  p_b={m.p_b:.3f}  [{fmt_proc(m.procedure)}]"
            )

    return {
        label_a: [
            {
                "procedure": fmt_proc(m.procedure),
                "log_odds": round(m.log_odds, 4),
                "p_a": round(m.p_a, 4),
                "p_b": round(m.p_b, 4),
            }
            for m in top_a
            if m.log_odds > 0
        ],
        label_b: [
            {
                "procedure": fmt_proc(m.procedure),
                "log_odds": round(m.log_odds, 4),
                "p_a": round(m.p_a, 4),
                "p_b": round(m.p_b, 4),
            }
            for m in top_b
            if m.log_odds > 0
        ],
    }


def main() -> None:
    all_results: dict = {}

    for layer, vocab_size in [("canonical", 128), ("native", 300)]:
        print(f"\n{'='*80}")
        print(f"DISCRIMINATIVE PROCEDURES @ {layer.upper()} (BPE V={vocab_size}, n500)")
        print(f"{'='*80}")

        traces = load_traces(layer)
        print(f"\n  Fitting BPE on {len(traces)} total trajectories...")
        vocab = fit_bpe([t.atoms for t in traces], vocab_size=vocab_size, seed=0)
        print(f"  Vocabulary: {len(vocab.atoms)} atoms + {len(vocab.merges)} merges")
        fps = encode(traces, vocab=vocab)

        # Save the fitted vocabulary
        from procgrep import save_vocab

        save_vocab(vocab, RESULTS / f"procedure_vocab_{layer}_n500.json")

        # Merge old-cluster agents for cleaner comparisons
        merged_traces: list[Trace] = []
        for t in traces:
            label = "old-cluster" if t.agent in OLD_CLUSTER else t.agent
            merged_traces.append(Trace(trace_id=t.trace_id, agent=label, atoms=t.atoms))
        fps_merged = encode(merged_traces, vocab=vocab)

        layer_results: dict = {}

        print("\n--- 1. Claude-4 Sonnet vs 2024-era models ---")
        layer_results["claude4_vs_old"] = run_comparison(
            fps_merged, vocab, "Claude-4 Sonnet", "old-cluster", merged_traces
        )

        print("\n--- 2. SWE-agent-LM-32B vs 2024-era models ---")
        layer_results["swe_lm_vs_old"] = run_comparison(
            fps_merged, vocab, "SWE-agent-LM-32B", "old-cluster", merged_traces
        )

        print("\n--- 3. SWE-agent-LM-32B vs Claude-4 Sonnet ---")
        layer_results["swe_lm_vs_claude4"] = run_comparison(
            fps, vocab, "SWE-agent-LM-32B", "Claude-4 Sonnet", traces
        )

        # Pass vs fail (labeled trajectories only)
        pass_traces = [
            t
            for t in traces
            if t.agent == "SWE-agent-LM-32B" and t.metadata.get("resolved") is True
        ]
        fail_traces = [
            t
            for t in traces
            if t.agent == "SWE-agent-LM-32B" and t.metadata.get("resolved") is False
        ]
        if pass_traces and fail_traces:
            print(f"\n--- 4. Pass (n={len(pass_traces)}) vs Fail (n={len(fail_traces)}) ---")
            pf_traces = [
                Trace(trace_id=t.trace_id, agent="pass", atoms=t.atoms) for t in pass_traces
            ] + [Trace(trace_id=t.trace_id, agent="fail", atoms=t.atoms) for t in fail_traces]
            fps_pf = encode(pf_traces, vocab=vocab)
            layer_results["pass_vs_fail"] = run_comparison(fps_pf, vocab, "pass", "fail", pf_traces)

        all_results[layer] = layer_results

        # Also save human-readable text
        out_txt = OUT_DIR / f"top_procedures_{layer}_n500.txt"
        lines = []
        for comp, sides in layer_results.items():
            lines.append(f"\n=== {comp} ===")
            for agent, procs in sides.items():
                lines.append(f"\n  {agent}:")
                for p in procs[:8]:
                    lines.append(f"    {p['log_odds']:+.3f}  [{p['procedure']}]")
        out_txt.write_text("\n".join(lines))
        print(f"\n  Saved: {out_txt.name}")

    # Save structured JSON for figures
    out_json = RESULTS / "discriminative_procedures_n500.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nStructured results: {out_json.name}")


if __name__ == "__main__":
    main()
