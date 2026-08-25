# Controlled base-cell collections — the sampling frame

Registered 2026-08-25, while the second-family pair and the cell-B rerun are being set up
and before any of their data is read. Purpose: the collections are systematic samples from a
stated frame, not pilots, so their results enter the reportable narrative on the same
footing as the public corpora.

## Which models and pairs are collected — the selection rule

Frame = the hub-lineage census (plateau/pair_inventory/hub_census, pipeline reruns
end-to-end; 646 open code fine-tune pairs). A pair enters collection when ALL hold:
1. The fine-tune's card carries an explicit `base_model` relation of type fine-tune to an
   open-weight base (no inferred lineage, no instruct siblings standing in for a base).
2. At least one side has public per-step trajectories on SWE-bench Verified, or both sides
   will be collected here under one rig.
3. Both models fit the collection rig (≤32B parameters on 4×L4 at 32k context).
Ordering within the frame: by the number of public derivatives a base unlocks, then by
base family not yet represented. Under this rule the cells are: Qwen2.5-Coder-32B
(unlocks SWE-agent-LM; rerun with function calling after text mode failed the harness),
then the Mistral-Small-3.1-24B-Base-2503 ↔ Devstral-Small-2-24B-Instruct-2512 pair (first
non-Qwen family with a documented relation). The earlier Mistral *Instruct* cell violated
rule 1 and is reported as a descriptive instruct-model profile, not as a base.

## Which tasks — the instance rule

All 500 SWE-bench Verified instances in one fixed seed-0 permutation shared by every cell.
A cell killed by its 12h cap after k instances therefore holds the first k of a random
permutation — a simple random subsample of Verified, with k stated on every number — and
all cells overlap on the same k-prefix.

## Gates before a fleet, and how partials report

Behavioral smoke gate on ≥10 rollouts: ≥3 voluntary finishes and <20% LLM query errors,
else the cell stops and the failure is the reported result (cell B, text mode, is the
precedent). Rig: r2e-gym, documented step budget, model-card decoding defaults, all recorded
in the cell manifest with revisions. Partials are never topped up silently; a top-up run is
its own recorded cell.
