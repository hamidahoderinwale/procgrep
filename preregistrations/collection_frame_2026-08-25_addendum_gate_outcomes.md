# Collection frame — addendum: gate outcomes and one erratum (2026-08-25, post-data)

Records what happened when the frame at f578fc9 was applied. The frame text is not edited.

## Erratum — the shared permutation is seed 42, not seed 0

The r2e-gym driver (`runagent_multiple`) applies its own `ds.shuffle(seed=42)` to SWE-bench
Verified before any cell sees an instance. Every r2e cell collected so far (Mistral instruct
216, cell B text 53, cell B function-calling 10, family-2 smokes) therefore already shares
the seed-42 order; no cell used a seed-0 order. The realized order is published at
`gs://hamidah-procgrep-ladder-20260821/frame/verified_r2e_seed42_order.json`. The frame's
instance rule holds with "seed 42" in place of "seed 0"; nothing about k-prefix reporting changes.

## Gate outcomes — three cells stopped, none rolled

| cell | model | mode | rollouts | voluntary | resolved | LLM errors | verdict |
|---|---|---|---|---|---|---|---|
| cell B (pair 2 base) | Qwen2.5-Coder-32B-Instruct | function calling | 10 | 0 | 0 | 0 | stop — hallucinated tool names (print, python, exit_process) in place of finish; 10/10 step-cap exits |
| family 2, base | Mistral-Small-3.1-24B-Base-2503 | text and function calling | 0 completed | — | — | — | stop — no end-of-turn; requests generate to the 32k limit, the agent loop never advances |
| family 2, fine-tune | Devstral-Small-2-24B-Instruct-2512 | text | 10 | 0 | 2 (at the cap) | 2 (20%) | stop under the frame as written — operates the rig, never stops voluntarily |
| family 2, fine-tune | Devstral-Small-2-24B-Instruct-2512 | function calling (mistral parser) | 10 | 1 | 0 | — | not a fair mode — emits a `think` tool the harness rejects on 344/452 completions |

Cost of the three stops: ≈ 2.6 VM-hours ≈ $17. Cell B text mode (53 rollouts, 0 voluntary,
38% errors) was the precedent and stands.

## Consequences (recorded, not acted on)

1. Pair 2's base side is not collectible under r2e-gym in either interface mode. A cell on the
   scaffold the model was trained to drive (SWE-agent, which also produced the public student
   trajectories) would be same-rig with the public side, but is a new cell needing its own
   registration before any rollout.
2. A pretrained base does not operate an agent rig. Same-rig base↔fine-tune pairs are
   therefore collectible only when the "base" is an instruct model with documented lineage to
   the derivative (the Qwen pairs), or through the teacher-pair design. This bounds the
   program's scaling route and belongs in the paper's scope sentence.
3. The gate criterion "≥3 voluntary finishes" conflates two failures: cannot drive the rig
   (cell B, Base-2503) and drives it but never stops voluntarily (Devstral: 2/10 resolved at
   the cap with coherent work; the same non-finisher profile as the Mistral instruct cell and
   KTH's Devstral-123B). A refined gate — LLM errors <20% AND (≥3 voluntary finishes OR ≥1
   resolved with coherent edits) — would pass Devstral and still stop the true failures. It is
   a candidate only; if adopted it is registered as a new frame before the next collection,
   and Devstral alone (no collectible base) buys a family stopping profile, not a pair.
