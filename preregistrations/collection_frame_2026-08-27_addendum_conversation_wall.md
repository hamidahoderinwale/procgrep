# Addendum — the 65,536-token conversation wall, and three cells with it lifted (2026-08-27)

Amends `collection_frame_2026-08-25.md` (f578fc9). Registered before any lifted-wall run exists.

**Finding that motivates it.** r2e-gym's agent compares a litellm token count of the whole conversation history (thinking included) with the constant `MAX_CONTEXT_TOKENS = 65536` before every query and raises when it is exceeded; the runner records the ending as `llm_query_error`. In the DeepSWE timeout rerun (gate 2026-08-26, timeout lifted to 7200 s) all 5 such endings were this wall; in R1 (Qwen3-32B, 100-step limit) all 3 were, at steps 44–74. The wall sits below the 100-step wall R1's decision rule reads, so R1 as registered is confounded. R1 is paused at 11 runs (ledger `r1_cap100/run2/runs/`), which are not reused.

**Cells.** `MAX_CONTEXT_TOKENS` patched 65536 → 400000; vLLM `--max-model-len 40960` unchanged, so the model's real context is the only remaining wall and vLLM's own refusal becomes the recorded error. Everything else per the frame (rig seed-42 order, 10-rollout gate, model-card decoding).

- `r1_cap100_ctx` — Qwen3-32B, 100-step limit, 100 runs from index 0. Decision rule unchanged from R1: finishes per 100 runs at steps 37–40 below 2 (from 8.9) and a rise in the last five steps before 100 → wall-triggered; spike stays at 37–40 → habit.
- `base_ctx` — Qwen3-32B, 40-step limit, 100 runs. `deepswe_ctx` — DeepSWE-Preview, 40-step limit, temperature 1.0, 100 runs, timeout patch kept.
  Decision rule for the pair: with the wall lifted, DeepSWE's own hand-in share rises above its 36% and its last-tenth share stays below 14% while the base's stays above 25% → the stopping contrast stands without the wall. DeepSWE's last-tenth share rises above 20% → the earlier contrast was manufactured by the wall ending its late runs, and the thesis's mechanism claim is withdrawn for this pair.

**Gate.** As the frame, with the 2026-08-26 error-rule waiver; vLLM context refusals count as errors.

**Cost.** ≈ $50–150 per cell on GCP Batch (taste-research), each attempt resuming from the ledger.

hamidah asked "continue given the timeout?" on 2026-08-26; this addendum is the proposed continuation and awaits a go naming the cells.
