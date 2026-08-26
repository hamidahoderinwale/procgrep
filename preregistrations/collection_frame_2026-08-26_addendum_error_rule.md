# Collection frame — addendum 2: the query-error rule is a rig check, not a behavior check (2026-08-26)

Registered before the R1 rerun; the frame at f578fc9 and addendum 328cd5c are not edited.

## What happened

R1 (Qwen3-32B, no fine-tuning, r2e-gym, 100-step limit) stopped at the gate on 2026-08-26: 7 of 10
rollouts handed in on their own (the ≥3 line, clearly passed) and 3 of 10 ended in `llm_query_error`
(30%, over the <20% line). The three errored runs had gone 23, 56 and 68 steps with prompts of
11.6k–29.7k tokens (inside the 40,960 window); one generation step took 570 s, at the harness's
query timeout; one of the three had already produced a passing patch. The rule fired on a
model × harness timing interaction under a longer limit, not on inability to drive the rig.

## Amendment

The `<20% LLM query errors` criterion exists to catch a model that cannot operate the harness
(the cell B and Base-2503 cases: 0 voluntary finishes). It is therefore waived when ≥5 of 10 gate
rollouts hand in on their own. Query-error runs remain in the data, are reported in every
readout's count, and are excluded from timing statistics exactly as Fig 3's error-excluded
companion does. The voluntary-finish criterion is unchanged.

## Applies to

R1's rerun (`r1_cap100/run2/`; the 10 gate rollouts of the first attempt are kept as
`r1_cap100/runs/r1_cap100/` and count toward the sample) and any later cell. Decoding, order,
limits and decision rule for R1 are unchanged from its manifest.
