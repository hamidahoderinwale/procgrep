# Stage 2 recollection — opener block, preregistered readouts

Registered 2026-08-21, before collection. Purpose: the battery's manipulation-check row —
a harness intervention with real bite, read with a position-aware statistic committed in
advance. Supersedes the 2026-08-14/15 attempt, whose search-block bit on only 15/50 rollouts
(the model's modal opener is `ls`, not search) and whose read-blocked arm was lost to the VM
backstop before retrieval.

## Collection

- Project `taste-playground`, 2 VMs (g2-standard-8 + L4, `--max-run-duration=12h` as the hard
  spend guard), Qwen3-8B via vLLM, mini-swe-agent text mode, temperature 0.4, step limit 40 —
  identical to the 08-15 configuration and the same 50 picked instances
  (`plateau/gcp_stage2/picked_instances.json`).
- Arm A `opener_blocked`: block the read-head opener family (`ls`, `cat`, `head`, `find . -maxdepth`
  style listings) using the stage-2 block mechanism, so the modal opening move is unavailable.
- Arm B `control_run2`: untouched rerun — drift control against 08-15's `control_run1` and the
  second half of the floor construction.
- Per-run results push to the collection bucket as each rollout finishes; nothing waits for
  the 12h boundary.

## Preregistered bets, with their basis

1. **Bite ≥ 50% of arm rollouts** fire the block. Basis: `ls` opened 29/50 (58%) of
   control_run1 rollouts (`gcp_stage2/analysis/manipulation_check.json`).
2. **Primary readout: per-atom positional-W1 max vs the joint Monte Carlo max-null**
   (2000 matched control splits, seed 0, cell inclusion ≥8 arm / ≥16 control occurrences —
   the `plateau/positional_w1/positional_w1.py` construction reused verbatim), arm vs
   control_run2. Bet: p(null max ≥ observed) < 0.05. Basis: the analogous
   consequence-structure intervention (A2 truncation) cleared this exact construction at
   p = 0.015 with only 25 affected actions, and an opener block displaces the first executed
   action by construction. Explicitly NOT the pooled-frequency-weighted mean, which diluted
   A2 to p = 0.063.
3. **Contrast, no bet:** whole-trajectory BPE-64 mean-fingerprint ratio-to-floor reported
   alongside — July/August evidence says means under-read mechanical interventions; this row
   documents it a fourth time or surprises us.
4. **Null control:** control_run2 vs control_run1 on both readouts sits within floor
   (temporal-drift check). If it does not, the arm contrast is read against run2 only and the
   drift is reported as a finding.
5. **Outcome axis, no bet:** resolve-rate delta with a paired interval, reported; n = 50 is
   underpowered for outcomes and the behavioral axis is primary.

## Instrument spec

Canonicalizer = `gcp_stage2/analysis/manipulation_check.py` classifier (v2 extensions),
coverage gate 85% on executed actions. Fingerprint vocabulary BPE-64 fit jointly over the
compared corpora, spec hash stamped in every output. All analysis outputs land under
`plateau/gcp_stage2/recollection_20260821/`.
