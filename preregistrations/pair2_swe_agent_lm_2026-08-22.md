# Pair 2 — Qwen2.5-Coder-32B base versus its SFT distill, preregistered bets

Registered 2026-08-22, while the base side (cell B, r2e rig, 500 SWE-bench Verified × 1
run) is still collecting and before any joint analysis. Fine-tune side: the public
SWE-agent-LM-32B r2e-gym trajectories on the same 500 Verified instances (revision
5744cd5a, attempts 1+2). Same harness family, same tasks — the program's second
same-harness base↔derivative pair. Constructions are the recorded ones: BPE-64 joint fits
with stamped specs, matched Monte Carlo floors and nulls (2000, seed 0), exceedance primary.

## Bets, with their basis

1. **Bases panic; distills do not.** The base's voluntary-finish deadline share (last 10%
   of its budget) is at least 3× the student's (student: 2.2% [1.2, 3.3] recorded), and its
   resolve-by-timing gradient falls (early > deadline). Basis: Qwen3-32B read 27.7% under
   the same rig family and every measured derivative lacks the wall response; this is the
   first test of the pattern on a second base.
2. **Difficulty preservation crosses the SFT pair:** per-instance difficulty correlation
   between base and student ≥ 0.6 (point-biserial, 1-run base × 2-attempt student rate;
   instance bootstrap). Basis: anchor pair r = 0.906 at 10×10 runs; the dose analysis's
   1-run cells read ~0.73 against a 0.898 split-half ceiling, so 0.6 allows the expected
   single-run attenuation. A partial base collection scores on the covered instances,
   count reported.
3. **The representation ordering ports:** finish timing detects the pair at a sample size
   no larger than procedure fingerprints need (exceedance construction, n grid 5–100).
   Basis: anchor ordering (timing n≈50, procedures n≈100); this is the portability test the
   matrix called untestable until now.
4. **The sweep's two new families confirm** (screening → confirmation, per the exploratory
   framing): the student side shows HIGHER transition entropy and LOWER error-contact
   co-occurrence than the base, both clearing their matched floors at run-level inference.
   Basis: anchor screening d = +0.41 (entropy, 11× floor) and −0.28…−0.32 (error contact,
   12–15× floors).
5. **No bet, reported:** end-type mix contrast; procedure JSD as multiples of the joint
   floor; the base's absolute-step finish density (banked for a future two-cap reading).

## Falsification honesty

Bet 1 failing (a non-panicking base) would be the more interesting outcome — it would
demote "bases panic" from a base-model trait to a Qwen-family trait and sharpen the Mistral
cell's role. Bet 4 failing retires the sweep families to anchor-specific descriptives. A
partial cell-B landing (the 12h cap may bind near 300–350 rollouts) scores every bet on the
landed instances with the count stated; nothing is deferred for fullness.
