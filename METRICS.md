# Procedural metrics

The numbers `procgrep` can put on a corpus of agent trajectories,
named as dimensions with formal definitions, units, ranges, and the
public-API call that computes each. This file is the reference for
the dimensional system the library exports — papers building on
`procgrep` should cite specific metrics by name.

This taxonomy is **subject to refinement.** Several of these
metrics are likely correlated on real corpora; the independent
set should be determined per-corpus via
[`examples/python/08_metric_orthogonality.py`](examples/python/08_metric_orthogonality.py).
A surviving independent set on the 84-agent SWE-bench corpus is the
empirical anchor; until that lands, treat the eight metrics below
as candidates, not final.

## Six procgrep-side metrics

These are computable from the procgrep public API alone — no edit
patches, no chain-of-thought, no ground-truth labels required.

### 1. Mean trajectory length

What it measures. How many canonical atoms a group's agents emit per
trajectory on average. A coarse proxy for procedural verbosity.

* Definition: `mean(len(t.atoms) for t in group)`.
* Units: atoms per trajectory.
* Range: `[1, inf)`. Higher means longer procedures.
* Computation: trivial; `len()` on `Trace.atoms`.
* What high/low mean. High = agents take many steps before
  submitting (could be careful exploration, could be
  thrashing). Low = agents act decisively (could be confident,
  could be lazy).

### 2. Effective vocabulary size

What it measures. How many distinct procedures a group "uses" in
practice, weighted by how often. Equivalent to the perplexity of
the group's mean procedure distribution.

* Definition: `exp(H(group-mean procedure distribution))` in nats.
* Units: dimensionless (effective count of equally-used procedures).
* Range: `[1, V]` where `V` is the BPE vocabulary size.
* Computation: `procgrep.effective_vocab_size_per_group(fps, group_by="group")`.
* What high/low mean. High = the group's procedural behavior
  spans many distinct procedures uniformly (rich repertoire). Low =
  the group concentrates on a few signature procedures.

### 3. Mean per-trajectory entropy

What it measures. How spread-out each individual trajectory's
procedure distribution is, averaged across the group's trajectories.

* Definition: `median(Fingerprint.entropy() for fp in group)` in nats.
* Units: nats.
* Range: `[0, log(V)]`.
* Computation: `procgrep.entropies_per_group(fps, group_by="group")` returns the median, IQR, and range.
* What high/low mean. High = the typical trajectory in the group
  uses many procedures roughly equally. Low = the typical trajectory
  is dominated by one or two procedures.

### 4. Within-group mean pairwise JSD

What it measures. How procedurally consistent the group is with
itself. Treated as the **noise floor** for any across-group
comparison.

* Definition: `mean(jsd(a.distribution(), b.distribution()) for a,b in pairs(group_fingerprints))`.
* Units: bits (log-base-2 JSD) ranging in `[0, 1]`.
* Range: `[0, 1]`.
* Computation: compose `procgrep.jsd` over fingerprint pairs;
  see [`examples/python/02_controlled_eval.py`](examples/python/02_controlled_eval.py)
  for the recipe.
* What high/low mean. High = trajectories inside the group differ
  procedurally even from each other (procedure is not well-defined
  for this group). Low = the group has a tight procedural
  identity. Any across-group JSD claim must be evaluated against
  this floor.

### 5. Procedure concentration (HHI)

What it measures. How concentrated a group's procedure distribution is
on a few dominant procedures. Borrows the Herfindahl-Hirschman index
from economics.

* Definition: `sum(p_i^2 for p_i in group_mean_distribution)`.
* Units: dimensionless probability-squared.
* Range: `[1/V, 1]`. Theoretical floor `1/V` at perfectly uniform
  distribution; ceiling 1 when all mass is on one procedure.
* Computation: not a single function call; ~3 lines composing
  `Fingerprint.distribution()` and numpy. See
  [`examples/python/08_metric_orthogonality.py`](examples/python/08_metric_orthogonality.py).
* What high/low mean. High = a handful of procedures account for most
  of the group's procedural mass (specialist). Low = procedural
  mass is spread across many procedures (generalist).

### 6. Atom-frequency Gini

What it measures. How unequally raw atoms (not procedures) are
distributed within a group. Independent of the BPE vocabulary.

* Definition: Gini coefficient on the group's per-atom count
  vector.
* Units: dimensionless.
* Range: `[0, 1]`. Zero = every atom equally used; one = a single
  atom dominates.
* Computation: standard Gini formula on `Counter(t.atoms for t in group)`.
* What high/low mean. High = the group relies heavily on a small
  number of action types (edit-heavy or search-heavy). Low = the
  group uses the full action repertoire roughly evenly.

## Two companion-paper metrics

These require artifacts the `bidirect-align-dev-traces` companion
repository computes (AST patches, structured edit-certificate
data); they are listed here as part of the dimensional system that
the joint paper proposes, with pointers to where each is computed.

### 7. Edit-certificate Jaccard

What it measures. Pairwise structural similarity between two
agents' patches on the same task. Distinct from JSD: operates on
the *patch* (output), not on the *trajectory* (procedure).

* Definition: Jaccard similarity over the set of `(direction,
  AST-node-type)` pairs extracted from each agent's patch.
* Units: dimensionless.
* Range: `[0, 1]`. Zero = no overlap; one = identical edit shape.
* Computation: `bidirect-align-dev-traces/analysis/` (scoped
  certificates module). Not yet exposed as a Python public-API
  function; packaging is a known gap.
* What high/low mean. High = two agents converged on structurally
  similar fixes. Low = different structural approaches to the
  same problem.

### 8. Composition-failure rate

What it measures. The fraction of agent failures on a task class
where the agent demonstrated every required primitive on *other*
tasks but failed to combine them here. Distinguishes "lacks
primitive" failures from "lacks composition" failures.

* Definition: see `bidirect-align-dev-traces/scripts/compositional_generalization.py`.
* Units: dimensionless ratio.
* Range: `[0, 1]`.
* Computation: companion repo only; not in procgrep.
* What high/low mean. High = failures are mostly compositional
  (agent has the parts but cannot assemble them). Low = failures
  are mostly primitive-missing (agent never learned the
  operations needed).

## How to use this taxonomy

1. **Pick a corpus.** Each dimension is defined per-group; a
   corpus partitioned into ≥10 groups gives the orthogonality
   analysis enough degrees of freedom to be meaningful.
2. **Compute the procgrep-side six.** Use
   [`examples/python/08_metric_orthogonality.py`](examples/python/08_metric_orthogonality.py)
   to compute all six metrics and the pairwise correlation
   between them.
3. **Drop redundant pairs.** The script suggests an independent
   subset using a default threshold of `|r| = 0.85`; pick a
   threshold that matches your paper's tolerance.
4. **Report the surviving set as your dimensional system** for
   that corpus, with the correlation matrix in an appendix or
   supplement.
5. **Augment with companion-paper metrics** (#7, #8) when patch
   data is available; the joint system is then the orthogonal
   subset of #1-#8.

## Status

This file is a starting point. The intended end-state is that
the orthogonal subset on the 84-agent SWE-bench corpus is the
"canonical" dimensional system the paper proposes — but that
analysis hasn't been run on the real corpus yet. Until it has,
treat this taxonomy as a hypothesis being refined.
