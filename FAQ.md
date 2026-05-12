# Frequently asked questions

## What does procgrep do, in one sentence?

It measures how LLM coding agents actually work, not just whether
they succeed, by canonicalizing their tool-call traces into a shared
alphabet and comparing them with Jensen-Shannon divergence.

## How is procgrep different from DSPy?

They sit on opposite sides of the language/action boundary.

- **DSPy** is a natural-language-layer instrument. It compiles and
  optimizes the prompts and demonstrations an agent receives.
- **procgrep** is a procedural-layer instrument. It characterizes
  the tool-call trajectory the agent produces after it reads its
  prompt.

procgrep does not depend on DSPy and does not call any LLM. The two
are useful together: a researcher can optimize an agent with DSPy
and use procgrep to measure whether the natural-language-layer
optimization produced a procedural-layer shift.

## Why count-based fingerprints and not embeddings?

Three reasons.

- **Interpretability.** Each fingerprint dimension is a named
  motif. Differences in JSD can be attributed to specific motifs;
  the stuck-edit-loop finding works because of this.
- **No model dependency.** Reproducible without loading a
  sentence-transformer; deterministic given a fixed vocabulary.
- **Faithful to the paper's framing.** The paper argues procedure
  has discrete structure; embedding the atoms with a continuous LM
  would smear that structure into a similarity space.

Embedding-based comparison is a legitimate alternative methodology
and a possible future capability, but it is not the choice the
paper commits to.

## Can procgrep run my agent?

No. procgrep is post-hoc analysis only. It reads trace files; it
does not execute agents or call models. You capture traces with
whatever harness you already have, then point procgrep at the
resulting JSONL.

## What is the canonicalization-layer caveat?

The procedural fingerprint reflects what the canonicalizer
captures, not the agent's full behavior. If a scaffold delegates
test execution to a planner whose actions do not surface as visible
tool calls (as we found with Moatless), the fingerprint will be
silent about those test executions. Reading "tests run zero times"
in a fingerprint does not mean "the agent tested zero times"; it
means "the captured atom stream did not include a test atom."

A direct audit of one Moatless trajectory found 71 internal
`RunTests` references inside the trace content, zero of which
surfaced as canonicalized atoms. Behavior at the canonicalization
layer is real and should be audited per-scaffold.

## How do I add a new scaffold?

Implement a `TraceAdapter`: a callable that takes one raw record
and returns an `AtomSequence`. Register it once.

```python
from procgrep.canonicalize import register_adapter
from procgrep.types import AtomSequence, ATOM_EDIT, ATOM_OTHER

def my_adapter(record):
    return [ATOM_EDIT if step.get("type") == "patch" else ATOM_OTHER
            for step in record.get("steps", [])]

register_adapter("my_scaffold", my_adapter)
```

See `examples/python/05_custom_adapter.py` for a complete worked
example.

## What is the difference between the Level 1 pattern matcher and the procedural-DSPy DSL?

The **Level 1 matcher** (current) takes regex patterns over
space-joined atom sequences. It can express things like "no run of
five consecutive edits" or "a localize atom must precede the first
edit atom." It is stateless and position-aware.

The **procedural-DSPy DSL** (future, v1.0) adds compositional
invariants with temporal operators ("eventually X follows Y"),
variable binding across atoms, bounded loops, and probabilistic
predicates ("edit-atom proportion is below 0.6"). Level 2 is what
the paper's `Implications` section points at as future work.

If your invariant is expressible in regex, the Level 1 matcher is
enough. If you need temporal or distributional reasoning, you are
waiting on procedural-DSPy.

## Why MIT?

procgrep is a research artifact intended for academic and commercial
use without friction. MIT is the standard permissive license for
this case. If you need a different license for compatibility
reasons, open an issue and we can discuss.

## How do I cite procgrep in a paper?

The repository ships `CITATION.cff`. GitHub's "Cite this repository"
button uses it. A BibTeX entry will land here on the first tagged
release. Until then, cite the software via its repository URL plus
the version tag.

The paper this library was extracted from is the primary citation
target; once that paper is published, both citations should appear.

## Will procgrep work on browser-agent or GUI-agent traces?

The atom alphabet shipped today is designed for coding-agent
tool-call traces. The canonicalization protocol is general: a
`TraceAdapter` can map any structured trace into an atom sequence.
The fingerprinting machinery (BPE, JSD, UMAP, probe) does not care
what the atoms represent.

That said, GUI-agent traces (mouse coordinates, screen state, OCR
text) are structurally different from tool-call traces and would
benefit from a different canonical alphabet than the one shipped
here. Validation against a non-coding domain is an open research
question rather than a procgrep configuration question.

## How is "fingerprint" used here?

Strictly: a fingerprint is a non-negative integer count vector over
a fixed BPE motif vocabulary, normalized to sum to one for use as a
probability distribution. It is not a hash, not an embedding, and
not a cryptographic identity. Multiple trajectories can share a
fingerprint, especially under aggressive BPE merges.
