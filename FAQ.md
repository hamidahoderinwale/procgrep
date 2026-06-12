# Frequently asked questions

## What does procgrep do, in one sentence?

It reads agent trace logs and tells you how agents differ in *how they work* — which actions they take, in what order, and whether that matches what successful agents do.

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

## Why count procedures rather than use embeddings?

Three reasons.

- **You can read the answer.** Each feature is a named action sequence. When two agents differ, you can see exactly which sequences account for the difference — "Claude-4 does `create_file → run_test` far more than GPT-4." An embedding distance can't tell you that.
- **No model required.** Reproducible without loading any neural network; deterministic given the same vocabulary.
- **The structure is discrete.** Agent actions are discrete events in a specific order, not a continuous signal. Compressing them into an embedding trades away the ordering information that often explains why trajectories differ.

Embedding-based comparison is a valid alternative but not what this library does.

## Can procgrep run my agent?

No. procgrep is post-hoc analysis only. It reads trace files; it
does not execute agents or call models. You capture traces with
whatever harness you already have, then point procgrep at the
resulting JSONL.

## Does procgrep see everything the agent does?

Only what appears in the trace log. Some scaffolds run internal steps (like test execution inside a planner) that never surface as visible tool calls. If those steps aren't in the log, procgrep doesn't see them.

One concrete example: a Moatless trajectory contained 71 internal `RunTests` references that never appeared as captured actions. procgrep would show "no tests run" — technically accurate about what was logged, but missing real behavior.

When you get an unexpected result, check whether the scaffold you're using surfaces all its actions to the trace. This varies per scaffold and should be audited once per new adapter.

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

## What can the pattern matcher express?

The current pattern matcher takes YAML rules where each rule is a regex over the space-joined atom sequence. It can express things like "no run of five consecutive edits without a test" or "a search must precede the first edit."

It cannot express temporal reasoning ("eventually X follows Y within 10 steps"), variable binding across atoms, or probabilistic predicates ("edit proportion below 0.6"). Those require a richer rule language that does not yet exist in procgrep.

If your check can be stated as a pattern over the sequence — including contiguous runs, prefix requirements, and absence conditions — the current matcher handles it. For anything more structural, check back.

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
a fixed BPE procedure vocabulary, normalized to sum to one for use as a
probability distribution. It is not a hash, not an embedding, and
not a cryptographic identity. Multiple trajectories can share a
fingerprint, especially under aggressive BPE merges.
