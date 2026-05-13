# Contributing to procgrep

## Setting up a development environment

```bash
git clone https://github.com/hamidahoderinwale/procgrep.git
cd procgrep
python3.10 -m venv .venv   # 3.10, 3.11, 3.12, or 3.13
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

After `pre-commit install` runs, every commit will trigger the
linting, formatting, and type-checking hooks automatically.

## Running the gates locally

```bash
ruff check .
ruff format --check .
mypy src/procgrep
pytest -q
```

All four must pass before opening a pull request. The CI runs the
same four on Python 3.10 through 3.13.

You can also run the example scripts as integration smoke tests:

```bash
for script in examples/python/*.py; do python "$script"; done
```

## Code style anchors

- Python 3.10 minimum.
- `from __future__ import annotations` at the top of every module.
- Type hints on every public signature; `mypy --strict` must pass.
- Single-purpose functions; keep modules under ~300 lines.
- Module-level docstrings stating intent before any code.
- No em dashes in prose or in docstrings.
- Deterministic seeds for any randomness; default seed is `0`.

## Adding a new scaffold adapter

1. Add the mapping logic in `src/procgrep/canonicalize.py` using
   `make_action_adapter` if the scaffold fits the list-of-actions
   shape, or write a custom `TraceAdapter` callable.
2. Register the adapter inside `_register_builtins`.
3. Add a per-adapter test in `tests/test_canonicalize.py` covering
   at least one synthetic record.
4. Document the adapter's expected raw-trace shape in its
   registration block.

If the adapter is non-trivial (multiple files, multiple action
schemas, version-sensitive), open an issue first so we can discuss
the right place to land it (in-tree built-in vs out-of-tree plugin).

## Adding a new analysis primitive

1. Decide where it lands. New helpers on top of existing types
   (`Fingerprint`, `Trace`, `MotifVocabulary`) go in
   `src/procgrep/stats.py` or alongside the primitive they extend.
2. Write the function as a pure function of the existing types
   when possible. Avoid adding required new state.
3. Add tests that cover at least one invariant of the primitive
   (idempotence, symmetry, expected bounds, determinism under
   fixed seed, error on degenerate input).
4. Add an example in `examples/python/` if the primitive enables a
   workflow the existing examples do not cover.

## Pull request conventions

- Branch name: `<short-topic>` (e.g., `add-ci-helper`, `fix-jsd-edge`).
- Title: imperative ("Add ...", "Fix ..."), under 70 characters.
- Body: what changed, why, validation steps, any reversibility notes.
- One logical change per PR. Split refactors from features.
- Update the README or FAQ if the change affects the public surface.

## Reporting issues

Open an issue with: a minimal reproducible example, the procgrep
version (`procgrep.__version__`), the Python version, the operating
system, and the relevant error or unexpected behavior.

For methodological questions (when is JSD the right metric, how do I
fit BPE on a small corpus, etc.), the FAQ is the first stop; if the
question is not answered there, open an issue with the `question`
label.

## Releasing

procgrep follows semantic versioning.

- Patch (0.1.x): documentation, internal cleanups, additive bug
  fixes.
- Minor (0.x.0): new public capabilities, additive only.
- Major (x.0.0): breaking API changes.

Each release bumps `__version__` in `src/procgrep/__init__.py` and
the `version` field in `CITATION.cff` and `pyproject.toml`, then
tags the commit on `main`.
