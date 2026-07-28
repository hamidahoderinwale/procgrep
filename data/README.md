# Spine store

Precomputed action-spine stores for the explorer and the figure scripts. One
row per trajectory, with columns: `dataset`, `trace_id`, `agent`, `task`,
`spine`, `outcome`, `cot_tokens`.

- `procgrep_spines.parquet`: current canonical store.
- `procgrep_spines_v1/…/v5.parquet`, `procgrep_spines_additions_v4.parquet`:
  prior versions, kept side by side so runs stay comparable (we never
  overwrite a prior run's output).
- `add_<dataset>.parquet`: per-dataset staging spines awaiting a merge into
  the canonical store.
- `student/`: local working traces, gitignored and not published.

All `*.parquet` files are gitignored. The canonical store is published to the
Hugging Face dataset `midah/procgrep-spines` and refreshed weekly by
`.github/workflows/refresh-spines.yml`. Regenerate locally with
`analysis/build_spines.py`.
