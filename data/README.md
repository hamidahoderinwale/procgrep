# Spine store

Precomputed action-spine stores for the explorer and the figure scripts. One
row per trajectory, with columns: `dataset`, `trace_id`, `agent`, `task`,
`spine`, `outcome`, `cot_tokens`.

- `procgrep_spines.parquet` — current canonical store.
- `procgrep_spines_v1/2/3.parquet` — prior versions, kept side by side so runs
  stay comparable (we never overwrite a prior run's output).

All `*.parquet` files are gitignored. The canonical store is published to the
Hugging Face dataset `midah/procgrep-spines` and refreshed weekly by
`.github/workflows/refresh-spines.yml`. Regenerate locally with
`scripts/build_spines.py`.
