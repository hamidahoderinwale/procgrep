# procgrep: ecosystem interface

A browsable map of the agent-trace ecosystem: which trajectory datasets exist
on the Hub, which procgrep can parse, and what is inside each, down to a single
trajectory. One self-contained `index.html` with the data embedded; opens from
`file://`, no server.

Three zoom levels, each row opening the next:

- **Ecosystem view**: adapter coverage, per-dataset redundancy, format mix,
  release timeline, and a sortable dataset index.
- **Dataset view**: traces grouped by producing model (counts, lengths,
  duplicate rates), the dataset's action mix and phase arc, and an
  export-a-diverse-subset panel.
- **Trace view**: one trajectory's canonical atom sequence aligned with the
  raw turns it came from, plus nearest neighbours by JSD.

Per-step wall-clock is rarely present in public trace data, so timing shows
dataset recency and the step-position arc instead of pretending otherwise.

## Regenerate

```bash
python interface/ecosystem_catalog.py --top 60 --out interface/data/catalog.json
python interface/build_interface.py --catalog interface/data/catalog.json \
    --profiles interface/data/profile_nebius.json --out interface/index.html
```

Pages are static over precomputed JSON from the discover → sniff → curate
pipeline; re-running the two scripts refreshes everything.
