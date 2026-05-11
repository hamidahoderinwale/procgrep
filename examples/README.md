# procgrep examples

Tiny synthetic corpus and rule file for trying out the pipeline.

## Quick run

```bash
procgrep canonicalize \
    --input examples/synthetic_traces.jsonl \
    --adapter swe-agent \
    --output /tmp/canonical.jsonl

procgrep fit-bpe \
    --input /tmp/canonical.jsonl \
    --vocab-size 20 \
    --output /tmp/vocab.json

procgrep encode \
    --input /tmp/canonical.jsonl \
    --vocab /tmp/vocab.json \
    --output /tmp/fingerprints.jsonl

procgrep jsd \
    --input /tmp/fingerprints.jsonl \
    --group-by agent \
    --output /tmp/jsd.json

procgrep match-patterns \
    --input /tmp/canonical.jsonl \
    --rules examples/rules/stuck_edit_loop.yaml \
    --output /tmp/violations.json
```

The synthetic corpus has two agents (`editor`, `searcher`) across two
groups (`control`, `treatment`), with one trajectory deliberately
violating `no_long_edit_loops` so the pattern matcher has something
to report.
