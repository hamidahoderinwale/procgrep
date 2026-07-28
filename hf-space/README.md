---
title: ProcGrep Explorer
emoji: 🔎
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# ProcGrep explorer (live backend)

**Intent.** A Hugging Face Docker Space that runs ProcGrep server-side: it ingests a
trajectory dataset, canonicalizes it into an action vocabulary, and answers structural
queries over the **whole** dataset, not a fixed sample. Read this when changing the live
explorer or its hosting. The static, embedded-data version lives in this repo at
`docs/explorer/`; this Space is the version that scales past the embed's size limit.

## Why a Space and not just the static page

The static explorer embeds every trace's atom spine in the HTML, so it is bounded by page
weight (a few hundred traces per dataset) and ships only datasets pre-profiled offline.
A Space has a backend, so it can load full datasets on demand and ingest any HF set. It also
runs on HF infrastructure, where reading HF datasets is fast: the parquet streaming that
times out elsewhere is not a problem here.

## Design decisions (benefit / price)

1. **Import procgrep; do not reimplement it.** One definition of a procedure across the
   paper and the demo. Price: the Space pins a procgrep revision and rebuilds to update.
2. **Cache canonicalized traces per dataset under an LRU.** The expensive ingest runs once;
   queries are then a microsecond scan. Price: a cold dataset pays the ingest cost once,
   and memory grows with the number of cached datasets (`MAX_DATASETS`).
3. **Bound each ingest (`MAX_TRACES`, timeout).** Predictable latency and memory on the
   free CPU tier. Price: a very large dataset is sampled to the cap; the response says so.
4. **Queries are regexes over the canonical atom spine.** One query language shared with the
   static essay's query box. Price: the spine drops argument-level detail, by design.

## Endpoints

- `GET /`: the query frontend.
- `GET /datasets`: suggested datasets + which are warm in cache.
- `POST /query`: `{dataset, pattern}` → match count, per-model rates, matched-vs-all action
  mix, and a sample of matched traces.

## Run locally

    pip install -r requirements.txt
    uvicorn app:app --reload --port 7860
    # open http://localhost:7860
