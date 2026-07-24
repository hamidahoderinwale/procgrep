# procgrep: interface design

A browsable map of the agent-trace ecosystem, on top of the dynamic
discover → sniff → curate pipeline. Three zoom levels: **Ecosystem → Dataset →
Trace**. Every component must name the decision it drives, or it is cut.

## Aesthetic (from the shared inspo)

Light **editorial-brutalist**: off-white paper (`#F7F5F2`), near-black ink
(`#14110E`), a real monospace for all data/atoms/queries, one serif line for the
single headline, hairline rules, one copper accent, generous whitespace, motion
only where it explains.

- **Poolside**: calm, huge whitespace, one confident statement.
- **Cursor**: a command bar is the primary verb; keyboard-first.
- **Martian**: a manifesto line; serif; science-forward framing.
- **hassaanraza**: brutalist/vintage, monospace, visible structure, numbers on show.
- **glenncatteeuw**: purposeful motion; interaction as storytelling.
- **Adaption**: (URL TBD; fold in once confirmed).

## Ecosystem view (the map)

Leads with **findings**, then the live index.

- **Coverage donut**: N discovered / M parseable, by adapter. *Decision: is the
  ecosystem mostly parseable, and which format to build an adapter for next.*
- **Redundancy bars**: per dataset, shortest-keeps → diverse-keeps gap, sorted.
  *Decision: which datasets need curation before training (SWE-Gym's 5% bar screams).*
- **Format mix**: openhands / react-text / swe-agent stacked. *Decision: where the
  fragmentation is.*
- **Release timeline**: datasets by `last_modified` (HF). *Decision: what's current.*
- **The index**: sortable table: dataset · downloads · likes · adapter · #models ·
  median length · exact-dup% · shortest→diverse · last-modified. A row opens its dataset view.

## Dataset view (the profile)

- **Header**: id, downloads/likes, adapter+confidence, last-modified, n traces.
- **By model** *(agent_field)*: group traces by the producing model; per model:
  n, median length, exact-dup%, action-mix fingerprint. Small multiples / table.
  *Decision: which model's traces to keep / which are redundant.*
- **Length**: trajectory-length distribution (histogram or ridgeline), split by
  model. *Decision: spot trivially-short or runaway traces.*
- **Redundancy / curate panel**: exact + near dup, shortest vs diverse vs random
  coverage, **export a diverse subset**. *Decision: dedup go/no-go + method.*
- **Procedural fingerprint**: the dataset's action-mix and phase arc (atom share
  by step position). *Decision: characterize "how" this corpus solves tasks.*
- **Temporal**: per-step wall-clock is rarely in the data. Show the
  **step-position arc** (atom share vs step index) + dataset recency + model era;
  surface token/cost timing only when a dataset carries it. A row opens its trace view.

## Trace view (the conversation)

Drill into one trajectory: *the conversations themselves*, not just stats.

- **Aligned transcript**: the canonical atom sequence (the procedural spine) laid
  beside the underlying turns: each turn shows its atom + the raw command/text the
  agent emitted. *Decision: see exactly how the agent worked, step by step.*
- **Spine sparkline**: the atom timeline; click an atom → jump to that turn.
- **Metadata**: model, repo, resolved?, length, fingerprint, nearest neighbours
  (by JSD) so you can see near-duplicate siblings.

## Data sources (honest)

| Field | Source | Note |
|---|---|---|
| model | `agent_field` (mapped) | reliable |
| length | `len(atoms)` | reliable |
| temporal | dataset `last_modified` + model era | per-step wall-clock usually absent → use step-position arc |
| conversation | normalized turns (already decoded) | reliable |
| redundancy / fingerprint | `curate` + `encode` | reliable |

## Build

Static pages over precomputed, cacheable JSON: re-running discover → sniff →
curate refreshes them (dynamic, not frozen):

- `catalog.json` feeds the ecosystem view (a v1 exists already).
- per-dataset `profile.json` (model breakdown, length dist, fingerprint, sampled
  traces) feeds the dataset view. **New engine step: `profile_dataset()` (group by model + sample).**
- sampled-trace JSON (atoms aligned to turns) feeds the trace view.

Deployable as an HF Space / Vercel. Stack stays close to clean defaults: one
static page + light JS, the theme palette, charts in the data-viz house style.
