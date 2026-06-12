# Changelog

Notable changes to procgrep. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows semver.

## [0.1.3] — current

- Procedural reward scoring against YAML specs (`reward.load_spec`, `reward.score`):
  phases, penalties, and bonuses producing a `[0, 1]` partial reward.
- Cross-corpus ingest adapters: OpenHands, ReAct-text, SWE-smith, and GumTree,
  alongside the existing SWE-agent, mini-swe-agent, Agentless, DARS, and Moatless.
- CLI: `compare` (two-group diff) and `grep` (structural pattern search) subcommands.
- Interactive D3 figures in the web essay (JSD, follow-through, representation-F1,
  distillation entropy), rendered through a shared chart module.

Earlier 0.1.x releases established the core library: canonicalization into an
action alphabet, BPE procedure induction, Jensen-Shannon divergence, lineage
diff, pattern matching, and the leave-one-group-out attribution probe.
