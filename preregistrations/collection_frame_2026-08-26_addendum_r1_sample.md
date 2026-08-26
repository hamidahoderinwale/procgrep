# Addendum — R1 sample size (2026-08-26)

Amends `collection_frame_2026-08-25.md` (f578fc9) for cell R1 (Qwen3-32B under a 100-step limit). Not an edit of the frame; the frame stands.

**Change.** R1 collects 100 runs (smoke 10 + fleet 90, rig seed-42 order from index 0), not 500.

**Why.** Measured throughput on the rig is ~3–4 tasks per hour per 4×L4 VM under the 100-step limit (4 smoke tasks in the first 1.5 h), so 500 runs is ~130 h ≈ $600. The decision rule is stated per 100 runs (finishes at steps 37–40 fall below 2 per 100 from 8.9, and a rise appears in the last five steps before 100) and does not need more.

**Unchanged.** Decoding, limits, gate, and the decision rule. Registered before any R1 fleet run had finished; the 5 smoke runs on disk at registration were not read.

hamidah, 2026-08-26 ("fix" on the proposed ~100-run sample).
