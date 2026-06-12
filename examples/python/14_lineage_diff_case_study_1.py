"""Case Study 1 audit skeleton: Claude 3.7 Sonnet -> SWE-agent-LM-32B.

This script wires `procgrep.lineage_diff` into a real lineage-step audit.
It is intentionally an orchestration skeleton: it expects you to supply
trajectory sources for the parent and the child, then produces a
structured diff report.

Lineage step under audit (L3 in the program arc):
    Parent (closed-weight, traces public): Claude 3.7 Sonnet running
        SWE-agent on SWE-smith instances. The 5,017 SFT-supervision
        trajectories are released as `SWE-bench/SWE-smith-trajectories`
        (MIT-licensed).
    Procedure: full-parameter SFT via torchtune (train_on_input=False;
        rejection sampling on resolved-only trajectories; 3 epochs;
        Adam fused; lr 1e-4 cosine warmup 5 steps; bf16; seed 42).
    Child: `SWE-bench/SWE-agent-LM-32B`, fine-tuned from
        Qwen2.5-Coder-32B-Instruct. 40.2% on SWE-bench Verified.

Caveats noted in the eventual writeup:
    1. The SFT supervision trajectories carry Claude 3.7 Sonnet's
       procedural fingerprint, so any "shift" measured here is partly
       Claude-procedural-transfer-via-SFT, not solely SWE-smith-task-
       distribution effects.
    2. Rejection sampling means the supervision distribution is
       successful Claude trajectories only; failure-mode procedures of
       Claude 3.7 do not appear in training data.

What you need to provide:
    parent_traces  -- canonical procgrep Traces from the 5,017
        SWE-smith-trajectories rows (loadable via from_hf below).
    child_traces   -- canonical Traces from running SWE-agent-LM-32B
        on a comparable task suite. The HF Hub may have these released
        by community runs; otherwise collect them by running the model
        with the SWE-agent scaffold and serializing the JSONL output.

Usage:
    1. Adjust the trajectory sources below to match your setup.
    2. Run: python examples/python/14_lineage_diff_case_study_1.py
    3. Audit report is written to case_study_1_report.md in CWD.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Parent traces: Claude 3.7 Sonnet via SWE-agent on SWE-smith instances.
# ---------------------------------------------------------------------------

# Option A (recommended) -- load directly from HuggingFace.
# Requires: pip install datasets
#
#     from procgrep.hf import from_hf
#     parent_traces = from_hf(
#         "SWE-bench/SWE-smith-trajectories",
#         adapter="swe-smith",
#         split="tool",
#         trace_id_field="traj_id",
#         agent_field="model",
#         limit=500,    # remove the cap for a full audit
#     )

# Option B -- load from a locally-downloaded JSONL mirror.
#
#     parent_records = list(read_jsonl(Path("data/swe-smith-trajectories.jsonl")))
#     parent_traces = canonicalize(
#         parent_records,
#         adapter="swe-smith",
#         trace_id_field="traj_id",
#         agent_field="model",
#     )

# ---------------------------------------------------------------------------
# Child traces: SWE-agent-LM-32B via SWE-agent on a comparable task suite.
# ---------------------------------------------------------------------------

# Option A -- if a public release of SWE-agent-LM-32B trajectories
# exists, load via from_hf with a matching split:
#
#     child_traces = from_hf(
#         "<dataset-id>/swe-agent-lm-32b-trajectories",
#         adapter="swe-smith",     # same chat-format schema as parent
#         trace_id_field="traj_id",
#         agent_field="model",
#         limit=500,
#     )

# Option B -- load from your own JSONL after running the model
# against SWE-bench Verified or SWE-smith with the SWE-agent scaffold:
#
#     child_records = list(read_jsonl(Path("data/swe-agent-lm-32b.jsonl")))
#     child_traces = canonicalize(
#         child_records,
#         adapter="swe-smith",
#         trace_id_field="traj_id",
#         agent_field="model",
#     )

# ---------------------------------------------------------------------------
# Compute the structured diff.
# ---------------------------------------------------------------------------

# Once both sides are loaded, the audit is a single call:
#
#     diff = lineage_diff(
#         parent=parent_traces,
#         child=child_traces,
#         parent_label="Claude 3.7 Sonnet (SWE-agent, SWE-smith)",
#         child_label="SWE-agent-LM-32B (SWE-agent, SWE-smith)",
#         along=["vocabulary", "entropy", "outcome_quadrant"],
#         outcome_field="resolved",
#     )
#
#     print(diff.summary())
#     report_path = Path("case_study_1_report.md")
#     report_path.write_text(diff.to_markdown())
#     print(f"\nFull audit report written to {report_path}")


def _main() -> None:
    """Print setup guidance and exit cleanly.

    This file is a documented skeleton, not a runnable analysis, so it returns
    success after explaining what to fill in. That keeps the "run every example"
    CI step green while leaving the template discoverable.
    """
    print(
        "This is a skeleton: pick Option A or B for both parent and "
        "child trace sources, uncomment the corresponding blocks above, "
        "and re-run. See module docstring for context."
    )


if __name__ == "__main__":
    _main()
