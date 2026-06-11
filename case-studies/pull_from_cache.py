"""Fingerprint agents from the local bidirect-align-dev-traces trajectory cache.

Some 2025 SWE-bench submissions store only patches in the public S3 archive;
their full trajectories are cached locally in the bidirect project.
This script reads those and produces standard fingerprint JSONL files.

Handles two formats currently present in the cache:
  sweagent_traj_subdir — standard SWE-agent .traj JSON (history + info)
  dars_traj_list       — DARS list of {role, content, thought, action} steps

Usage:
    python pull_from_cache.py --agent dars
    python pull_from_cache.py --agent claude37
    python pull_from_cache.py --agent all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
CACHE = Path(
    "/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces/output/trajectories/.cache"
)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path("/Users/hamidaho/learning-from-dev/procgrep/src")))
from pull_and_fingerprint import fetch_resolve_label
from pull_and_fingerprint import fingerprint as swe_fingerprint

from procgrep.ingest.adapters.swe_smith import classify_swe_smith_action
from procgrep.types import ATOM_THINK

AGENTS = {
    "dars": {
        "sub": "lite/20250205_dars_agent_claude_3.5_sonnet_deepseek_r1",
        "format": "dars_traj_list",
        "out": "fingerprints_dars_r1_n300.jsonl",
        "label": "DARS+R1",
    },
    "claude37": {
        "sub": "20250226_sweagent_claude-3-7-sonnet-20250219",
        "format": "sweagent_traj_subdir",
        "out": "fingerprints_claude37_parent_n300.jsonl",
        "label": "Claude-3.7 Sonnet",
    },
}


# ── Atom extractors ───────────────────────────────────────────────────────────


def atoms_from_sweagent(content: dict) -> tuple[list[str], list[str]]:
    """Extract canonical + native atoms from a SWE-agent traj dict.

    Reuses the fingerprint() function from pull_and_fingerprint which
    already handles the trajectory → atoms pipeline correctly.
    """
    atoms_c, atoms_n, _ = swe_fingerprint(content)
    return atoms_c, atoms_n


def atoms_from_dars(steps: list[dict]) -> tuple[list[str], list[str]]:
    """Extract canonical + native atoms from DARS traj_list steps.

    DARS action strings are raw bash commands (same class as SWE-smith),
    so we use classify_swe_smith_action rather than the high-level ATOM_MAP.
    """
    atoms_c, atoms_n = [], []
    for step in steps:
        if step.get("role") != "assistant":
            continue
        thought = step.get("thought", "") or ""
        action = step.get("action", "") or ""
        if thought.strip():
            atoms_c.append(ATOM_THINK)
            atoms_n.append("think")
        if action.strip():
            canon, native = classify_swe_smith_action(action)
            atoms_c.append(canon)
            atoms_n.append(native)
    return atoms_c, atoms_n


def get_resolved_from_cache(content: dict | list, fmt: str) -> bool | None:
    """Extract resolved label from trajectory info if present."""
    if not isinstance(content, dict):
        return None
    info = content.get("info", {})
    resolved = info.get("resolved")
    if resolved is not None:
        return bool(resolved)
    return None


# ── Main ─────────────────────────────────────────────────────────────────────


def fingerprint_agent(cfg: dict) -> None:
    sub_dir = CACHE / cfg["sub"]
    if not sub_dir.exists():
        print(f"  Cache directory not found: {sub_dir}")
        return

    traj_files = sorted(sub_dir.glob("*.json"))
    out_path = RES / cfg["out"]

    # Resume
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["instance_id"])
            except Exception:
                pass

    traj_files = [f for f in traj_files if f.stem not in done]
    print(f"\n{cfg['label']}  ({cfg['sub']})")
    print(f"  {len(done)} already done, {len(traj_files)} remaining → {out_path.name}")

    n_ok = n_fail = n_empty = 0
    with out_path.open("a") as f:
        for tf in traj_files:
            iid = tf.stem
            try:
                d = json.loads(tf.read_text())
                fmt = d.get("format", "")
                content = d["content"]
                resolved = get_resolved_from_cache(content, fmt)

                if fmt == "sweagent_traj_subdir":
                    atoms_c, atoms_n = atoms_from_sweagent(content)
                elif fmt == "dars_traj_list":
                    atoms_c, atoms_n = atoms_from_dars(content)
                else:
                    n_fail += 1
                    continue

                if not atoms_c:
                    n_empty += 1
                    continue

                rec = {
                    "instance_id": iid,
                    "n_steps": len(atoms_c),
                    "resolved": resolved,
                    "atoms_canonical": atoms_c,
                    "atoms_native": atoms_n,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_ok += 1
            except Exception:
                n_fail += 1

    print(f"  Done: {n_ok} ok, {n_fail} failed, {n_empty} empty")


def backfill_resolved(cfg: dict) -> None:
    """Fetch resolved labels from S3 report.json for rows that are missing them."""
    out_path = RES / cfg["out"]
    if not out_path.exists():
        return
    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    needs = [r for r in rows if r.get("resolved") is None]
    if not needs:
        print(f"  {cfg['label']}: all resolved labels present")
        return

    print(f"  {cfg['label']}: fetching resolved labels for {len(needs)} rows...")
    updated = 0
    for r in needs:
        res = fetch_resolve_label(cfg["sub"], r["instance_id"])
        if res is not None:
            r["resolved"] = res
            updated += 1

    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    labeled = sum(1 for r in rows if r.get("resolved") is not None)
    pr = sum(bool(r["resolved"]) for r in rows if r.get("resolved") is not None)
    print(
        f"  {cfg['label']}: {updated} labels added → {labeled}/{len(rows)} labeled, "
        f"pass rate={pr/max(1,labeled):.1%}"
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="all", help="dars | claude37 | all")
    args = ap.parse_args()

    targets = AGENTS if args.agent == "all" else {args.agent: AGENTS[args.agent]}
    for name, cfg in targets.items():
        fingerprint_agent(cfg)
    print("\nBackfilling resolved labels from S3...")
    for name, cfg in targets.items():
        backfill_resolved(cfg)


if __name__ == "__main__":
    main()
