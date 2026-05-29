"""Stream-fetch SWE-bench/experiments trajectories, fingerprint with procgrep, save only atoms.

Disk-frugal pipeline: each ``.traj`` is ~1.5MB raw but only ~few KB of atoms.
Stream each trajectory through the classifier, save just the canonical/native
atom sequences + minimal metadata, discard the raw payload.

Usage:
    python pull_and_fingerprint.py --submission verified/20250511_sweagent_lm_32b --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

_PROCGREP_SRC = Path(__file__).resolve().parent.parent / "procgrep" / "src"
if _PROCGREP_SRC.exists() and str(_PROCGREP_SRC) not in sys.path:
    sys.path.insert(0, str(_PROCGREP_SRC))

from procgrep.adapters.swe_smith import classify_swe_smith_action  # noqa: E402
from procgrep.types import ATOM_THINK  # noqa: E402

BUCKET = "swe-bench-submissions"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def list_keys(prefix: str, max_keys: int = 1000) -> list[str]:
    """List object keys under ``prefix`` via S3 XML listing (paginated)."""
    keys: list[str] = []
    continuation = None
    while True:
        url = f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&prefix={prefix}&max-keys=1000"
        if continuation:
            url += f"&continuation-token={urllib.parse.quote(continuation)}"
        with urllib.request.urlopen(url, context=CTX, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        body = re.sub(r' xmlns="[^"]+"', "", body, count=1)
        root = ET.fromstring(body)
        keys.extend(n.text or "" for n in root.findall(".//Contents/Key"))
        if len(keys) >= max_keys:
            break
        is_truncated = (root.find(".//IsTruncated") or ET.Element("dummy")).text
        if is_truncated != "true":
            break
        continuation = (root.find(".//NextContinuationToken") or ET.Element("dummy")).text
        if not continuation:
            break
    return keys[:max_keys]


def fetch_trajectory(key: str) -> dict:
    url = f"https://{BUCKET}.s3.amazonaws.com/{key}"
    with urllib.request.urlopen(url, context=CTX, timeout=60) as resp:
        return json.loads(resp.read())


def fingerprint(traj_record: dict) -> tuple[list[str], list[str], dict]:
    """Walk the trajectory, emit (canonical_atoms, native_atoms, meta)."""
    traj = traj_record.get("trajectory") or []
    canonical: list[str] = []
    native: list[str] = []
    for step in traj:
        if not isinstance(step, dict):
            continue
        thought = step.get("thought")
        if isinstance(thought, str) and thought.strip():
            canonical.append(ATOM_THINK)
            native.append(ATOM_THINK)
        action = step.get("action") or ""
        if not isinstance(action, str):
            action = ""
        c, n = classify_swe_smith_action(action)
        canonical.append(c)
        native.append(n)

    info = traj_record.get("info") or {}
    meta = {
        "n_steps": len(traj),
        "exit_status": info.get("exit_status"),
        "has_submission": bool(info.get("submission")),
    }
    return canonical, native, meta


def fetch_resolve_label(submission: str, instance_id: str) -> bool | None:
    """Fetch resolved label, trying two formats:

    Format A (2025): logs/<iid>/report.json  — flat or nested JSON
    Format B (2024): logs/<iid>.<slug>.eval.log — raw pytest output
    """
    # Format A
    key = f"{submission}/logs/{instance_id}/report.json"
    try:
        url = f"https://{BUCKET}.s3.amazonaws.com/{key}"
        with urllib.request.urlopen(url, context=CTX, timeout=30) as resp:
            data = json.loads(resp.read())
        if "resolved" in data:
            return bool(data["resolved"])
        nested = data.get(instance_id, {})
        if "resolved" in nested:
            return bool(nested["resolved"])
    except Exception:
        pass

    # Format B — slug is the part after the first "/" (strips "verified/" or "lite/")
    slug = submission.split("/", 1)[-1]
    log_key = f"{submission}/logs/{instance_id}.{slug}.eval.log"
    try:
        url = f"https://{BUCKET}.s3.amazonaws.com/{log_key}"
        with urllib.request.urlopen(url, context=CTX, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        if "All Tests Passed" in text:
            return True
        if "Tests Failed" in text:
            return False
    except Exception:
        pass

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        default="verified/20250511_sweagent_lm_32b",
        help="Submission path under the bucket (no trailing slash)",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Number of trajectories to fingerprint"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: results/fingerprints_<submission>.jsonl)",
    )
    parser.add_argument(
        "--with-resolved",
        action="store_true",
        help="Also fetch resolve labels from logs/<instance>/report.json",
    )
    args = parser.parse_args()

    sub_name = args.submission.rstrip("/").split("/")[-1]
    out_path = (
        Path(args.out)
        if args.out
        else (Path(__file__).parent / "results" / f"fingerprints_{sub_name}.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.submission}/trajs/"
    print(f"Listing keys under: s3://{BUCKET}/{prefix}")
    keys = list_keys(prefix, max_keys=10000)
    # Accept both flat (verified/<sub>/trajs/<id>.traj) and nested
    # (lite/<sub>/trajs/<id>/<id>.traj) layouts.
    traj_keys = [k for k in keys if k.endswith(".traj")]
    traj_keys = traj_keys[: args.limit]
    print(
        f"  {len(traj_keys)} .traj keys to fetch (limit={args.limit}, "
        f"resolve_labels={'yes' if args.with_resolved else 'no'})"
    )
    print(f"Output: {out_path}")
    print()

    started = time.time()
    n_done = 0
    n_failed = 0
    canonical_atom_counter: dict = {}
    native_atom_counter: dict = {}
    n_resolved = 0
    n_unresolved = 0
    n_unknown = 0

    with out_path.open("w") as out:
        for i, key in enumerate(traj_keys, 1):
            # Strip prefix and .traj extension; nested has the dir twice
            stem = key[len(prefix) :].rsplit(".traj", 1)[0]
            instance_id = stem.split("/")[-1]
            try:
                traj_record = fetch_trajectory(key)
                canonical, native, meta = fingerprint(traj_record)
            except Exception as e:
                n_failed += 1
                print(
                    f"  [{i:3d}/{len(traj_keys)}] FAIL  {instance_id}  ({type(e).__name__}: {str(e)[:80]})"
                )
                continue

            resolved = None
            if args.with_resolved:
                resolved = fetch_resolve_label(args.submission, instance_id)
                if resolved is True:
                    n_resolved += 1
                elif resolved is False:
                    n_unresolved += 1
                else:
                    n_unknown += 1

            for a in canonical:
                canonical_atom_counter[a] = canonical_atom_counter.get(a, 0) + 1
            for a in native:
                native_atom_counter[a] = native_atom_counter.get(a, 0) + 1

            out.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "n_steps": meta["n_steps"],
                        "exit_status": meta["exit_status"],
                        "has_submission": meta["has_submission"],
                        "resolved": resolved,
                        "atoms_canonical": canonical,
                        "atoms_native": native,
                    }
                )
                + "\n"
            )
            out.flush()
            n_done += 1
            if i % 25 == 0 or i == len(traj_keys):
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0
                resolve_summary = (
                    f"R={n_resolved}/U={n_unresolved}/?={n_unknown}" if args.with_resolved else ""
                )
                print(
                    f"  [{i:4d}/{len(traj_keys)}] ok    {instance_id}  ({rate:.1f} traj/s, {elapsed:.0f}s) {resolve_summary}"
                )

    elapsed = time.time() - started
    print()
    print("=" * 72)
    print(f"Done: {n_done} fingerprinted, {n_failed} failed in {elapsed:.1f}s")
    print()
    total_atoms = sum(canonical_atom_counter.values())
    print(f"Canonical atom distribution (n={total_atoms} total):")
    for a, c in sorted(canonical_atom_counter.items(), key=lambda x: -x[1]):
        print(f"  {c:7d}  {100 * c / max(1, total_atoms):5.1f}%  {a}")
    print()
    print(f"Native top 12 (n={sum(native_atom_counter.values())} total):")
    for a, c in sorted(native_atom_counter.items(), key=lambda x: -x[1])[:12]:
        print(f"  {c:7d}  {a}")


if __name__ == "__main__":
    main()
