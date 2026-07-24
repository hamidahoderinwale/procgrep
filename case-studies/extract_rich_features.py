"""Extended extractor: one-pass re-stream capturing atoms + file paths + tokens.

Augments the base fingerprints with two new axes:

TOKEN ANALYSIS (from info.model_stats per trajectory):
  - tokens_sent, tokens_received, api_calls per trajectory
  - Derived: tokens_per_action, cost_per_action, verbosity ratio

FILE CONSUMPTION (from action strings per step):
  - Set of file paths READ (view/open/cat) per trajectory
  - Set of file paths EDITED (str_replace/edit) per trajectory
  - Total distinct files touched

Writes a SEPARATE JSONL with the same instance_id key so it joins
on top of the existing fingerprints_*.jsonl for combined analysis.

    python extract_rich_features.py --submission <s3_submission_prefix> --limit 50 --out rich_features.jsonl
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

BUCKET = "swe-bench-submissions"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

_FILE_RE = re.compile(
    r"(?:"
    # Absolute paths: /testbed/foo.py, /tmp/bar.py, /workspace/x.py
    r"/(?:testbed|tmp|workspace|repo)[/\w.\-]+"
    r"|"
    # Relative paths with at least one directory: astropy/modeling/foo.py, src/lib.py
    r"(?:\w[\w.\-]*/)+[\w.\-]+"
    r"|"
    # Bare filenames with common extensions: reproduce_issue.py, fix.py
    r"[\w][\w.\-]*"
    r")"
    r"\.(?:py|js|ts|java|cpp|c|h|rb|go|rs|md|txt|yaml|yml|toml|json|sh|bash)"
)


def list_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        url = f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        with urllib.request.urlopen(url, context=CTX, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        body = re.sub(r' xmlns="[^"]+"', "", body, count=1)
        root = ET.fromstring(body)
        keys.extend(n.text or "" for n in root.findall(".//Contents/Key"))
        is_truncated = (root.find(".//IsTruncated") or ET.Element("d")).text
        if is_truncated != "true":
            break
        token_el = root.find(".//NextContinuationToken")
        token = token_el.text if token_el is not None else None
        if not token:
            break
    return keys


def fetch(key: str) -> bytes:
    url = f"https://{BUCKET}.s3.amazonaws.com/{key}"
    with urllib.request.urlopen(url, context=CTX, timeout=60) as resp:
        return resp.read()


def extract_files_from_action(action: str) -> tuple[list[str], list[str]]:
    """Return (read_paths, edit_paths) found in an action string."""
    paths = _FILE_RE.findall(action)
    if not paths:
        return [], []
    action_lower = action.lower().lstrip()
    is_edit = any(
        action_lower.startswith(v)
        for v in (
            "edit ",
            "str_replace ",
            "str_replace_editor str_replace",
            "str_replace_editor insert",
            "str_replace_editor create",
        )
    )
    if is_edit:
        return [], paths
    return paths, []


def process_trajectory(traj_record: dict) -> dict:
    """Extract token stats + file paths from one trajectory record."""
    info = traj_record.get("info") or {}
    stats = info.get("model_stats") or {}
    tokens_sent = int(stats.get("tokens_sent", 0))
    tokens_received = int(stats.get("tokens_received", 0))
    api_calls = int(stats.get("api_calls", 0))
    instance_cost = float(stats.get("instance_cost", 0.0))

    traj = traj_record.get("trajectory") or []
    n_actions = len(traj)
    all_read: set[str] = set()
    all_edit: set[str] = set()
    for step in traj:
        if not isinstance(step, dict):
            continue
        action = step.get("action") or ""
        if not isinstance(action, str):
            action = str(action)
        r, e = extract_files_from_action(action)
        all_read.update(r)
        all_edit.update(e)

    return {
        "tokens_sent": tokens_sent,
        "tokens_received": tokens_received,
        "api_calls": api_calls,
        "instance_cost": instance_cost,
        "n_action_steps": n_actions,
        "tokens_per_action": round(tokens_received / max(1, n_actions), 1),
        "files_read": sorted(all_read),
        "files_edited": sorted(all_edit),
        "n_files_read": len(all_read),
        "n_files_edited": len(all_edit),
        "n_files_total": len(all_read | all_edit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    prefix = f"{args.submission}/trajs/"
    keys = [k for k in list_keys(prefix) if k.endswith(".traj")][: args.limit]
    print(f"{args.submission}: {len(keys)} trajectories → {args.out}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n_done = 0
    n_fail = 0
    total_tokens = 0
    total_files = 0

    with out_path.open("w") as f:
        for i, key in enumerate(keys, 1):
            stem = key[len(prefix) :].rsplit(".traj", 1)[0]
            instance_id = stem.split("/")[-1]
            try:
                traj_record = json.loads(fetch(key))
                features = process_trajectory(traj_record)
            except Exception as e:
                n_fail += 1
                print(f"  [{i:3d}] FAIL {instance_id}: {e}")
                continue
            total_tokens += features["tokens_received"]
            total_files += features["n_files_total"]
            f.write(json.dumps({"instance_id": instance_id, **features}) + "\n")
            f.flush()
            n_done += 1
            if i % 10 == 0 or i == len(keys):
                elapsed = time.time() - started
                print(
                    f"  [{i:3d}/{len(keys)}] ok  tokens/action={features['tokens_per_action']}  files={features['n_files_total']}  ({elapsed:.0f}s)"
                )

    elapsed = time.time() - started
    print(f"\nDone: {n_done} ok, {n_fail} failed in {elapsed:.0f}s")
    if n_done:
        print(f"  avg tokens/action: {total_tokens / n_done / 10:.1f} (rough)")
        print(f"  avg files touched: {total_files / n_done:.1f}")


if __name__ == "__main__":
    main()
