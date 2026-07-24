"""Ingest OpenHands-format trajectories (tool-calling JSON).

OpenHands stores each trajectory as a flat list of {role, content, tool_calls}
messages. Assistant turns use OpenAI-style tool_calls to invoke functions like
execute_bash, str_replace, create, finish. We extract the effective bash-string
action from each tool call and pass it through the existing swe_smith classifier.

Tested on: GPT-5 (openhands), Claude-4.5, Qwen3-Coder.

    python pull_openhands.py --submission <s3_submission_prefix> --limit 50 --out fingerprints.jsonl [--with-resolved]
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

from procgrep.ingest.adapters.swe_smith import classify_swe_smith_action
from procgrep.types import ATOM_THINK

BUCKET = "swe-bench-submissions"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def list_keys(prefix):
    keys = []
    token = None
    while True:
        url = f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        with urllib.request.urlopen(url, context=CTX, timeout=30) as r:
            body = r.read().decode()
            body = re.sub(r' xmlns="[^"]+"', "", body, count=1)
        root = ET.fromstring(body)
        keys.extend(n.text or "" for n in root.findall(".//Contents/Key"))
        el = root.find(".//IsTruncated")
        if (el is not None and el.text) != "true":
            break
        tel = root.find(".//NextContinuationToken")
        token = tel.text if tel is not None else None
        if not token:
            break
    return keys


def fetch(key):
    with urllib.request.urlopen(
        f"https://{BUCKET}.s3.amazonaws.com/{key}", context=CTX, timeout=60
    ) as r:
        return json.loads(r.read())


def tool_call_to_action(tc: dict) -> str:
    """Convert one OpenAI-style tool call to a bash-like action string."""
    if not isinstance(tc, dict):
        return ""
    fn = tc.get("function") or {}
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except (json.JSONDecodeError, TypeError):
        args = {}

    if name in ("execute_bash", "run_bash", "bash", "execute_command"):
        return args.get("command", "") or args.get("cmd", "")
    if name in ("str_replace_editor", "text_editor"):
        # OpenHands/Anthropic tool: has a `command` subfield (view, str_replace, create, etc.)
        subcmd = args.get("command", "view")
        path = args.get("path", args.get("file_path", ""))
        return f"str_replace_editor {subcmd} {path}".rstrip()
    if name in ("str_replace", "str_replace_based_edit_tool"):
        path = args.get("path", args.get("file_path", ""))
        return (
            f"str_replace_editor str_replace {path}" if path else "str_replace_editor str_replace"
        )
    if name in ("create_file", "write_file"):
        path = args.get("path", args.get("file_path", ""))
        return f"str_replace_editor create {path}" if path else "str_replace_editor create"
    if name in ("view", "read_file", "open"):
        path = args.get("path", args.get("file_path", ""))
        return f"str_replace_editor view {path}" if path else "str_replace_editor view"
    if name in ("finish", "submit", "end"):
        return "submit"
    if name in ("search", "find", "grep"):
        query = args.get("query", args.get("pattern", ""))
        return f"grep -r {query!r} ." if query else "grep"
    # Fallback: use function name so classifier tags it
    return name


def fingerprint_openhands(messages: list) -> tuple[list[str], list[str]]:
    """Walk an OpenHands message list, return (canonical_atoms, native_atoms)."""
    canonical, native = [], []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        # Thought is in content (may be text blocks or plain string)
        content = msg.get("content") or ""
        if isinstance(content, list):
            text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        else:
            text = str(content)
        if text.strip():
            canonical.append(ATOM_THINK)
            native.append(ATOM_THINK)

        # Action is from tool_calls (top-level OR in additional_kwargs for some schemas)
        tool_calls = (
            msg.get("tool_calls") or (msg.get("additional_kwargs") or {}).get("tool_calls") or []
        )
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except:
                tool_calls = []
        for tc in tool_calls if isinstance(tool_calls, list) else []:
            action = tool_call_to_action(tc)
            c, n = classify_swe_smith_action(action)
            canonical.append(c)
            native.append(n)
            break  # one action per assistant turn

    return canonical, native


def get_resolved(submission: str, instance_id: str) -> bool | None:
    key = f"{submission}/logs/{instance_id}/report.json"
    try:
        with urllib.request.urlopen(
            f"https://{BUCKET}.s3.amazonaws.com/{key}", context=CTX, timeout=20
        ) as r:
            return bool(json.loads(r.read()).get("resolved"))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", required=True)
    parser.add_argument("--with-resolved", action="store_true")
    args = parser.parse_args()

    prefix = f"{args.submission}/trajs/"
    keys = [k for k in list_keys(prefix) if k.endswith(".json")][: args.limit]
    print(f"{args.submission}: {len(keys)} trajectories → {args.out}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = n_fail = 0
    started = time.time()
    canon_counter: dict = {}
    native_counter: dict = {}

    with out_path.open("w") as f:
        for i, key in enumerate(keys, 1):
            iid = key[len(prefix) :].rsplit(".json", 1)[0].split("/")[-1]
            try:
                messages = fetch(key)
                if isinstance(messages, dict):
                    messages = messages.get("messages") or messages.get("history") or []
                canonical, native = fingerprint_openhands(messages)
            except Exception as e:
                n_fail += 1
                print(f"  [{i:3d}] FAIL {iid}: {e}")
                continue

            resolved = get_resolved(args.submission, iid) if args.with_resolved else None
            for a in canonical:
                canon_counter[a] = canon_counter.get(a, 0) + 1
            for a in native:
                native_counter[a] = native_counter.get(a, 0) + 1
            f.write(
                json.dumps(
                    {
                        "instance_id": iid,
                        "resolved": resolved,
                        "atoms_canonical": canonical,
                        "atoms_native": native,
                        "n_steps": sum(1 for a in canonical if a != ATOM_THINK),
                    }
                )
                + "\n"
            )
            f.flush()
            n_done += 1
            if i % 10 == 0 or i == len(keys):
                print(f"  [{i:3d}/{len(keys)}] ok  ({time.time()-started:.0f}s)")

    total = sum(canon_counter.values())
    print(f"\nDone: {n_done} ok, {n_fail} failed")
    print("Canonical distribution:")
    for a, c in sorted(canon_counter.items(), key=lambda x: -x[1]):
        print(f"  {c:6d}  {100*c/max(1,total):5.1f}%  {a}")


if __name__ == "__main__":
    main()
