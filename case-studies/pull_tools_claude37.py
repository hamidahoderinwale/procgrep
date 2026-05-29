"""Pull Claude-3.7 Sonnet tools-format fingerprints for scaffold ablation.

The tools-format submission stores trajectories as .txt files containing
Claude's native XML tool-call format:

  <function_calls>
  <invoke name="bash">
  <parameter name="command">ls /testbed</parameter>
  </invoke>
  </function_calls>

This is a different format from SWE-agent .traj JSON. We parse the XML
to extract tool calls and map them to canonical atoms.

Output: results/fingerprints_tools_claude37_n500.jsonl
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
BUCKET = "swe-bench-submissions"
SUB = "verified/20250224_tools_claude-3-7-sonnet"
OUT = RES / "fingerprints_tools_claude37_n500.jsonl"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

EDIT_CMDS = {"str_replace", "insert", "undo_edit"}
VIEW_CMDS = {"view", ""}
CREATE_CMDS = {"create"}

# Bash commands that are clearly test runners
TEST_RE = re.compile(
    r"\b(pytest|python -m pytest|python -m unittest|tox|nose2|" r"make test|./test|bash.*test)\b",
    re.IGNORECASE,
)
# Bash commands that are search operations
SEARCH_RE = re.compile(r"\b(grep|find|ag|rg|git log|git diff)\b", re.IGNORECASE)


def classify_invoke(name: str, params: dict[str, str]) -> tuple[str, str]:
    """Map a <invoke name=...> call to (canonical_atom, native_tag)."""
    cmd = params.get("command", "").strip()

    if name in ("bash", "execute_bash", "terminal"):
        if TEST_RE.search(cmd):
            return "run_test", "bash:test"
        if SEARCH_RE.search(cmd):
            return "search_repo", "bash:search"
        if cmd.startswith("cat ") or cmd.startswith("less ") or cmd.startswith("head "):
            return "read_file", "bash:read"
        return "other", f"bash:{cmd[:20].strip() or 'cmd'}"

    if name in ("str_replace_editor", "text_editor"):
        sub_cmd = params.get("command", "view")
        if sub_cmd in EDIT_CMDS:
            return "edit", f"str_replace_editor:{sub_cmd}"
        if sub_cmd in VIEW_CMDS:
            return "read_file", "str_replace_editor:view"
        if sub_cmd in CREATE_CMDS:
            return "create_file", "str_replace_editor:create"
        return "edit", f"str_replace_editor:{sub_cmd}"

    if name in ("read_file", "view_file"):
        return "read_file", f"read_file:{name}"

    if name in ("write_file", "create_file"):
        return "create_file", f"create:{name}"

    if name in ("search_files", "search_dir", "file_search", "search_code"):
        return "search_repo", f"search:{name}"

    if name in ("finish", "submit"):
        return "submit", "submit"

    return "other", f"other:{name}"


# Match <function_calls>...</function_calls> blocks including multi-invoke
FC_BLOCK_RE = re.compile(
    r"<function_calls>(.*?)</function_calls>",
    re.DOTALL,
)
INVOKE_RE = re.compile(
    r"<invoke\s+name=[\"']([^\"']+)[\"']>(.*?)</invoke>",
    re.DOTALL,
)
PARAM_RE = re.compile(
    r"<parameter\s+name=[\"']([^\"']+)[\"']>(.*?)</parameter>",
    re.DOTALL,
)


def parse_txt_trajectory(text: str) -> tuple[list[str], list[str]]:
    """Extract (atoms_canonical, atoms_native) from a tools .txt trajectory."""
    canon: list[str] = []
    native: list[str] = []

    # Split on function_call blocks; text between blocks is assistant reasoning → think
    last_end = 0
    for fc_match in FC_BLOCK_RE.finditer(text):
        # Any non-empty text before this block = thinking
        gap = text[last_end : fc_match.start()].strip()
        if gap:
            # Only count as think if it's substantive (not just whitespace)
            canon.append("think")
            native.append("think")

        # Parse all invokes inside this block
        for inv_match in INVOKE_RE.finditer(fc_match.group(1)):
            inv_name = inv_match.group(1)
            params = {m.group(1): m.group(2) for m in PARAM_RE.finditer(inv_match.group(2))}
            c, n = classify_invoke(inv_name, params)
            canon.append(c)
            native.append(n)

        last_end = fc_match.end()

    return canon, native


def list_keys(prefix: str, max_keys: int = 3000) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        url = f"https://{BUCKET}.s3.amazonaws.com/" f"?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        with urllib.request.urlopen(url, context=CTX, timeout=30) as r:
            body = r.read().decode()
        body = re.sub(r' xmlns="[^"]+"', "", body, count=1)
        root = ET.fromstring(body)
        keys.extend(n.text or "" for n in root.findall(".//Contents/Key"))
        if len(keys) >= max_keys:
            break
        if (root.findtext(".//IsTruncated") or "").lower() != "true":
            break
        tel = root.find(".//NextContinuationToken")
        token = tel.text if tel is not None else None
        if not token:
            break
    return keys[:max_keys]


def fetch_text(key: str) -> str:
    with urllib.request.urlopen(
        f"https://{BUCKET}.s3.amazonaws.com/{key}", context=CTX, timeout=60
    ) as r:
        return r.read().decode("utf-8", errors="replace")


def get_resolved(instance_id: str) -> bool | None:
    key = f"{SUB}/logs/{instance_id}/report.json"
    try:
        with urllib.request.urlopen(
            f"https://{BUCKET}.s3.amazonaws.com/{key}", context=CTX, timeout=20
        ) as r:
            data = json.loads(r.read())
        if "resolved" in data:
            return bool(data["resolved"])
        nested = data.get(instance_id, {})
        if "resolved" in nested:
            return bool(nested["resolved"])
    except Exception:
        pass
    return None


def main():
    prefix = f"{SUB}/trajs/"
    all_keys = list_keys(prefix)
    txt_keys = [k for k in all_keys if k.endswith(".txt")]
    print(f"Found {len(txt_keys)} .txt traj files in {SUB}")

    # Resume
    done: set[str] = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                done.add(json.loads(line)["instance_id"])
            except Exception:
                pass
    txt_keys = [k for k in txt_keys if k[len(prefix) :].rsplit(".txt", 1)[0] not in done]
    print(f"  {len(done)} already done, {len(txt_keys)} remaining → {OUT.name}")

    n_ok = n_fail = 0
    started = time.time()

    with OUT.open("a") as f:
        for i, key in enumerate(txt_keys, 1):
            iid = key[len(prefix) :].rsplit(".txt", 1)[0]
            try:
                text = fetch_text(key)
                canon, native = parse_txt_trajectory(text)
                resolved = get_resolved(iid)
                rec = {
                    "instance_id": iid,
                    "n_steps": len(canon),
                    "resolved": resolved,
                    "atoms_canonical": canon,
                    "atoms_native": native,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_ok += 1
            except Exception:
                n_fail += 1
            if i % 50 == 0 or i == len(txt_keys):
                elapsed = time.time() - started
                print(f"  [{i:3d}/{len(txt_keys)}]  ok={n_ok}  fail={n_fail}  ({elapsed:.0f}s)")

    print(f"\nDone: {n_ok} ok, {n_fail} failed → {OUT}")


if __name__ == "__main__":
    main()
