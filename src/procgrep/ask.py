"""Compile an English behavioral question into a spine regex.

The LLM runs only at query-authoring time; execution stays `grep`. The
compiler returns the regex plus its own paraphrase of what the regex
literally matches, so the user can check the compilation before trusting
counts. Questions the regex layer cannot express (temporal windows,
variable binding, probabilistic thresholds) are refused with a reason,
never approximated silently.

No SDK: one stdlib HTTP POST to the Anthropic Messages API, so the core
claim (procgrep never needs an LLM dependency) survives. The response is
schema-constrained via structured outputs, so parsing cannot drift.
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass

from procgrep.types import CANONICAL_ATOMS

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-opus-5"

# English<->regex pairs shipped in the explorer chips; they double as the
# compiler's few-shots and as the gold set for eval.
GOLD_PAIRS: list[tuple[str, str]] = [
    ("submitted without ever running a test", "^(?:(?!run_test).)*submit"),
    ("never searched the repo", "^(?:(?!search_repo).)*$"),
    ("stuck in a reading loop", "(read_file (?:think )?){4,}"),
    ("an edit streak of five or more", "(edit (?:think |other )?){5,}"),
    ("recovered from an error by editing", "error (?:think |other )?edit"),
    ("ran the canonical resolve loop", "search_repo read_file edit run_test"),
    ("wrote a repro script before submitting", "create_file.*submit"),
    ("ran a test before its first edit", "^(?:(?!edit).)*run_test"),
]

_INEXPRESSIBLE_EXAMPLE = (
    "edited the same file twice within five steps",
    "needs variable binding (same file) and a step window; the spine regex"
    " layer has neither. Nearest expressible: an edit streak, (edit ){2,}.",
)

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "expressible": {"type": "boolean"},
        "regex": {"type": ["string", "null"]},
        "paraphrase": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["expressible", "regex", "paraphrase", "reason"],
    "additionalProperties": False,
}


def _system_prompt() -> str:
    shots = "\n".join(f'- "{q}" -> {r}' for q, r in GOLD_PAIRS)
    return (
        "You compile English questions about coding-agent behavior into a"
        " Python regex over a trace's space-joined atom sequence.\n\n"
        f"The atom alphabet: {', '.join(sorted(CANONICAL_ATOMS))}.\n\n"
        "Encoding contract: a trace is ' '.join(atoms) plus one trailing"
        " space, so atoms are space-separated words and a right-anchored"
        " match on the last atom works. The raw spine keeps think/other"
        " interleaved between actions, so adjacent-action idioms usually"
        " need optional skips like (?:think |other )?.\n\n"
        "Idioms:\n"
        "- streaks: (edit (?:think |other )?){5,}\n"
        "- absence before an event: ^(?:(?!run_test).)*submit\n"
        "- absence everywhere: ^(?:(?!search_repo).)*$\n"
        "- ordering: a match position implies sequence order.\n\n"
        f"Examples:\n{shots}\n\n"
        "The regex layer CANNOT express temporal windows (within N steps),"
        " variable binding (the same file twice), counts across gaps, or"
        " probabilistic thresholds. When a question needs one of these,"
        " set expressible to false, explain why in reason, and suggest the"
        " nearest expressible query there. Example: "
        f'"{_INEXPRESSIBLE_EXAMPLE[0]}" -> {_INEXPRESSIBLE_EXAMPLE[1]}\n\n'
        "When expressible, return the regex and a one-line paraphrase of"
        " what the regex LITERALLY matches (not what was asked), so the"
        " user can spot a mistranslation."
    )


class AskError(RuntimeError):
    """The compile call failed: HTTP error, refusal, or an invalid regex."""


@dataclass(frozen=True)
class CompiledQuery:
    question: str
    expressible: bool
    regex: str | None
    paraphrase: str | None
    reason: str | None
    model: str


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats()["x509_ca"]:
        # Homebrew-OpenSSL Pythons can ship an empty CA store; certifi is
        # the standard fix and rides in via existing deps when present.
        try:
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError as exc:
            raise AskError(
                "the system CA store is empty and certifi is not installed;"
                " cannot verify TLS to the Anthropic API"
            ) from exc
    return ctx


def _post_messages(payload: dict[str, object], api_key: str, timeout: float) -> dict[str, object]:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            body: dict[str, object] = json.loads(resp.read())
            return body
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise AskError(f"Anthropic API returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AskError(f"could not reach the Anthropic API: {exc.reason}") from exc


def compile_query(
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> CompiledQuery:
    """Compile ``question`` to a spine regex via one Messages API call.

    Reads ``ANTHROPIC_API_KEY`` when ``api_key`` is not given. Raises
    :class:`AskError` on transport errors, refusals, or a regex that does
    not compile.
    """
    import os

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise AskError("set ANTHROPIC_API_KEY (or pass api_key=) to use ask")

    payload: dict[str, object] = {
        "model": model,
        "max_tokens": 16000,
        "system": _system_prompt(),
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA},
        },
        "messages": [{"role": "user", "content": question}],
    }
    body = _post_messages(payload, key, timeout)

    if body.get("stop_reason") == "refusal":
        raise AskError("the model declined to compile this question")
    content = body.get("content")
    if not isinstance(content, list):
        raise AskError(f"unexpected response shape: {str(body)[:200]}")
    text = next(
        (b.get("text") for b in content if isinstance(b, dict) and b.get("type") == "text"),
        None,
    )
    if not isinstance(text, str):
        raise AskError("response carried no text block")
    parsed = json.loads(text)

    regex = parsed.get("regex")
    if parsed["expressible"]:
        if not isinstance(regex, str):
            raise AskError("model marked the question expressible but returned no regex")
        try:
            re.compile(regex)
        except re.error as exc:
            raise AskError(
                f"model returned a regex that does not compile: {regex!r} ({exc})"
            ) from exc

    return CompiledQuery(
        question=question,
        expressible=bool(parsed["expressible"]),
        regex=regex if isinstance(regex, str) else None,
        paraphrase=parsed.get("paraphrase"),
        reason=parsed.get("reason"),
        model=model,
    )
