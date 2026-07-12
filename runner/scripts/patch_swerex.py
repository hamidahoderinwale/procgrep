"""Idempotent security patch: encrypted Modal tunnels for swe-rex 1.4.0.

swe-rex 1.4.0 hardcodes ``unencrypted_ports`` for the Modal sandbox tunnel, so
agent<->sandbox traffic crosses the public internet in plaintext. Flip to
``encrypted_ports`` (TLS). Same patch the rct harness applied; re-run after
every reinstall of the modal extra:

    uv run python scripts/patch_swerex.py
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    import swerex.deployment.modal as m

    path = Path(m.__file__)
    src = path.read_text()
    # Check unencrypted FIRST: the unpatched line contains the patched one as
    # a substring, so testing encrypted first would false-positive.
    if "unencrypted_ports=[self._port]" in src:
        path.write_text(
            src.replace("unencrypted_ports=[self._port]", "encrypted_ports=[self._port]")
        )
        print(f"patched unencrypted->encrypted: {path}")
    elif "encrypted_ports=[self._port]" in src:
        print(f"already encrypted: {path}")
    else:
        raise SystemExit(f"pattern not found (swe-rex version drift?): {path}")


if __name__ == "__main__":
    main()
