"""The sealed manifest: completeness, hash verification, no-overwrite."""

from __future__ import annotations

import pytest
from helpers import READ_LOOP_SPEC, scripted_config
from procgrep_runner.manifest import read_manifest, sha256_path
from procgrep_runner.run import prepare_run

INSTANCES = [
    {"instance_id": "demo__demo-1", "problem_statement": "fix a"},
    {"instance_id": "demo__demo-2", "problem_statement": "fix b"},
]


def prepared(tmp_path):
    run_dir = tmp_path / "run_v1"
    manifest = prepare_run(
        run_dir,
        spec=READ_LOOP_SPEC,
        instances=INSTANCES,
        config=scripted_config(),
        mode="guard",
        on_violation="block",
        replicates=2,
        seed=7,
        subset="verified",
        split="test",
    )
    return run_dir, manifest


def test_manifest_pins_everything_the_measurement_needs(tmp_path):
    run_dir, manifest = prepared(tmp_path)

    for name in ("manifest.json", "spec.yaml", "instances.jsonl", "config.json"):
        assert (run_dir / name).exists()
    assert manifest.run_id == "run_v1"
    assert manifest.spec_name == "no_read_loops"
    assert manifest.spec_sha256 == sha256_path(run_dir / "spec.yaml")
    assert manifest.mode == "guard"
    assert manifest.on_violation == "block"
    assert manifest.arms == ("baseline", "enforced")
    assert manifest.instance_ids == ("demo__demo-1", "demo__demo-2")
    assert manifest.replicates == 2
    assert manifest.seed == 7
    assert manifest.procgrep_version not in ("", None)
    assert manifest.mini_swe_agent_version not in ("", None)
    assert read_manifest(run_dir) == manifest


def test_tampered_spec_fails_hash_verification(tmp_path):
    run_dir, _ = prepared(tmp_path)
    spec_path = run_dir / "spec.yaml"
    spec_path.write_text(spec_path.read_text() + "\n# tampered\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        read_manifest(run_dir)
    assert read_manifest(run_dir, verify_spec_hash=False).run_id == "run_v1"


def test_run_dir_is_never_overwritten(tmp_path):
    prepared(tmp_path)
    with pytest.raises(FileExistsError):
        prepared(tmp_path)


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown enforcement mode"):
        prepare_run(
            tmp_path / "bad",
            spec=READ_LOOP_SPEC,
            instances=INSTANCES,
            config=scripted_config(),
            mode="decode",
        )
