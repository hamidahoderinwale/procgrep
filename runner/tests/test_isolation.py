"""Crash isolation: one failing task never takes down the batch."""

from __future__ import annotations

import sys

from helpers import READ_LOOP_SPEC, scripted_config
from procgrep_runner.run import run_instance_subprocess, run_paired

INSTANCES = [{"instance_id": f"demo__demo-{k}", "problem_statement": "fix it"} for k in (1, 2, 3)]


def test_one_crashing_instance_does_not_kill_the_batch(tmp_path):
    def runner(run_dir, instance_id, arm, replicate):
        if instance_id == "demo__demo-2":
            raise ValueError("boom")
        return {
            "instance_id": instance_id,
            "arm": arm,
            "replicate": replicate,
            "exit_status": "Submitted",
            "submitted": True,
        }

    rows = run_paired(
        spec=READ_LOOP_SPEC,
        instances=INSTANCES,
        config=scripted_config(),
        run_dir=tmp_path / "run_v1",
        mode="guard",
        instance_runner=runner,
    )

    assert len(rows) == 6
    crashed = [r for r in rows if r["instance_id"] == "demo__demo-2"]
    assert [r["exit_status"] for r in crashed] == ["ValueError", "ValueError"]
    survived = [r for r in rows if r["instance_id"] != "demo__demo-2"]
    assert all(r["exit_status"] == "Submitted" for r in survived)
    assert (tmp_path / "run_v1" / "rows.jsonl").read_text().count("\n") == 6


def argv(code: str):
    return lambda run_dir, instance_id, arm, replicate: [sys.executable, "-c", code]


def test_subprocess_row_is_parsed_from_the_last_json_line(tmp_path):
    row = run_instance_subprocess(
        tmp_path,
        "demo__demo-1",
        "baseline",
        0,
        argv_builder=argv(
            'print("log noise"); print(\'{"instance_id": "demo__demo-1", "exit_status": "Submitted"}\')'
        ),
    )
    assert row == {"instance_id": "demo__demo-1", "exit_status": "Submitted"}


def test_subprocess_crash_degrades_to_a_row(tmp_path):
    row = run_instance_subprocess(
        tmp_path, "demo__demo-1", "baseline", 0, argv_builder=argv("import sys; sys.exit(3)")
    )
    assert row["exit_status"] == "RunnerSubprocessError"
    assert row["returncode"] == 3


def test_subprocess_timeout_degrades_to_a_row(tmp_path):
    row = run_instance_subprocess(
        tmp_path,
        "demo__demo-1",
        "baseline",
        0,
        timeout=0.5,
        argv_builder=argv("import time; time.sleep(30)"),
    )
    assert row["exit_status"] == "RunnerTimeout"


def test_garbled_subprocess_output_degrades_to_a_row(tmp_path):
    row = run_instance_subprocess(
        tmp_path, "demo__demo-1", "baseline", 0, argv_builder=argv('print("not json")')
    )
    assert row["exit_status"] == "RunnerProtocolError"
