"""Tests for the live watch module and its CLI wiring."""

from __future__ import annotations

import argparse
import threading

from typer.testing import CliRunner

from procgrep.cli import app
from procgrep.watch import DEMO_ATOMS, PAGE, _Bus, _producer


def test_demo_producer_streams_all_atoms_then_done() -> None:
    bus = _Bus()
    args = argparse.Namespace(tail=None, demo=True, interval=0.0)
    t = threading.Thread(target=_producer, args=(bus, args))
    t.start()
    t.join(timeout=5)
    events = []
    while not bus.q.empty():
        events.append(bus.q.get_nowait())
    atoms = [e["atom"] for e in events if "atom" in e]
    assert atoms == DEMO_ATOMS
    assert events[-1] == {"done": True}


def test_page_is_selfcontained_and_wires_the_stream() -> None:
    assert "EventSource" in PAGE
    assert "/events" in PAGE
    assert "<script src=" not in PAGE  # no external dependencies


def test_cli_watch_help() -> None:
    result = CliRunner().invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    assert "live" in result.output
