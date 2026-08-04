"""Local client discovery: candidate layout, variant globbing, explicit override."""

from __future__ import annotations

import procgrep.ingest.clients as clients_mod
from procgrep.ingest.clients import LocalClient, discover_clients, find_client


def _make_cursor_store(root, variant="Cursor", size=1234):
    """A minimal Cursor-shaped tree under ``root``, returning the store path."""
    db = root / variant / "User" / "globalStorage" / "state.vscdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"x" * size)
    return db


def test_finds_cursor_under_app_data_root(tmp_path, monkeypatch):
    db = _make_cursor_store(tmp_path)
    monkeypatch.setattr(clients_mod, "_app_data_roots", lambda: [tmp_path])
    found = discover_clients("cursor")
    assert [c.path for c in found] == [db]
    assert found[0].adapter == "cursor-vscdb"


def test_finds_cursor_variants_by_glob(tmp_path, monkeypatch):
    """A channel this code has never heard of is still found, largest first."""
    _make_cursor_store(tmp_path, "Cursor", size=100)
    _make_cursor_store(tmp_path, "Cursor Nightly", size=9000)
    monkeypatch.setattr(clients_mod, "_app_data_roots", lambda: [tmp_path])
    assert [c.name for c in discover_clients("cursor")] == ["Cursor Nightly", "Cursor"]


def test_ignores_cursor_named_dir_without_a_store(tmp_path, monkeypatch):
    """``cursor-pkl-extension`` and friends match the glob but hold no sessions."""
    (tmp_path / "cursor-pkl-extension").mkdir()
    monkeypatch.setattr(clients_mod, "_app_data_roots", lambda: [tmp_path])
    assert discover_clients("cursor") == []


def test_unknown_family_is_empty_not_an_error():
    assert discover_clients("emacs") == []


def test_explicit_path_short_circuits_discovery(tmp_path, monkeypatch):
    db = _make_cursor_store(tmp_path / "elsewhere")
    monkeypatch.setattr(clients_mod, "_app_data_roots", list)
    client = find_client("cursor", path=db)
    assert client is not None
    assert client.path == db


def test_explicit_missing_path_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(clients_mod, "_app_data_roots", list)
    assert find_client("cursor", path=tmp_path / "nope.vscdb") is None


def test_find_client_none_when_nothing_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(clients_mod, "_app_data_roots", lambda: [tmp_path])
    monkeypatch.setattr(clients_mod, "_claude_code_clients", list)
    assert find_client("cursor") is None


def test_size_label_scales():
    def label(n):
        return LocalClient("c", "cursor-vscdb", clients_mod.Path("/x"), n).size_label

    assert label(512) == "512 B"
    assert label(2048) == "2.0 KB"
    assert label(16_022_675_456) == "14.9 GB"
