from __future__ import annotations

from pathlib import Path

import pytest

import triforces.data.asedb_dataset as asedb_module
from triforces.data.asedb_dataset import ASEDBDataset


def test_from_huggingface_file_path(monkeypatch, tmp_path: Path):
    target = tmp_path / "export" / "train.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    calls = {"download": 0}

    class FakeHF:
        @staticmethod
        def hf_hub_download(**kwargs):
            calls["download"] += 1
            assert kwargs["filename"] == "export/train.db"
            return str(target)

        @staticmethod
        def snapshot_download(**kwargs):
            raise AssertionError(
                "snapshot_download should not be called for .db file path"
            )

    monkeypatch.setattr(asedb_module, "_require_hf_hub", lambda: FakeHF)

    captured = {}

    def fake_init(self, path=None, **kwargs):
        captured["path"] = Path(path)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ASEDBDataset, "__init__", fake_init)
    ASEDBDataset.from_huggingface(repo_id="Org/repo", path_in_repo="export/train.db")

    assert calls["download"] == 1
    assert captured["path"] == target
    assert captured["kwargs"]["keep_db_open"] is True


def test_from_huggingface_directory_path(monkeypatch, tmp_path: Path):
    snapshot_root = tmp_path / "snapshot"
    data_dir = snapshot_root / "exports" / "v1"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "part_0000.db").touch()
    calls = {"snapshot": 0}

    class FakeHF:
        @staticmethod
        def hf_hub_download(**kwargs):
            raise AssertionError(
                "hf_hub_download should not be called for directory path"
            )

        @staticmethod
        def snapshot_download(**kwargs):
            calls["snapshot"] += 1
            assert kwargs["allow_patterns"] == [
                "exports/v1/*.db",
                "exports/v1/**/*.db",
                "exports/v1/*.aselmdb",
                "exports/v1/**/*.aselmdb",
            ]
            return str(snapshot_root)

    monkeypatch.setattr(asedb_module, "_require_hf_hub", lambda: FakeHF)

    captured = {}

    def fake_init(self, path=None, **kwargs):
        captured["path"] = Path(path)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ASEDBDataset, "__init__", fake_init)
    ASEDBDataset.from_huggingface(repo_id="Org/repo", path_in_repo="exports/v1")

    assert calls["snapshot"] == 1
    assert captured["path"] == data_dir
    assert captured["kwargs"]["keep_db_open"] is True


def test_from_huggingface_directory_missing_path_raises(monkeypatch, tmp_path: Path):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    class FakeHF:
        @staticmethod
        def snapshot_download(**kwargs):
            return str(snapshot_root)

    monkeypatch.setattr(asedb_module, "_require_hf_hub", lambda: FakeHF)

    with pytest.raises(FileNotFoundError):
        ASEDBDataset.from_huggingface(
            repo_id="Org/repo",
            path_in_repo="exports/missing",
        )


def test_from_huggingface_without_path_in_repo_with_files(monkeypatch, tmp_path: Path):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    (snapshot_root / "train.db").touch()
    (snapshot_root / "val.aselmdb").touch()

    class FakeHF:
        @staticmethod
        def snapshot_download(**kwargs):
            assert kwargs["allow_patterns"] == [
                "*.db",
                "**/*.db",
                "*.aselmdb",
                "**/*.aselmdb",
            ]
            return str(snapshot_root)

    monkeypatch.setattr(asedb_module, "_require_hf_hub", lambda: FakeHF)

    captured = {}

    def fake_init(self, path=None, **kwargs):
        captured["path"] = Path(path)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ASEDBDataset, "__init__", fake_init)
    ASEDBDataset.from_huggingface(repo_id="Org/repo")

    assert captured["path"] == snapshot_root
    assert captured["kwargs"]["keep_db_open"] is True


def test_from_huggingface_without_path_in_repo_raises_when_no_db(
    monkeypatch, tmp_path: Path
):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    class FakeHF:
        @staticmethod
        def snapshot_download(**kwargs):
            return str(snapshot_root)

    monkeypatch.setattr(asedb_module, "_require_hf_hub", lambda: FakeHF)

    with pytest.raises(FileNotFoundError, match="No ASE DB files found"):
        ASEDBDataset.from_huggingface(repo_id="Org/repo")
