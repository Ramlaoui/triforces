from __future__ import annotations

from pathlib import Path

import pytest

import triforces.data.atompack_dataset as atompack_module
from triforces.data.atompack_dataset import AtompackDataset


def test_from_huggingface_file_path(monkeypatch, tmp_path: Path):
    target = tmp_path / "export" / "data.atp"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    calls = {"download": 0}

    class FakeHF:
        @staticmethod
        def hf_hub_download(**kwargs):
            calls["download"] += 1
            assert kwargs["filename"] == "export/data.atp"
            return str(target)

        @staticmethod
        def snapshot_download(**kwargs):
            raise AssertionError(
                "snapshot_download should not be called for .atp file path"
            )

    monkeypatch.setattr(atompack_module, "_require_hf_hub", lambda: FakeHF)

    captured = {}

    def fake_init(self, path=None, **kwargs):
        captured["path"] = Path(path)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(AtompackDataset, "__init__", fake_init)
    AtompackDataset.from_huggingface(repo_id="Org/repo", path_in_repo="export/data.atp")

    assert calls["download"] == 1
    assert captured["path"] == target
    assert captured["kwargs"]["use_mmap"] is True


def test_from_huggingface_directory_path(monkeypatch, tmp_path: Path):
    snapshot_root = tmp_path / "snapshot"
    data_dir = snapshot_root / "exports" / "v1"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "part_0000.atp").touch()
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
                "exports/v1/*.atp",
                "exports/v1/**/*.atp",
            ]
            return str(snapshot_root)

    monkeypatch.setattr(atompack_module, "_require_hf_hub", lambda: FakeHF)

    captured = {}

    def fake_init(self, path=None, **kwargs):
        captured["path"] = Path(path)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(AtompackDataset, "__init__", fake_init)
    AtompackDataset.from_huggingface(repo_id="Org/repo", path_in_repo="exports/v1")

    assert calls["snapshot"] == 1
    assert captured["path"] == data_dir
    assert captured["kwargs"]["use_mmap"] is True


def test_from_huggingface_directory_missing_path_raises(monkeypatch, tmp_path: Path):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    class FakeHF:
        @staticmethod
        def snapshot_download(**kwargs):
            return str(snapshot_root)

    monkeypatch.setattr(atompack_module, "_require_hf_hub", lambda: FakeHF)

    with pytest.raises(FileNotFoundError):
        AtompackDataset.from_huggingface(
            repo_id="Org/repo",
            path_in_repo="exports/missing",
        )


def test_from_huggingface_without_path_in_repo_with_single_atp(
    monkeypatch, tmp_path: Path
):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    only_file = snapshot_root / "data.atp"
    only_file.touch()

    class FakeHF:
        @staticmethod
        def snapshot_download(**kwargs):
            assert kwargs["allow_patterns"] == ["*.atp", "**/*.atp"]
            return str(snapshot_root)

    monkeypatch.setattr(atompack_module, "_require_hf_hub", lambda: FakeHF)

    captured = {}

    def fake_init(self, path=None, **kwargs):
        captured["path"] = Path(path)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(AtompackDataset, "__init__", fake_init)
    AtompackDataset.from_huggingface(repo_id="Org/repo")

    assert captured["path"] == only_file
    assert captured["kwargs"]["use_mmap"] is True


def test_from_huggingface_without_path_in_repo_raises_on_multiple_atp(
    monkeypatch, tmp_path: Path
):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    (snapshot_root / "a.atp").touch()
    (snapshot_root / "b.atp").touch()

    class FakeHF:
        @staticmethod
        def snapshot_download(**kwargs):
            return str(snapshot_root)

    monkeypatch.setattr(atompack_module, "_require_hf_hub", lambda: FakeHF)

    with pytest.raises(ValueError, match="Multiple \\.atp files"):
        AtompackDataset.from_huggingface(repo_id="Org/repo")
