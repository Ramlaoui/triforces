from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from triforces.train import _resolve_checkpoint_dir


def test_resolve_checkpoint_dir_defaults_to_cwd_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint_cfg = OmegaConf.create({"dir": None})

    resolved = _resolve_checkpoint_dir(checkpoint_cfg)

    assert resolved == (tmp_path / "checkpoints").resolve()


def test_resolve_checkpoint_dir_uses_explicit_dir(tmp_path: Path) -> None:
    custom_dir = tmp_path / "my_ckpts"
    checkpoint_cfg = OmegaConf.create({"dir": str(custom_dir)})

    resolved = _resolve_checkpoint_dir(checkpoint_cfg)

    assert resolved == custom_dir.resolve()


def test_resolve_checkpoint_dir_uses_run_name_subdir_when_dir_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint_cfg = OmegaConf.create({"dir": None})

    resolved = _resolve_checkpoint_dir(checkpoint_cfg, run_name="orb pretrain / run #1")

    assert resolved == (tmp_path / "checkpoints" / "orb_pretrain_run_1").resolve()
