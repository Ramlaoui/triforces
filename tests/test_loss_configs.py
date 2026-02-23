from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_all_loss_configs_instantiate() -> None:
    loss_dir = _repo_root() / "src" / "triforces" / "configs" / "loss"
    paths = sorted(loss_dir.glob("*.yaml"))
    assert paths, f"No loss configs found in {loss_dir}"

    for path in paths:
        loss_cfg = OmegaConf.load(path)
        instantiate(loss_cfg, _convert_="object")
