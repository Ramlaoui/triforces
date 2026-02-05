from __future__ import annotations

import hydra
from omegaconf import DictConfig

from triforces.train.contrastive import run

__all__ = ["main"]


@hydra.main(
    version_base=None,
    config_path="pkg://triforces/configs",
    config_name="train_contrastive",
)
def main(cfg: DictConfig) -> int:
    return run(cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
