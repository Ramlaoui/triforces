from __future__ import annotations

import hydra
from omegaconf import DictConfig

from triforces.train import run

ALLOWED_DATASETS = ("cif", "atompack", "bulk", "lemat_bulk")

__all__ = ["ALLOWED_DATASETS", "validate_config", "main"]


def _is_augmentation_wrapper_target(target: str) -> bool:
    return target.endswith("ContrastiveDataset") or target.endswith(
        "AugmentationDataset"
    )


def _dataset_name_from_cfg(cfg: DictConfig) -> str | None:
    dataset = cfg.get("dataset")

    if isinstance(dataset, str):
        return dataset

    if isinstance(dataset, DictConfig):
        target = str(dataset.get("_target_", "") or "")
        if _is_augmentation_wrapper_target(target):
            base = dataset.get("dataset")
            if isinstance(base, DictConfig):
                target = str(base.get("_target_", "") or "")

        if target.endswith("CifFolderDataset"):
            return "cif"
        if target.endswith("AtompackDataset"):
            return "atompack"
        if target.endswith("LeMatBulkDataset"):
            return "lemat_bulk"

    return None


def _base_dataset_cfg(cfg: DictConfig) -> DictConfig | None:
    dataset = cfg.get("dataset")
    if not isinstance(dataset, DictConfig):
        return None
    target = str(dataset.get("_target_", "") or "")
    if _is_augmentation_wrapper_target(target):
        nested = dataset.get("dataset")
        if isinstance(nested, DictConfig):
            return nested
    return dataset


def _has_dataset_path(cfg: DictConfig, dataset_name: str) -> bool:
    dataset_cfg = _base_dataset_cfg(cfg)
    if dataset_name == "cif":
        root = dataset_cfg.get("root") if isinstance(dataset_cfg, DictConfig) else None
        return bool(root)
    if dataset_name == "atompack":
        path = dataset_cfg.get("path") if isinstance(dataset_cfg, DictConfig) else None
        return bool(path)
    return True


def validate_config(cfg: DictConfig) -> None:
    dataset_name = _dataset_name_from_cfg(cfg)
    if not dataset_name:
        raise ValueError(
            "Missing `dataset`. Expected one of: " + ", ".join(ALLOWED_DATASETS)
        )
    if dataset_name not in ALLOWED_DATASETS:
        raise ValueError(
            f"Invalid dataset={dataset_name!r}. Allowed: " + ", ".join(ALLOWED_DATASETS)
        )

    if cfg.get("model") is None:
        raise ValueError("Missing required config key: model")

    if not _has_dataset_path(cfg, dataset_name):
        required = "dataset.root" if dataset_name == "cif" else "dataset.path"
        raise ValueError(
            f"Missing required config key: {required} "
            f"(required for dataset={dataset_name!r})"
        )


@hydra.main(
    version_base=None,
    config_path="pkg://triforces/configs",
    config_name="train_contrastive",
)
def main(cfg: DictConfig) -> int:
    validate_config(cfg)
    return run(cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
