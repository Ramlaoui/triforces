import pytest
from omegaconf import OmegaConf

from triforces.cli.train_contrastive import validate_config


def make_cfg(**overrides):
    base = {
        "dataset": {
            "_target_": "triforces.data.ase_contrastive.CifFolderDataset",
            "root": "/tmp/data",
            "glob": "**/*.cif",
        },
        "model": {"_target_": "triforces.models.adapter_model.AdapterModel"},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_validate_config_allows_atompack_with_repo_id_only():
    cfg = make_cfg(
        dataset={
            "_target_": "triforces.data.atompack_dataset.AtompackDataset",
            "repo_id": "Org/repo",
        }
    )
    validate_config(cfg)


def test_validate_config_rejects_atompack_without_path_or_repo():
    cfg = make_cfg(
        dataset={
            "_target_": "triforces.data.atompack_dataset.AtompackDataset",
        }
    )
    with pytest.raises(ValueError) as excinfo:
        validate_config(cfg)
    assert "dataset.path or dataset.repo_id" in str(excinfo.value)
