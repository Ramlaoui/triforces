import pytest
from omegaconf import OmegaConf

from triforces.cli.train_contrastive import ALLOWED_DATASETS, validate_config


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


def test_validate_config_accepts_defaults():
    cfg = make_cfg()
    validate_config(cfg)


def test_validate_config_rejects_dataset():
    cfg = make_cfg(dataset="nope")
    with pytest.raises(ValueError) as excinfo:
        validate_config(cfg)
    assert "dataset" in str(excinfo.value)
    for name in ALLOWED_DATASETS:
        assert name in str(excinfo.value)


def test_validate_config_rejects_missing_model():
    cfg = make_cfg()
    del cfg["model"]
    with pytest.raises(ValueError) as excinfo:
        validate_config(cfg)
    assert "model" in str(excinfo.value)


def test_validate_config_rejects_missing_data_path():
    cfg = make_cfg()
    del cfg["dataset"]["root"]
    with pytest.raises(ValueError) as excinfo:
        validate_config(cfg)
    assert "dataset.root" in str(excinfo.value)


def test_validate_config_allows_lemat_bulk_without_data_path():
    cfg = make_cfg(
        dataset={
            "_target_": "triforces.data.lemat_bulk.LeMatBulkDataset",
            "name": "compatible_pbe",
            "split": "train",
        }
    )
    validate_config(cfg)


def test_validate_config_allows_asedb_with_path():
    cfg = make_cfg(
        dataset={
            "_target_": "triforces.data.asedb_dataset.ASEDBDataset",
            "path": "/tmp/data/train.db",
        }
    )
    validate_config(cfg)


def test_validate_config_allows_asedb_with_repo_id_only():
    cfg = make_cfg(
        dataset={
            "_target_": "triforces.data.asedb_dataset.ASEDBDataset",
            "repo_id": "Org/repo",
        }
    )
    validate_config(cfg)


def test_validate_config_rejects_asedb_without_path_or_repo_id():
    cfg = make_cfg(
        dataset={
            "_target_": "triforces.data.asedb_dataset.ASEDBDataset",
        }
    )
    with pytest.raises(ValueError) as excinfo:
        validate_config(cfg)
    assert "dataset.path or dataset.repo_id" in str(excinfo.value)


def test_validate_config_requires_explicit_dataset_node():
    cfg = OmegaConf.create(
        {
            "hydra": {"runtime": {"choices": {"dataset": "cif"}}},
            "model": {"_target_": "triforces.models.adapter_model.AdapterModel"},
        }
    )
    with pytest.raises(ValueError) as excinfo:
        validate_config(cfg)
    assert "Missing `dataset`" in str(excinfo.value)
