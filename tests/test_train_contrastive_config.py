import pytest
from omegaconf import OmegaConf

from triforces.cli.train_contrastive import ALLOWED_DATASETS, validate_config


def make_cfg(**overrides):
    base = {
        "dataset": "cif",
        "model": "triforces_esen",
        "data_path": "/tmp/data",
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
    del cfg["data_path"]
    with pytest.raises(ValueError) as excinfo:
        validate_config(cfg)
    assert "data_path" in str(excinfo.value)


def test_validate_config_allows_lemat_bulk_without_data_path():
    cfg = make_cfg(dataset="lemat_bulk")
    del cfg["data_path"]
    validate_config(cfg)
