from __future__ import annotations

import importlib.util

import pytest
from hydra import compose, initialize_config_module
from hydra.utils import instantiate


def _has_modules(*module_names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in module_names)


def test_train_configs_model_instantiation() -> None:
    cases = [
        ("train_contrastive", [], ("fairchem",)),
        ("train_supervised", [], ("fairchem",)),
        ("train_contrastive", ["model=orb/triforces"], ("orb_models",)),
        ("train_contrastive", ["model=mace/triforces"], ("mace",)),
        ("train_contrastive", ["model=orb/triforces_barlow"], ("orb_models",)),
        ("train_contrastive_orb_barlow", [], ("orb_models",)),
        ("experiments/pretraining/orb/main_triforces", [], ("orb_models",)),
        ("experiments/pretraining/esen/main_triforces", [], ("fairchem",)),
        ("experiments/pretraining/mace/main_triforces", [], ("mace",)),
        ("experiments/supervised/orb/direct", [], ("orb_models",)),
        ("experiments/supervised/orb/energy_conserving", [], ("orb_models",)),
    ]
    instantiated_cases = 0

    with initialize_config_module(config_module="triforces.configs", version_base=None):
        for config_name, overrides, required_modules in cases:
            if not _has_modules(*required_modules):
                continue
            cfg = compose(config_name=config_name, overrides=overrides)
            model = instantiate(cfg.model, _convert_="object")
            assert model.__class__.__name__ == "AdapterModel"
            instantiated_cases += 1

    if instantiated_cases == 0:
        pytest.skip("No optional interaction backends are installed.")


def test_head_group_swaps_instantiate_for_orb_barlow() -> None:
    if not _has_modules("orb_models"):
        pytest.skip("ORB backend is not installed.")

    head_options = [
        "contrastive/simclr",
        "contrastive/barlow_twins",
        "contrastive/split_barlow_twins",
        "contrastive/byol/projection",
        "contrastive/byol/combined",
        "contrastive/ibot/projection",
    ]

    with initialize_config_module(config_module="triforces.configs", version_base=None):
        for option in head_options:
            cfg = compose(
                config_name="train_contrastive_orb_barlow",
                overrides=[f"head@model.heads.proj={option}"],
            )
            instantiate(cfg.model, _convert_="object")
