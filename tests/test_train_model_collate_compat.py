import pytest
from omegaconf import OmegaConf

from triforces.train import _validate_model_collate_compat


def _cfg(interaction_target: str, graph_target: str):
    return OmegaConf.create(
        {
            "model": {
                "_target_": "triforces.models.adapter_model.AdapterModel",
                "backbone": {
                    "_target_": "triforces.models.triforces.TriForcesModel",
                    "interaction": {"_target_": interaction_target},
                },
            },
            "collate": {
                "_target_": "triforces.data.pyg_collate",
                "_partial_": True,
                "graph": {"_target_": graph_target},
            },
        }
    )


def test_orb_requires_orb_graph_collate():
    cfg = _cfg(
        "triforces.models.interaction.orb.Orb",
        "triforces.models.interaction.orb.orb_graph",
    )
    _validate_model_collate_compat(cfg)


def test_mace_rejects_orb_graph_collate():
    cfg = _cfg(
        "triforces.models.interaction.mace.MACE",
        "triforces.models.interaction.orb.orb_graph",
    )
    with pytest.raises(ValueError):
        _validate_model_collate_compat(cfg)
