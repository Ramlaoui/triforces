from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from triforces.augmentations import AtomMasking
from triforces.evaluation.linear_probe import _graph_embeddings_from_outputs
from triforces.losses.simsiam import SimSiamLoss
from triforces.models import graph_build


def test_simsiam_requires_data_and_preds_interface() -> None:
    loss = SimSiamLoss()
    with pytest.raises(ValueError, match="expects `data` and `preds`"):
        loss()


def test_linear_probe_does_not_fallback_to_graph_feats() -> None:
    outputs_dict = {"graph_feats": torch.randn(4, 8)}
    assert _graph_embeddings_from_outputs(outputs_dict, "graph_projections") is None

    outputs_obj = SimpleNamespace(graph_feats=torch.randn(4, 8))
    assert _graph_embeddings_from_outputs(outputs_obj, "graph_projections") is None


def test_radius_graph_propagates_backend_errors(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("radius backend failure")

    monkeypatch.setattr(graph_build, "pyg_radius_graph", _boom)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="radius backend failure"):
        graph_build.radius_graph(pos=pos, batch=None, r=1.0)


def test_atom_masking_requires_configured_mask_prob() -> None:
    masking = AtomMasking(mask_prob=None)
    with pytest.raises(ValueError, match="mask_prob must be configured"):
        masking._sample_mask_probability()
