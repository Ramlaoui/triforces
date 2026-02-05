from types import SimpleNamespace

import torch

from triforces.losses.lejepa import LeJEPALoss


def _make_data():
    return SimpleNamespace(
        pair_id=torch.tensor([0, 0], dtype=torch.long),
        pair_idx1=torch.tensor([0], dtype=torch.long),
        pair_idx2=torch.tensor([1], dtype=torch.long),
        node_pair_idx1=torch.tensor([0], dtype=torch.long),
        node_pair_idx2=torch.tensor([1], dtype=torch.long),
        batch=torch.tensor([0, 1], dtype=torch.long),
    )


def test_lejepa_prediction_loss_graph_only():
    data = _make_data()
    embeddings = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

    loss_fn = LeJEPALoss(
        prediction_weight=1.0,
        sigreg_weight=0.0,
        lambda_graph=1.0,
        lambda_node=0.0,
    )

    loss, metrics = loss_fn(data, {"graph_projections": embeddings})

    assert torch.isclose(loss, torch.tensor(1.0))
    assert metrics["loss/prediction"] == 1.0
    assert metrics["loss/sigreg"] == 0.0


def test_lejepa_prediction_loss_node_only():
    data = _make_data()
    node_embeddings = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

    loss_fn = LeJEPALoss(
        prediction_weight=1.0,
        sigreg_weight=0.0,
        lambda_graph=0.0,
        lambda_node=1.0,
    )

    loss, metrics = loss_fn(data, {"node_projections": node_embeddings})

    assert torch.isclose(loss, torch.tensor(1.0))
    assert metrics["loss/prediction"] == 1.0
    assert metrics["loss/sigreg"] == 0.0
