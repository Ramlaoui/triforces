import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from triforces.models.heads.simclr import ProjectionHead
from triforces.models.outputs import BackboneOutputs
from triforces.models.triforces import TriForcesModel


def _mean_pool(
    node_feats: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    out = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
    out.index_add_(0, batch, node_feats)
    count = torch.bincount(batch, minlength=num_graphs).clamp_min(1).to(out.dtype)
    return out / count.unsqueeze(1)


class DummyInteraction(nn.Module):
    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(120, embed_dim)

    def forward(self, batch, training=False, transform=None):
        if transform is not None:
            batch = transform(batch)
        z = batch.z
        batch_idx = batch.batch
        num_graphs = batch.num_graphs
        node_feats = self.embed(z)
        graph_feats = _mean_pool(node_feats, batch_idx, num_graphs)
        return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)


def _sample_batch() -> Batch:
    z = torch.tensor([1, 6, 8, 1, 6])
    pos = torch.randn(z.size(0), 3)
    batch = torch.tensor([0, 0, 0, 1, 1])
    data_list = [
        Data(z=z[batch == 0], pos=pos[batch == 0]),
        Data(z=z[batch == 1], pos=pos[batch == 1]),
    ]
    return Batch.from_data_list(data_list)


def test_triforces_model_uses_interaction_backbone_outputs():
    batch = _sample_batch()
    num_graphs = batch.num_graphs

    interaction = DummyInteraction(embed_dim=6)
    model = TriForcesModel(
        interaction=interaction,
        interaction_dim=6,
        enable_composition=False,
        enable_structural=False,
    )

    out = model(batch)

    assert out.node_feats.shape == (batch.z.size(0), 6)
    assert out.graph_feats.shape == (num_graphs, 6)
    assert torch.allclose(
        out.graph_feats, _mean_pool(out.node_feats, batch.batch, num_graphs)
    )
    assert "stream_node_feats" in out.extras


def test_triforces_model_preserves_interaction_graph_feats_in_extras():
    batch = _sample_batch()
    num_graphs = batch.num_graphs

    interaction = DummyInteraction(embed_dim=4)
    model = TriForcesModel(
        interaction=interaction,
        interaction_dim=4,
        enable_composition=False,
        enable_structural=False,
    )

    out = model(batch)

    expected = _mean_pool(out.node_feats, batch.batch, num_graphs)
    assert torch.allclose(out.graph_feats, expected)
    assert "stream_graph_feats" in out.extras
    assert "interaction" in out.extras["stream_graph_feats"]
    assert torch.allclose(out.extras["stream_graph_feats"]["interaction"], expected)


def test_projection_head_accepts_node_feats_from_model():
    batch = _sample_batch()
    num_graphs = batch.num_graphs

    interaction = DummyInteraction(embed_dim=7)
    model = TriForcesModel(
        interaction=interaction,
        interaction_dim=7,
        enable_composition=False,
        enable_structural=False,
    )

    out = model(batch)
    head = ProjectionHead(input_dim=7, node_projection_dim=3, graph_projection_dim=3)
    preds = head(out, batch)

    assert preds["node_feats"].shape == (batch.z.size(0), 7)
    assert preds["graph_features"].shape == (num_graphs, 7)
    assert preds["node_projections"].shape == (batch.z.size(0), 3)
    assert preds["graph_projections"].shape == (num_graphs, 3)


def test_triforces_structural_stream_requires_precomputed_edges():
    batch = _sample_batch()
    interaction = DummyInteraction(embed_dim=5)
    model = TriForcesModel(
        interaction=interaction,
        interaction_dim=5,
        enable_composition=False,
        enable_structural=True,
        structural_dim=4,
        num_radial=2,
        num_radial_out=2,
        l_max=1,
        structural_num_layers=1,
        use_lattice=False,
    )

    with pytest.raises(ValueError, match="edge_index"):
        model(batch)


def test_triforces_structural_stream_uses_batch_edges():
    data = Data(
        z=torch.tensor([1, 6], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_vec=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32),
        edge_dist=torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    batch = Batch.from_data_list([data])

    interaction = DummyInteraction(embed_dim=6)
    model = TriForcesModel(
        interaction=interaction,
        interaction_dim=6,
        enable_composition=False,
        enable_structural=True,
        structural_dim=4,
        num_radial=2,
        num_radial_out=2,
        l_max=1,
        structural_num_layers=1,
        use_lattice=False,
    )

    out = model(batch)
    assert out.node_feats.shape[0] == 2
    assert out.graph_feats.shape[0] == 1


def test_triforces_rejects_non_backbone_interaction_outputs():
    class BadInteraction(nn.Module):
        def forward(self, batch, training=False, transform=None):
            return torch.zeros((batch.z.size(0), 4), dtype=torch.float32)

    batch = _sample_batch()
    model = TriForcesModel(
        interaction=BadInteraction(),
        interaction_dim=4,
        enable_composition=False,
        enable_structural=False,
    )
    with pytest.raises(TypeError, match="BackboneOutputs"):
        model(batch)
