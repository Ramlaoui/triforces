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
    def __init__(
        self,
        embed_dim: int = 8,
        *,
        return_type: str = "dict",
        include_graph: bool = False,
        use_features_key: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(120, embed_dim)
        self.return_type = return_type
        self.include_graph = include_graph
        self.use_features_key = use_features_key

    def forward(self, batch, training=False, transform=None):
        if transform is not None:
            batch = transform(batch)
        z = batch.z
        batch_idx = batch.batch
        num_graphs = batch.num_graphs
        node_feats = self.embed(z)
        graph_feats = None
        if self.include_graph:
            graph_feats = _mean_pool(node_feats, batch_idx, num_graphs)

        if self.return_type == "tensor":
            return node_feats
        if self.return_type == "tuple":
            return node_feats, graph_feats
        if self.return_type == "backbone":
            return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)

        if self.use_features_key:
            payload = {"node_features": node_feats}
            if graph_feats is not None:
                payload["graph_features"] = graph_feats
        else:
            payload = {"node_feats": node_feats}
            if graph_feats is not None:
                payload["graph_feats"] = graph_feats
        return payload


def _sample_batch() -> Batch:
    z = torch.tensor([1, 6, 8, 1, 6])
    pos = torch.randn(z.size(0), 3)
    batch = torch.tensor([0, 0, 0, 1, 1])
    data_list = [
        Data(z=z[batch == 0], pos=pos[batch == 0]),
        Data(z=z[batch == 1], pos=pos[batch == 1]),
    ]
    return Batch.from_data_list(data_list)


def test_triforces_model_pools_graph_feats_when_missing():
    batch = _sample_batch()
    num_graphs = batch.num_graphs

    interaction = DummyInteraction(embed_dim=6, return_type="dict", include_graph=False)
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

    interaction = DummyInteraction(
        embed_dim=4, return_type="tuple", include_graph=True, use_features_key=True
    )
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

    interaction = DummyInteraction(embed_dim=7, return_type="dict", include_graph=False)
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
