import torch
from torch_geometric.data import Batch, Data

from triforces.models.adapter_model import AdapterModel
from triforces.models.heads.simclr import ProjectionHead
from triforces.models.heads.multi_stream_barlow_twins import (
    MultiStreamBarlowTwinsProjectionHead,
)
from triforces.models.outputs import BackboneOutputs


class DummyBackbone(torch.nn.Module):
    def forward(self, batch, training=False, transform=None):
        z = batch.z
        batch_idx = batch.batch
        node_feats = torch.stack([z.float(), z.float()], dim=-1)
        num_graphs = batch.num_graphs
        graph_feats = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
        graph_feats.index_add_(0, batch_idx, node_feats)
        return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)


class DummyStreamBackbone(torch.nn.Module):
    def forward(self, batch, training=False, transform=None):
        z = batch.z
        batch_idx = batch.batch
        node_a = torch.stack([z.float(), z.float()], dim=-1)
        node_b = torch.stack([z.float(), z.float(), z.float()], dim=-1)
        num_graphs = batch.num_graphs
        graph_a = node_a.new_zeros((num_graphs, node_a.size(-1)))
        graph_b = node_b.new_zeros((num_graphs, node_b.size(-1)))
        graph_a.index_add_(0, batch_idx, node_a)
        graph_b.index_add_(0, batch_idx, node_b)
        node_feats = torch.cat([node_a, node_b], dim=-1)
        graph_feats = torch.cat([graph_a, graph_b], dim=-1)
        return BackboneOutputs(
            node_feats=node_feats,
            graph_feats=graph_feats,
            extras={
                "stream_node_feats": {"interaction": node_a, "composition": node_b},
                "stream_graph_feats": {"interaction": graph_a, "composition": graph_b},
            },
        )


class DummyBatchBackbone(torch.nn.Module):
    def forward(self, batch, training=False, transform=None):
        node_feats = batch.x
        num_graphs = batch.num_graphs
        graph_feats = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
        graph_feats.index_add_(0, batch.batch, node_feats)
        return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)


def test_adapter_model_projection_head():
    backbone = DummyBackbone()
    head = ProjectionHead(input_dim=2, node_projection_dim=3, graph_projection_dim=5)
    model = AdapterModel(backbone=backbone, heads={"proj": head})

    z = torch.tensor([1, 2, 1, 3])
    pos = torch.zeros((4, 3))
    data_list = [
        Data(z=z[:2], pos=pos[:2]),
        Data(z=z[2:], pos=pos[2:]),
    ]
    batch = Batch.from_data_list(data_list)

    out = model(batch)
    assert out.node_projections.shape == (4, 3)
    assert out.graph_projections.shape == (2, 5)


def test_adapter_model_multistream_head():
    backbone = DummyStreamBackbone()
    head = MultiStreamBarlowTwinsProjectionHead(
        stream_dims={"interaction": 2, "composition": 3},
        projection_dim=4,
        compute_node_level=True,
        compute_graph_level=True,
    )
    model = AdapterModel(backbone=backbone, heads={"multi": head})

    z = torch.tensor([1, 2, 1, 3])
    pos = torch.zeros((4, 3))
    data_list = [
        Data(z=z[:2], pos=pos[:2]),
        Data(z=z[2:], pos=pos[2:]),
    ]
    batch = Batch.from_data_list(data_list)

    out = model(batch)
    assert hasattr(out, "node_projections_interaction")
    assert hasattr(out, "graph_projections_composition")


def test_adapter_model_accepts_batch_input():
    data1 = Data(x=torch.randn(3, 4))
    data2 = Data(x=torch.randn(2, 4))
    batch = Batch.from_data_list([data1, data2])

    model = AdapterModel(backbone=DummyBatchBackbone())
    out = model(batch)

    assert out.node_feats.shape == (5, 4)
    assert out.graph_feats.shape == (2, 4)
