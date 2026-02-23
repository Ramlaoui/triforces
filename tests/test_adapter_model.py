from functools import partial

import torch
from torch_geometric.data import Batch, Data

from triforces.models.adapter_model import AdapterModel
from triforces.models.heads.byol import BYOLPredictorHead, BYOLProjectionHead
from triforces.models.heads.simclr import ProjectionHead
from triforces.models.outputs import BackboneOutputs


class DummyBackbone(torch.nn.Module):
    output_dim = 2

    def forward(self, batch, training=False, transform=None):
        z = batch.z
        batch_idx = batch.batch
        node_feats = torch.stack([z.float(), z.float()], dim=-1)
        num_graphs = batch.num_graphs
        graph_feats = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
        graph_feats.index_add_(0, batch_idx, node_feats)
        return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)


class DummyBatchBackbone(torch.nn.Module):
    def forward(self, batch, training=False, transform=None):
        node_feats = batch.x
        num_graphs = batch.num_graphs
        graph_feats = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
        graph_feats.index_add_(0, batch.batch, node_feats)
        return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)


class BuildFromInfoHead(torch.nn.Module):
    called_with = None

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @classmethod
    def build_from_backbone_info(cls, backbone_info, **kwargs):
        cls.called_with = dict(backbone_info)
        return cls(input_dim=int(backbone_info["output_dim"]))

    def forward(self, backbone_outputs, batch, outputs=None, **kwargs):
        return {"head_input_dim": torch.tensor(float(self.input_dim))}


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


def test_adapter_model_accepts_batch_input():
    data1 = Data(x=torch.randn(3, 4))
    data2 = Data(x=torch.randn(2, 4))
    batch = Batch.from_data_list([data1, data2])

    model = AdapterModel(backbone=DummyBatchBackbone())
    out = model(batch)

    assert out.node_feats.shape == (5, 4)
    assert out.graph_feats.shape == (2, 4)


def test_adapter_model_infers_input_dim_for_partial_projection_head():
    backbone = DummyBackbone()
    head_factory = partial(
        ProjectionHead,
        node_projection_dim=3,
        graph_projection_dim=5,
    )
    model = AdapterModel(backbone=backbone, heads={"proj": head_factory})

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


def test_adapter_model_infers_predictor_input_dim_from_projection_head():
    backbone = DummyBackbone()
    model = AdapterModel(
        backbone=backbone,
        heads={
            "proj": partial(
                BYOLProjectionHead,
                projection_dim=7,
                hidden_dim=16,
                use_bn=False,
            ),
            "pred": partial(
                BYOLPredictorHead,
                hidden_dim=12,
                use_bn=False,
            ),
        },
    )

    z = torch.tensor([1, 2, 1, 3])
    pos = torch.zeros((4, 3))
    data_list = [
        Data(z=z[:2], pos=pos[:2]),
        Data(z=z[2:], pos=pos[2:]),
    ]
    batch = Batch.from_data_list(data_list)
    out = model(batch)

    assert out.node_projections.shape == (4, 7)
    assert out.graph_projections.shape == (2, 7)
    assert out.node_predictions.shape == (4, 7)
    assert out.graph_predictions.shape == (2, 7)


def test_adapter_model_uses_head_build_from_backbone_info():
    backbone = DummyBackbone()
    BuildFromInfoHead.called_with = None
    model = AdapterModel(
        backbone=backbone, heads={"custom": partial(BuildFromInfoHead)}
    )

    z = torch.tensor([1, 2, 1, 3])
    pos = torch.zeros((4, 3))
    data_list = [
        Data(z=z[:2], pos=pos[:2]),
        Data(z=z[2:], pos=pos[2:]),
    ]
    batch = Batch.from_data_list(data_list)
    out = model(batch)

    assert BuildFromInfoHead.called_with is not None
    assert BuildFromInfoHead.called_with["output_dim"] == 2
    assert out.head_input_dim.item() == 2.0
