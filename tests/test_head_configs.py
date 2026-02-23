from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from triforces.models.outputs import BackboneOutputs
from triforces.models.triforces import TriForcesModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_all_head_configs_instantiate():
    head_dir = _repo_root() / "src" / "triforces" / "configs" / "head"
    paths = sorted(head_dir.rglob("*.yaml"))
    assert paths, f"No head configs found in {head_dir}"

    root = OmegaConf.create({"model": {"backbone": {"output_dim": 64}}})

    for path in paths:
        head_cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(root, OmegaConf.create({"head": head_cfg}))
        instantiate(cfg.head, _convert_="object")


class _DummyInteraction(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(120, embed_dim)

    def forward(self, batch, training: bool = False, transform=None):
        node_feats = self.embed(batch.z)
        num_graphs = batch.num_graphs
        graph_feats = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
        graph_feats.index_add_(0, batch.batch, node_feats)
        count = torch.bincount(batch.batch, minlength=num_graphs).clamp_min(1)
        graph_feats = graph_feats / count.to(graph_feats.dtype).unsqueeze(1)
        return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)


def test_triforces_model_aliases_interaction_stream():
    z = torch.tensor([1, 6, 8, 1, 6])
    pos = torch.randn(z.size(0), 3)
    batch = torch.tensor([0, 0, 0, 1, 1])
    data_list = [
        Data(z=z[batch == 0], pos=pos[batch == 0]),
        Data(z=z[batch == 1], pos=pos[batch == 1]),
    ]
    pyg_batch = Batch.from_data_list(data_list)

    model = TriForcesModel(
        interaction=_DummyInteraction(embed_dim=6),
        interaction_dim=6,
        interaction_name="orb",
        enable_composition=False,
        enable_structural=False,
    )
    out = model(pyg_batch)

    streams = out.extras["stream_node_feats"]
    assert "orb" in streams
    assert "interaction" in streams
    assert torch.allclose(streams["orb"], streams["interaction"])

    graph_streams = out.extras["stream_graph_feats"]
    assert "orb" in graph_streams
    assert "interaction" in graph_streams
    assert torch.allclose(graph_streams["orb"], graph_streams["interaction"])
