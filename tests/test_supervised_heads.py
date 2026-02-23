from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from triforces.models.heads import (
    DirectSupervisedHead,
    EnergyConservingHead,
    EquivariantVectorHead,
)
from triforces.models.outputs import BackboneOutputs
from triforces.models.triforces import TriForcesModel


def _mean_pool(
    node_feats: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    out = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
    out.index_add_(0, batch, node_feats)
    count = torch.bincount(batch, minlength=num_graphs).clamp_min(1).to(out.dtype)
    return out / count.unsqueeze(1)


def test_direct_supervised_head_outputs_expected_shapes() -> None:
    node_feats = torch.randn(5, 16)
    graph_feats = torch.randn(2, 16)
    backbone_outputs = BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)
    batch = SimpleNamespace(
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        num_graphs=2,
    )

    head = DirectSupervisedHead(
        input_dim=16,
        hidden_dims=[32],
        predict_energy=True,
        predict_forces=True,
        predict_stress=True,
        stress_representation="voigt",
    )
    out = head(backbone_outputs, batch)

    assert out["energy"].shape == (2,)
    assert out["forces"].shape == (5, 3)
    assert out["stress"].shape == (2, 6)


def test_energy_conserving_head_produces_forces_from_energy_gradient() -> None:
    pos = torch.randn(4, 3, requires_grad=True)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    num_graphs = 2
    graph_feats = _mean_pool(pos, batch_idx, num_graphs)
    backbone_outputs = BackboneOutputs(node_feats=pos, graph_feats=graph_feats)
    batch = SimpleNamespace(pos=pos, batch=batch_idx, num_graphs=num_graphs)

    head = EnergyConservingHead(
        input_dim=3,
        hidden_dims=[],
        predict_forces=True,
        predict_stress=False,
    )
    out = head(backbone_outputs, batch, training=True)

    assert out["energy"].shape == (2,)
    assert out["forces"].shape == (4, 3)
    assert torch.isfinite(out["forces"]).all()


def test_energy_conserving_head_can_emit_custom_force_key_without_energy() -> None:
    pos = torch.randn(4, 3, requires_grad=True)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    num_graphs = 2
    graph_feats = _mean_pool(pos, batch_idx, num_graphs)
    backbone_outputs = BackboneOutputs(node_feats=pos, graph_feats=graph_feats)
    batch = SimpleNamespace(pos=pos, batch=batch_idx, num_graphs=num_graphs)

    head = EnergyConservingHead(
        input_dim=3,
        hidden_dims=[],
        predict_forces=True,
        predict_stress=False,
        forces_output_key="noise_displacement",
        include_energy=False,
    )
    out = head(backbone_outputs, batch, training=True)

    assert "energy" not in out
    assert out["noise_displacement"].shape == (4, 3)
    assert torch.isfinite(out["noise_displacement"]).all()


def test_equivariant_vector_head_rotates_equivariantly() -> None:
    n = 6
    c = 4
    scalar = torch.randn(n, c)
    vectors = torch.randn(n, c, 3)
    equivariant = torch.cat([scalar, vectors.reshape(n, -1)], dim=-1)

    theta = torch.tensor(torch.pi / 3)
    rot = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=equivariant.dtype,
    )
    vectors_rot = torch.einsum("nci,ij->ncj", vectors, rot.T)
    equivariant_rot = torch.cat([scalar, vectors_rot.reshape(n, -1)], dim=-1)

    backbone_outputs = BackboneOutputs(
        node_feats=torch.randn(n, 8),
        graph_feats=torch.randn(2, 8),
    )
    batch = SimpleNamespace(
        batch=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        num_graphs=2,
    )
    head = EquivariantVectorHead(output_key="forces")

    out = head(backbone_outputs, batch, outputs={"node_feats_equivariant": equivariant})
    out_rot = head(
        backbone_outputs,
        batch,
        outputs={"node_feats_equivariant": equivariant_rot},
    )

    expected_rot = torch.einsum("ni,ij->nj", out["forces"], rot.T)
    assert torch.allclose(out_rot["forces"], expected_rot, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_equivariant_vector_head_lazy_gate_uses_input_device() -> None:
    n = 6
    c = 4
    device = torch.device("cuda")
    scalar = torch.randn(n, c, device=device)
    vectors = torch.randn(n, c, 3, device=device)
    equivariant = torch.cat([scalar, vectors.reshape(n, -1)], dim=-1)

    backbone_outputs = BackboneOutputs(
        node_feats=torch.randn(n, 8, device=device),
        graph_feats=torch.randn(2, 8, device=device),
    )
    batch = SimpleNamespace(
        batch=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        num_graphs=2,
    )
    head = EquivariantVectorHead(output_key="forces").to(device)
    out = head(backbone_outputs, batch, outputs={"node_feats_equivariant": equivariant})

    assert out["forces"].device == scalar.device
    assert head.gate is not None
    assert next(head.gate.parameters()).device == scalar.device


class _DummyInteractionWithExtras(nn.Module):
    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(120, embed_dim)

    def forward(self, batch, training: bool = False, transform=None):
        node_feats = self.embed(batch.z)
        graph_feats = _mean_pool(node_feats, batch.batch, batch.num_graphs)
        return BackboneOutputs(
            node_feats=node_feats,
            graph_feats=graph_feats,
            extras={
                "node_feats_equivariant": torch.randn(
                    node_feats.size(0), 4 * node_feats.size(1), device=node_feats.device
                ),
                "pos": batch.pos,
            },
        )


def test_triforces_preserves_interaction_extras_for_heads() -> None:
    data = Data(
        z=torch.tensor([1, 6, 8, 1], dtype=torch.long),
        pos=torch.randn(4, 3),
    )
    batch = Batch.from_data_list([data])

    model = TriForcesModel(
        interaction=_DummyInteractionWithExtras(embed_dim=6),
        interaction_dim=6,
        enable_composition=False,
        enable_structural=False,
    )
    out = model(batch)

    assert "node_feats_equivariant" in out.extras
    assert out.extras["node_feats_equivariant"].shape[0] == data.z.numel()
