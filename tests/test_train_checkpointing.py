from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch_geometric.data import Data

from triforces.models.outputs import BackboneOutputs
from triforces.train import _load_checkpoint_weights, run


class _ToyAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)


class _DummyInteractionBackbone(nn.Module):
    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(119, embed_dim)

    def forward(self, batch, training: bool = False, transform=None):
        _ = training, transform
        node_feats = self.embed(batch.z)
        num_graphs = batch.num_graphs
        graph_feats = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
        graph_feats.index_add_(0, batch.batch, node_feats)
        count = torch.bincount(batch.batch, minlength=num_graphs).clamp_min(1)
        graph_feats = graph_feats / count.to(graph_feats.dtype).unsqueeze(1)
        return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)


class _ToyGraphDataset(Dataset[Data]):
    def __init__(self) -> None:
        self._items: list[Data] = []
        for i in range(4):
            z = torch.tensor([14, 14], dtype=torch.long)
            pos = torch.tensor(
                [[0.0 + 0.1 * i, 0.0, 0.0], [1.5 + 0.1 * i, 0.0, 0.0]],
                dtype=torch.float32,
            )
            energy = torch.tensor([float(i + 1)], dtype=torch.float32)
            self._items.append(Data(z=z, pos=pos, energy=energy))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Data:
        return self._items[int(idx)]


def _base_train_cfg(checkpoint_dir: Path, *, epochs: int, resume_from: str | None = None):
    return OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {
                "_target_": "tests.test_train_checkpointing._ToyGraphDataset",
            },
            "collate": {
                "_target_": "triforces.data.pyg_supervised_collate",
                "_partial_": True,
            },
            "model": {
                "_target_": "triforces.models.adapter_model.AdapterModel",
                "backbone": {
                    "_target_": "triforces.models.triforces.TriForcesModel",
                    "interaction": {
                        "_target_": "tests.test_train_checkpointing._DummyInteractionBackbone",
                        "embed_dim": 8,
                    },
                    "interaction_dim": 8,
                    "interaction_name": "dummy",
                    "enable_composition": False,
                    "enable_structural": False,
                    "use_final_mlp": False,
                },
                "heads": {
                    "pred": {
                        "_target_": "triforces.models.heads.DirectSupervisedHead",
                        "input_dim": 8,
                        "hidden_dims": [8],
                        "predict_energy": True,
                        "predict_forces": False,
                        "predict_stress": False,
                    }
                },
            },
            "loss": {
                "_target_": "triforces.losses.SupervisedLoss",
                "energy_weight": 1.0,
                "forces_weight": 0.0,
                "stress_weight": 0.0,
                "energy_references": {14: -1.25},
                "standardization": {
                    "energy": {"mean": 0.0, "std": 2.0},
                },
            },
            "train": {
                "epochs": int(epochs),
                "batch_size": 2,
                "lr": 1e-3,
                "tqdm": False,
                "log_every": 1,
                "checkpoint": {
                    "enabled": True,
                    "dir": str(checkpoint_dir),
                    "save_every_epochs": 1,
                    "save_last": True,
                    "save_best": True,
                    "keep_last_n": 1,
                    "monitor": "loss",
                    "mode": "min",
                    "resume_from": resume_from,
                    "resume_strict": True,
                    "init_from": None,
                    "init_mode": "full",
                    "init_strict": False,
                },
            },
            "logger": {"enabled": False},
        }
    )


def test_load_checkpoint_weights_backbone_only_keeps_head() -> None:
    torch.manual_seed(0)
    source = _ToyAdapter()
    target = _ToyAdapter()
    head_before = {k: v.detach().clone() for k, v in target.head.state_dict().items()}

    checkpoint = {"model_state_dict": source.state_dict()}
    _load_checkpoint_weights(
        model=target,
        checkpoint=checkpoint,
        mode="backbone",
        strict=True,
    )

    for key, value in source.backbone.state_dict().items():
        assert torch.allclose(value, target.backbone.state_dict()[key])
    for key, value in target.head.state_dict().items():
        assert torch.allclose(value, head_before[key])


def test_run_saves_and_resumes_checkpoints(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"

    first_cfg = _base_train_cfg(ckpt_dir, epochs=1)
    assert run(first_cfg) == 0

    first_last = ckpt_dir / "last.pt"
    first_best = ckpt_dir / "best.pt"
    first_epoch = ckpt_dir / "epoch_0001.pt"
    assert first_last.exists()
    assert first_best.exists()
    assert first_epoch.exists()

    first_payload = torch.load(first_last, map_location="cpu", weights_only=False)
    assert int(first_payload["epoch"]) == 0
    first_step = int(first_payload["global_step"])
    assert first_step > 0
    assert "loss_checkpoint_state" in first_payload
    loss_state = first_payload["loss_checkpoint_state"]
    assert "energy_references" in loss_state
    assert "standardization" in loss_state
    assert "energy" in loss_state["standardization"]

    second_cfg = _base_train_cfg(ckpt_dir, epochs=2, resume_from=str(first_last))
    assert run(second_cfg) == 0

    second_last = ckpt_dir / "last.pt"
    second_payload = torch.load(second_last, map_location="cpu", weights_only=False)
    assert int(second_payload["epoch"]) == 1
    assert int(second_payload["global_step"]) > first_step
    assert "loss_checkpoint_state" in second_payload

    # keep_last_n=1 should keep only the most recent epoch checkpoint.
    assert not (ckpt_dir / "epoch_0001.pt").exists()
    assert (ckpt_dir / "epoch_0002.pt").exists()
