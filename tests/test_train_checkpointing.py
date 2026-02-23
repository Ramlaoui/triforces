from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch_geometric.data import Data

from triforces.models.outputs import BackboneOutputs
from triforces.train import (
    _load_checkpoint_weights,
    _model_cfg_with_backbone_from_checkpoint,
    _save_checkpoint,
    _validate_resume_data_pipeline_consistency,
    run,
)


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


def _base_train_cfg(
    checkpoint_dir: Path, *, epochs: int, resume_from: str | None = None
):
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
                    "save_last_every_steps": 1,
                    "save_best": True,
                    "keep_last_n": 1,
                    "monitor": "loss",
                    "mode": "min",
                    "resume_from": resume_from,
                    "resume_strict": True,
                    "init_from": None,
                    "init_mode": "full",
                    "init_strict": False,
                    "init_use_backbone_config": False,
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


def test_save_checkpoint_creates_parent_dir(tmp_path: Path) -> None:
    model = _ToyAdapter()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    ckpt_path = tmp_path / "missing" / "nested" / "last.pt"

    _save_checkpoint(
        path=ckpt_path,
        model=model,
        optim=optim,
        loss_fn=loss_fn,
        epoch=0,
        global_step=1,
        best_metric=None,
    )

    assert ckpt_path.exists()


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
    assert int(first_payload["checkpoint_schema_version"]) == 2
    assert isinstance(first_payload["config_resolved"], dict)
    assert isinstance(first_payload["model_config_resolved"], dict)
    assert isinstance(first_payload["loss_config_resolved"], dict)
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


def test_run_saves_last_checkpoint_each_step(tmp_path: Path, monkeypatch) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    cfg = _base_train_cfg(ckpt_dir, epochs=1)

    import triforces.train as train_module

    saved_last = {"count": 0}
    real_save_checkpoint = train_module._save_checkpoint

    def _spy_save_checkpoint(
        *,
        path: Path,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        loss_fn: nn.Module,
        epoch: int,
        global_step: int,
        best_metric: float | None,
        extra_payload: dict[str, object] | None = None,
    ) -> None:
        if path.name == "last.pt":
            saved_last["count"] += 1
        real_save_checkpoint(
            path=path,
            model=model,
            optim=optim,
            loss_fn=loss_fn,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            extra_payload=extra_payload,
        )

    monkeypatch.setattr(train_module, "_save_checkpoint", _spy_save_checkpoint)

    assert run(cfg) == 0

    # Dataset has 4 samples with batch_size=2, so we expect at least 2 step saves.
    assert saved_last["count"] >= 2


def test_run_saves_last_checkpoint_on_configured_step_interval(
    tmp_path: Path, monkeypatch
) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    cfg = _base_train_cfg(ckpt_dir, epochs=1)
    cfg.train.checkpoint.save_last_every_steps = 2

    import triforces.train as train_module

    saved_last = {"count": 0}
    real_save_checkpoint = train_module._save_checkpoint

    def _spy_save_checkpoint(
        *,
        path: Path,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        loss_fn: nn.Module,
        epoch: int,
        global_step: int,
        best_metric: float | None,
        extra_payload: dict[str, object] | None = None,
    ) -> None:
        if path.name == "last.pt":
            saved_last["count"] += 1
        real_save_checkpoint(
            path=path,
            model=model,
            optim=optim,
            loss_fn=loss_fn,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            extra_payload=extra_payload,
        )

    monkeypatch.setattr(train_module, "_save_checkpoint", _spy_save_checkpoint)

    assert run(cfg) == 0

    # With 2 train steps and save_last_every_steps=2, we save once in-step (step 2).
    # Epoch-end save is skipped because the last step already wrote last.pt.
    assert saved_last["count"] == 1


def test_resume_requires_checkpoint_config_metadata(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    legacy_path = ckpt_dir / "legacy_last.pt"

    model = _ToyAdapter()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    _save_checkpoint(
        path=legacy_path,
        model=model,
        optim=optim,
        loss_fn=loss_fn,
        epoch=0,
        global_step=1,
        best_metric=None,
    )

    cfg = _base_train_cfg(ckpt_dir, epochs=2, resume_from=str(legacy_path))
    with pytest.raises(RuntimeError, match="config_resolved"):
        run(cfg)


def test_run_rejects_resume_and_init_conflict(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    cfg = _base_train_cfg(ckpt_dir, epochs=1)
    cfg.train.checkpoint.resume_from = "/tmp/resume.pt"
    cfg.train.checkpoint.init_from = "/tmp/init.pt"

    with pytest.raises(ValueError, match="set only one"):
        run(cfg)


def test_run_rejects_invalid_init_mode(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    cfg = _base_train_cfg(ckpt_dir, epochs=1)
    cfg.train.checkpoint.init_mode = "bad_mode"

    with pytest.raises(ValueError, match="init_mode"):
        run(cfg)


def test_run_rejects_invalid_save_last_every_steps(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    cfg = _base_train_cfg(ckpt_dir, epochs=1)
    cfg.train.checkpoint.save_last_every_steps = 0

    with pytest.raises(ValueError, match="save_last_every_steps"):
        run(cfg)


def test_run_rejects_resume_path_directory(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    bad_resume = tmp_path / "not_a_file"
    bad_resume.mkdir(parents=True)
    cfg = _base_train_cfg(ckpt_dir, epochs=1)
    cfg.train.checkpoint.resume_from = str(bad_resume)

    with pytest.raises(ValueError, match="must point to a checkpoint file"):
        run(cfg)


def test_run_rejects_missing_collate_target(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    cfg = _base_train_cfg(ckpt_dir, epochs=1)
    cfg.collate = {"contrastive": False}

    with pytest.raises(ValueError, match="missing `_target_`"):
        run(cfg)


def test_run_rejects_init_use_backbone_config_without_init_from(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    cfg = _base_train_cfg(ckpt_dir, epochs=1)
    cfg.train.checkpoint.init_use_backbone_config = True
    cfg.train.checkpoint.init_mode = "backbone"

    with pytest.raises(ValueError, match="requires `train.checkpoint.init_from`"):
        run(cfg)


def test_run_rejects_init_use_backbone_config_without_backbone_mode(
    tmp_path: Path,
) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    init_path = tmp_path / "dummy.pt"
    torch.save({"model_state_dict": {}}, init_path)
    cfg = _base_train_cfg(ckpt_dir, epochs=1)
    cfg.train.checkpoint.init_from = str(init_path)
    cfg.train.checkpoint.init_use_backbone_config = True
    cfg.train.checkpoint.init_mode = "full"

    with pytest.raises(
        ValueError, match="requires `train.checkpoint.init_mode=backbone`"
    ):
        run(cfg)


def test_model_cfg_with_backbone_from_checkpoint_replaces_backbone_only() -> None:
    model_cfg = OmegaConf.create(
        {
            "_target_": "triforces.models.adapter_model.AdapterModel",
            "backbone": {"output_dim": 128, "interaction_name": "orb"},
            "heads": {
                "proj": {
                    "_target_": "dummy",
                    "hidden_dims": ["${model.backbone.output_dim}"],
                }
            },
        }
    )
    checkpoint_payload = {
        "model_config_resolved": {
            "backbone": {"output_dim": 512, "interaction_name": "orb"}
        }
    }
    updated_cfg = _model_cfg_with_backbone_from_checkpoint(
        model_cfg=model_cfg,
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=Path("/tmp/checkpoint.pt"),
    )

    assert updated_cfg.backbone.output_dim == 512
    assert updated_cfg.heads.proj._target_ == "dummy"
    assert OmegaConf.to_container(updated_cfg.heads.proj.hidden_dims, resolve=True) == [
        512
    ]


def test_validate_resume_data_pipeline_consistency_rejects_mismatch() -> None:
    launch_cfg = OmegaConf.create(
        {
            "dataset": {"_target_": "pkg.DatasetA", "root": "/a"},
            "collate": {"_target_": "pkg.collate_a", "contrastive": False},
        }
    )
    checkpoint_cfg = OmegaConf.create(
        {
            "dataset": {"_target_": "pkg.DatasetA", "root": "/b"},
            "collate": {"_target_": "pkg.collate_a", "contrastive": False},
        }
    )

    with pytest.raises(RuntimeError, match="data pipeline mismatch"):
        _validate_resume_data_pipeline_consistency(
            launch_cfg=launch_cfg,
            checkpoint_cfg=checkpoint_cfg,
            allow_override=False,
        )


def test_validate_resume_data_pipeline_consistency_allows_override() -> None:
    launch_cfg = OmegaConf.create(
        {
            "dataset": {"_target_": "pkg.DatasetA", "root": "/a"},
            "collate": {"_target_": "pkg.collate_a", "contrastive": False},
        }
    )
    checkpoint_cfg = OmegaConf.create(
        {
            "dataset": {"_target_": "pkg.DatasetA", "root": "/b"},
            "collate": {"_target_": "pkg.collate_a", "contrastive": False},
        }
    )

    _validate_resume_data_pipeline_consistency(
        launch_cfg=launch_cfg,
        checkpoint_cfg=checkpoint_cfg,
        allow_override=True,
    )
