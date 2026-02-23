from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch_geometric.data import Data

import triforces.train as train_module
from triforces.train import run


class _ToySupervisedDataset(Dataset[Data]):
    def __init__(self, n: int = 8) -> None:
        self._items: list[Data] = []
        for i in range(n):
            z = torch.tensor([14, 14], dtype=torch.long)
            pos = torch.tensor(
                [[0.0 + 0.05 * i, 0.0, 0.0], [1.4 + 0.05 * i, 0.0, 0.0]],
                dtype=torch.float32,
            )
            energy = torch.tensor(float(i + 1), dtype=torch.float32)
            self._items.append(Data(z=z, pos=pos, energy=energy))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Data:
        return self._items[int(idx)].clone()


class _ToyEnergyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Linear(1, 1, bias=False)

    def forward(self, batch, training: bool = False, transform=None):
        _ = training, transform
        num_graphs = batch.num_graphs
        n_atoms = torch.bincount(batch.batch, minlength=num_graphs).to(torch.float32)
        pred_energy = self.project(n_atoms.unsqueeze(1)).reshape(-1)
        return {"energy": pred_energy}


class _ToySupervisedLoss(nn.Module):
    def forward(self, batch, out):
        pred = out["energy"].reshape(-1)
        target = batch.energy.to(device=pred.device, dtype=pred.dtype).reshape(-1)
        mse = torch.mean((pred - target) ** 2)
        mae = torch.mean(torch.abs(pred - target))
        return mse, {"mae_energy": mae.detach()}


class _FakeWandbConfig:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def update(
        self, payload: dict[str, object], allow_val_change: bool = False
    ) -> None:
        _ = allow_val_change
        self.payloads.append(dict(payload))


class _FakeWandbRun:
    def __init__(self) -> None:
        self.config = _FakeWandbConfig()
        self.logs: list[tuple[dict[str, object], int | None]] = []
        self.finished = False

    def log(self, payload: dict[str, object], step: int | None = None) -> None:
        self.logs.append((dict(payload), step))

    def finish(self) -> None:
        self.finished = True


def test_train_run_with_supervised_eval_hook_logs_validation_metrics(
    monkeypatch,
) -> None:
    fake_run = _FakeWandbRun()

    def _fake_wandb_init(**kwargs):
        _ = kwargs
        return fake_run

    monkeypatch.setattr(train_module.wandb, "init", _fake_wandb_init)

    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {
                "_target_": "tests.test_supervised_eval_hook._ToySupervisedDataset",
                "n": 8,
            },
            "collate": {
                "_target_": "triforces.data.pyg_supervised_collate",
                "_partial_": True,
            },
            "model": {"_target_": "tests.test_supervised_eval_hook._ToyEnergyModel"},
            "loss": {"_target_": "tests.test_supervised_eval_hook._ToySupervisedLoss"},
            "train": {
                "epochs": 1,
                "batch_size": 2,
                "lr": 1e-3,
                "tqdm": False,
                "log_every": 1,
                "hooks": {
                    "supervised_eval": {
                        "enabled": True,
                        "every_n_steps": 1,
                        "every_n_epochs": 100,
                        "run_on_final_epoch": False,
                        "val_fraction": 0.5,
                        "max_batches": 1,
                        "progress_bar": False,
                        "log_prefix": "val",
                    }
                },
            },
            "logger": {
                "enabled": True,
                "project": "unit-test-project",
                "mode": "offline",
            },
        }
    )

    assert run(cfg) == 0
    assert fake_run.finished is True

    val_payloads = [payload for payload, _ in fake_run.logs if "val/trigger" in payload]
    assert len(val_payloads) > 0
    assert any("val/loss" in payload for payload in val_payloads)
    assert any("val/mae_energy" in payload for payload in val_payloads)
    assert any("val/mae_energy_per_atom" in payload for payload in val_payloads)


def test_train_run_with_dataset_val_updates_wandb_config(monkeypatch) -> None:
    fake_run = _FakeWandbRun()

    def _fake_wandb_init(**kwargs):
        _ = kwargs
        return fake_run

    monkeypatch.setattr(train_module.wandb, "init", _fake_wandb_init)

    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {
                "_target_": "tests.test_supervised_eval_hook._ToySupervisedDataset",
                "n": 8,
            },
            "dataset_val": {
                "_target_": "tests.test_supervised_eval_hook._ToySupervisedDataset",
                "n": 3,
            },
            "collate": {
                "_target_": "triforces.data.pyg_supervised_collate",
                "_partial_": True,
            },
            "model": {"_target_": "tests.test_supervised_eval_hook._ToyEnergyModel"},
            "loss": {"_target_": "tests.test_supervised_eval_hook._ToySupervisedLoss"},
            "train": {
                "epochs": 1,
                "batch_size": 2,
                "lr": 1e-3,
                "tqdm": False,
                "log_every": 1,
                "hooks": {
                    "supervised_eval": {
                        "enabled": True,
                        "every_n_steps": 1,
                        "every_n_epochs": 100,
                        "run_on_final_epoch": False,
                        "val_fraction": 0.01,
                        "max_batches": 1,
                        "progress_bar": False,
                        "log_prefix": "val",
                    }
                },
            },
            "logger": {
                "enabled": True,
                "project": "unit-test-project",
                "mode": "offline",
            },
        }
    )

    assert run(cfg) == 0

    merged_config_updates: dict[str, object] = {}
    for payload in fake_run.config.payloads:
        merged_config_updates.update(payload)
    assert merged_config_updates.get("dataset_size") == 8
    assert merged_config_updates.get("dataset_val_size") == 3
