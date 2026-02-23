from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from triforces.evaluation.linear_probe import LinearProbeEvaluator
from triforces.train import run


class _ToyGraphDataset(Dataset[Data]):
    def __init__(self, n: int = 32) -> None:
        self._items: list[Data] = []
        for i in range(n):
            n_atoms = 2 + (i % 5)
            if i % 2 == 0:
                z = torch.tensor([14] * n_atoms, dtype=torch.long)  # intermetallic-like
                space_group = torch.tensor(225, dtype=torch.long)  # cubic
            else:
                z = torch.tensor(([11, 17] * ((n_atoms + 1) // 2))[:n_atoms], dtype=torch.long)
                space_group = torch.tensor(62, dtype=torch.long)  # orthorhombic

            pos = torch.zeros((n_atoms, 3), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_dist = torch.empty((0,), dtype=torch.float32)
            cell_scale = float(2.0 + 0.1 * n_atoms)
            cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * cell_scale
            data = Data(
                z=z,
                atomic_numbers=z,
                pos=pos,
                edge_index=edge_index,
                edge_dist=edge_dist,
                cell=cell,
                space_group=space_group,
                pair_id=torch.tensor(i, dtype=torch.long),
                energy=torch.tensor(float(n_atoms), dtype=torch.float32),
            )
            self._items.append(data)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Data:
        return self._items[int(idx)].clone()


class _ProbeToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Linear(1, 4, bias=False)

    def forward(self, batch, training: bool = False, transform=None):
        num_graphs = batch.num_graphs
        counts = torch.bincount(batch.batch, minlength=num_graphs).to(torch.float32)
        counts = counts.unsqueeze(1)
        graph_proj = self.project(counts)
        return {"graph_projections": graph_proj, "graph_feats": graph_proj}


class _ProbeToyBatchOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Linear(1, 4, bias=False)

    def forward(self, batch, training: bool = False, transform=None):
        num_graphs = batch.num_graphs
        counts = torch.bincount(batch.batch, minlength=num_graphs).to(torch.float32)
        counts = counts.unsqueeze(1)
        graph_proj = self.project(counts)
        return Data(graph_projections=graph_proj, graph_feats=graph_proj)


class _ToyLoss(nn.Module):
    def forward(self, batch, out):
        value = out["graph_projections"]
        loss = (value.pow(2).mean()) + 0.0 * batch.z.float().mean()
        return loss, {"toy_metric": loss.detach()}


def test_linear_probe_evaluator_extracts_expected_targets() -> None:
    g1 = Data(
        z=torch.tensor([14, 14], dtype=torch.long),
        atomic_numbers=torch.tensor([14, 14], dtype=torch.long),
        pos=torch.zeros((2, 3)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_dist=torch.empty((0,), dtype=torch.float32),
        cell=torch.eye(3, dtype=torch.float32).unsqueeze(0) * 2.0,
        space_group=torch.tensor(225, dtype=torch.long),
    )
    g2 = Data(
        z=torch.tensor([11, 17, 11], dtype=torch.long),
        atomic_numbers=torch.tensor([11, 17, 11], dtype=torch.long),
        pos=torch.zeros((3, 3)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_dist=torch.empty((0,), dtype=torch.float32),
        cell=torch.eye(3, dtype=torch.float32).unsqueeze(0) * 3.0,
        space_group=torch.tensor(62, dtype=torch.long),
    )
    batch = Batch.from_data_list([g1, g2])
    evaluator = LinearProbeEvaluator(min_samples=2)
    targets = evaluator._extract_batch_targets(batch)

    assert "n_atoms" in targets
    assert "mean_atomic_number" in targets
    assert "volume_per_atom" in targets
    assert "density" in targets
    assert "chemical_family" in targets
    assert "crystal_system" in targets

    assert targets["n_atoms"].tolist() == [2.0, 3.0]
    assert abs(float(targets["mean_atomic_number"][0]) - 14.0) < 1e-6
    assert abs(float(targets["mean_atomic_number"][1]) - 13.0) < 1e-6
    assert targets["crystal_system"].tolist() == ["cubic", "orthorhombic"]


def test_linear_probe_evaluator_produces_metrics() -> None:
    dataset = _ToyGraphDataset(n=40)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=Batch.from_data_list,
    )
    model = _ProbeToyModel()
    evaluator = LinearProbeEvaluator(
        regression_properties=["n_atoms"],
        classification_properties=[],
        min_samples=8,
    )
    metrics = evaluator.evaluate(model=model, loader=loader, device=torch.device("cpu"))
    assert "n_atoms/r2" in metrics
    assert metrics["n_atoms/r2"] > 0.95


def test_linear_probe_evaluator_accepts_non_dict_model_outputs() -> None:
    dataset = _ToyGraphDataset(n=40)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=Batch.from_data_list,
    )
    model = _ProbeToyBatchOutputModel()
    evaluator = LinearProbeEvaluator(
        regression_properties=["n_atoms"],
        classification_properties=[],
        min_samples=8,
    )
    metrics = evaluator.evaluate(model=model, loader=loader, device=torch.device("cpu"))
    assert "n_atoms/r2" in metrics
    assert metrics["n_atoms/r2"] > 0.95


def test_linear_probe_progress_callback_receives_updates() -> None:
    dataset = _ToyGraphDataset(n=24)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=Batch.from_data_list,
    )
    model = _ProbeToyModel()
    evaluator = LinearProbeEvaluator(
        regression_properties=["n_atoms"],
        classification_properties=[],
        min_samples=8,
    )
    seen_stages: list[str] = []

    def _on_progress(stage: str, payload: dict[str, object]) -> None:
        _ = payload
        seen_stages.append(stage)

    metrics = evaluator.evaluate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        max_samples=16,
        progress_callback=_on_progress,
        progress_every_batches=1,
    )

    assert "n_atoms/r2" in metrics
    assert "collect_start" in seen_stages
    assert "collect_progress" in seen_stages
    assert "collect_done" in seen_stages
    assert "fit_start" in seen_stages
    assert "fit_done" in seen_stages


def test_train_run_with_linear_probe_enabled() -> None:
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {"_target_": "tests.test_linear_probe._ToyGraphDataset", "n": 24},
            "collate": {
                "_target_": "triforces.data.pyg_supervised_collate",
                "_partial_": True,
            },
            "model": {"_target_": "tests.test_linear_probe._ProbeToyModel"},
            "loss": {"_target_": "tests.test_linear_probe._ToyLoss"},
            "train": {
                "epochs": 1,
                "batch_size": 8,
                "lr": 1e-3,
                "tqdm": False,
                "log_every": 1,
                "hooks": {
                    "linear_probe": {
                        "enabled": True,
                        "batch_size": None,
                        "every_n_epochs": 1,
                        "run_on_final_epoch": True,
                        "embedding_key": "graph_projections",
                        "max_samples": 64,
                        "regression_properties": ["n_atoms"],
                        "classification_properties": [],
                        "min_samples": 8,
                        "log_prefix": "linear_probe",
                    }
                },
            },
            "logger": {"enabled": False},
        }
    )
    assert run(cfg) == 0


def test_train_run_with_linear_probe_every_n_steps(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_evaluate(self, **kwargs):
        _ = self
        _ = kwargs
        calls["count"] += 1
        return {}

    monkeypatch.setattr(LinearProbeEvaluator, "evaluate", _fake_evaluate)

    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {"_target_": "tests.test_linear_probe._ToyGraphDataset", "n": 24},
            "collate": {
                "_target_": "triforces.data.pyg_supervised_collate",
                "_partial_": True,
            },
            "model": {"_target_": "tests.test_linear_probe._ProbeToyModel"},
            "loss": {"_target_": "tests.test_linear_probe._ToyLoss"},
            "train": {
                "epochs": 1,
                "batch_size": 8,
                "lr": 1e-3,
                "tqdm": False,
                "log_every": 1,
                "hooks": {
                    "linear_probe": {
                        "enabled": True,
                        "every_n_steps": 1,
                        "every_n_epochs": 100,
                        "run_on_final_epoch": False,
                        "embedding_key": "graph_projections",
                        "max_samples": 64,
                        "regression_properties": ["n_atoms"],
                        "classification_properties": [],
                        "min_samples": 8,
                        "log_prefix": "linear_probe",
                    }
                },
            },
            "logger": {"enabled": False},
        }
    )
    assert run(cfg) == 0
    assert calls["count"] == 3
