from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from triforces.evaluation.linear_probe import LinearProbeEvaluator


def _unwrap_dataset_for_probe(dataset: object, *, use_base_dataset: bool) -> object:
    if not use_base_dataset:
        return dataset
    cls_name = dataset.__class__.__name__
    if (
        (cls_name.endswith("ContrastiveDataset") or cls_name.endswith("AugmentationDataset"))
        and hasattr(dataset, "dataset")
    ):
        return getattr(dataset, "dataset")
    return dataset


def _build_probe_loader(
    *,
    dataset: object,
    collate_fn: object,
    train_batch_size: int,
    probe_cfg: DictConfig,
) -> DataLoader:
    raw_probe_batch_size = probe_cfg.get("batch_size", None)
    probe_batch_size = (
        train_batch_size if raw_probe_batch_size is None else int(raw_probe_batch_size)
    )
    if probe_batch_size < 1:
        probe_batch_size = 1
    raw_probe_workers = probe_cfg.get("num_workers", 0)
    probe_workers = 0 if raw_probe_workers is None else int(raw_probe_workers)
    if probe_workers < 0:
        probe_workers = 0
    return DataLoader(
        dataset,
        batch_size=probe_batch_size,
        shuffle=False,
        num_workers=probe_workers,
        collate_fn=collate_fn,
    )


def _build_probe_evaluator(probe_cfg: DictConfig) -> LinearProbeEvaluator:
    regression_properties = probe_cfg.get("regression_properties")
    classification_properties = probe_cfg.get("classification_properties")
    reg_props = (
        [str(p) for p in regression_properties]
        if isinstance(regression_properties, (list, tuple))
        else None
    )
    cls_props = (
        [str(p) for p in classification_properties]
        if isinstance(classification_properties, (list, tuple))
        else None
    )
    return LinearProbeEvaluator(
        regression_properties=reg_props,
        classification_properties=cls_props,
        test_size=float(probe_cfg.get("test_size", 0.2)),
        random_seed=int(probe_cfg.get("random_seed", 42)),
        ridge_alpha=float(probe_cfg.get("ridge_alpha", 1.0)),
        min_samples=int(probe_cfg.get("min_samples", 24)),
    )


def _probe_step_interval(probe_cfg: DictConfig) -> int | None:
    raw_value = probe_cfg.get("every_n_steps", None)
    if raw_value is None:
        return None
    interval = int(raw_value)
    if interval < 1:
        return None
    return interval


class LinearProbeHook:
    def __init__(
        self,
        *,
        probe_cfg: DictConfig,
        dataset: object,
        collate_fn: object,
        train_batch_size: int,
        device: torch.device,
        wandb_run: object | None,
    ) -> None:
        probe_dataset = _unwrap_dataset_for_probe(
            dataset,
            use_base_dataset=bool(probe_cfg.get("use_base_dataset", True)),
        )
        self.loader = _build_probe_loader(
            dataset=probe_dataset,
            collate_fn=collate_fn,
            train_batch_size=train_batch_size,
            probe_cfg=probe_cfg,
        )
        self.evaluator = _build_probe_evaluator(probe_cfg)

        self.device = device
        self.wandb_run = wandb_run
        self.embedding_key = str(probe_cfg.get("embedding_key", "graph_projections"))
        raw_max_samples = probe_cfg.get("max_samples", None)
        self.max_samples = None if raw_max_samples is None else int(raw_max_samples)
        self.log_prefix = str(probe_cfg.get("log_prefix", "linear_probe"))
        self.every_n_steps = _probe_step_interval(probe_cfg)
        self.every_n_epochs = max(1, int(probe_cfg.get("every_n_epochs", 5)))
        self.run_on_final_epoch = bool(probe_cfg.get("run_on_final_epoch", True))

        raw_progress_batches = probe_cfg.get("progress_every_batches", 500)
        if raw_progress_batches is None:
            self.progress_every_batches = None
        else:
            self.progress_every_batches = max(1, int(raw_progress_batches))

        self.last_probe_step: int | None = None

    def on_train_start(
        self,
        *,
        start_epoch: int,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None:
        _ = start_epoch
        _ = epochs
        _ = global_step
        _ = model
        print(
            "linear_probe: enabled "
            f"(embedding_key={self.embedding_key}, batch_size={self.loader.batch_size}, "
            f"max_samples={self.max_samples}, every_n_steps={self.every_n_steps}, "
            f"progress_every_batches={self.progress_every_batches})"
        )

    def _should_run_epoch(self, *, epoch: int, epochs: int) -> bool:
        is_periodic = ((epoch + 1) % self.every_n_epochs) == 0
        is_final = (epoch + 1) == int(epochs)
        return bool(is_periodic or (self.run_on_final_epoch and is_final))

    def _run_probe(
        self,
        *,
        model: torch.nn.Module,
        epoch: int,
        step: int,
        trigger: str,
    ) -> None:
        print(f"epoch={epoch + 1} step={step} {self.log_prefix} {trigger} started")

        def on_probe_progress(stage: str, payload: dict[str, Any]) -> None:
            if stage == "collect_progress":
                batches_seen = int(payload.get("batches_seen", 0))
                samples_collected = int(payload.get("samples_collected", 0))
                max_samples = payload.get("max_samples", None)
                if isinstance(max_samples, int):
                    print(
                        f"epoch={epoch + 1} step={step} {self.log_prefix} "
                        f"{trigger} collecting batches={batches_seen} "
                        f"samples={samples_collected}/{max_samples}"
                    )
                else:
                    print(
                        f"epoch={epoch + 1} step={step} {self.log_prefix} "
                        f"{trigger} collecting batches={batches_seen} "
                        f"samples={samples_collected}"
                    )
            elif stage == "fit_start":
                num_samples = int(payload.get("num_samples", 0))
                num_features = int(payload.get("num_features", 0))
                print(
                    f"epoch={epoch + 1} step={step} {self.log_prefix} "
                    f"{trigger} fitting samples={num_samples} "
                    f"features={num_features}"
                )

        probe_metrics = self.evaluator.evaluate(
            model=model,
            loader=self.loader,
            device=self.device,
            embedding_key=self.embedding_key,
            max_samples=self.max_samples,
            progress_callback=on_probe_progress,
            progress_every_batches=self.progress_every_batches,
        )
        if probe_metrics:
            print(
                f"epoch={epoch + 1} step={step} {self.log_prefix} "
                f"{trigger} metrics={len(probe_metrics)}"
            )
            if self.wandb_run is not None:
                payload = {
                    f"{self.log_prefix}/{key}": value
                    for key, value in probe_metrics.items()
                }
                payload["train/epoch"] = epoch + 1
                payload["train/global_step"] = step
                payload[f"{self.log_prefix}/trigger"] = trigger
                self.wandb_run.log(payload, step=step)
        else:
            print(
                f"epoch={epoch + 1} step={step} {self.log_prefix} {trigger} "
                "skipped (not enough valid probe targets)"
            )
        self.last_probe_step = step

    def on_step_end(
        self,
        *,
        epoch: int,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None:
        _ = epochs
        if self.every_n_steps is None:
            return
        if global_step % self.every_n_steps != 0:
            return
        if self.last_probe_step == global_step:
            return
        self._run_probe(model=model, epoch=epoch, step=global_step, trigger="step")

    def on_epoch_end(
        self,
        *,
        epoch: int,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None:
        if not self._should_run_epoch(epoch=epoch, epochs=epochs):
            return
        if self.last_probe_step == global_step:
            return
        self._run_probe(model=model, epoch=epoch, step=global_step, trigger="epoch")

    def on_train_end(
        self,
        *,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None:
        _ = epochs
        _ = global_step
        _ = model
