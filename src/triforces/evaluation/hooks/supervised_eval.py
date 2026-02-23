from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from triforces.utils.stress import stress_to_voigt_6


def _unwrap_dataset_for_eval(dataset: object, *, use_base_dataset: bool) -> object:
    if not use_base_dataset:
        return dataset
    cls_name = dataset.__class__.__name__
    if (
        cls_name.endswith("ContrastiveDataset")
        or cls_name.endswith("AugmentationDataset")
    ) and hasattr(dataset, "dataset"):
        return getattr(dataset, "dataset")
    return dataset


def _step_interval(eval_cfg: DictConfig) -> int | None:
    raw_value = eval_cfg.get("every_n_steps", None)
    if raw_value is None:
        return None
    interval = int(raw_value)
    if interval < 1:
        return None
    return interval


def _resolve_prediction(outputs: object, key: str) -> torch.Tensor | None:
    if isinstance(outputs, dict):
        value = outputs.get(key)
        return value if torch.is_tensor(value) else None
    value = getattr(outputs, key, None)
    return value if torch.is_tensor(value) else None


def _as_voigt6(stress: torch.Tensor) -> torch.Tensor:
    if stress.ndim >= 2 and stress.shape[-2:] == (3, 3):
        out = stress_to_voigt_6(stress)
        if out is None:
            raise ValueError("Failed to convert stress tensor to Voigt-6.")
        return out
    if stress.shape[-1] == 9:
        out = stress_to_voigt_6(stress.reshape(*stress.shape[:-1], 3, 3))
        if out is None:
            raise ValueError("Failed to convert stress tensor to Voigt-6.")
        return out
    if stress.shape[-1] == 6:
        return stress
    raise ValueError(
        "Stress tensor must be (..., 6), (..., 9), or (..., 3, 3), got "
        f"{tuple(stress.shape)}"
    )


def _resolve_num_atoms_per_graph(
    *,
    batch: object,
    ref: torch.Tensor,
) -> torch.Tensor | None:
    num_graphs = int(ref.reshape(-1).shape[0])
    for name in ("natoms", "num_atoms_per_graph", "n_atoms"):
        value = getattr(batch, name, None)
        if value is None:
            continue
        num_atoms = torch.as_tensor(value, device=ref.device, dtype=ref.dtype).reshape(
            -1
        )
        if num_atoms.numel() == num_graphs:
            return num_atoms.clamp_min(1.0)
    batch_index = getattr(batch, "batch", None)
    if batch_index is not None:
        batch_index_t = torch.as_tensor(
            batch_index, device=ref.device, dtype=torch.long
        ).reshape(-1)
        num_atoms = torch.bincount(batch_index_t, minlength=num_graphs).to(
            dtype=ref.dtype
        )
        return num_atoms.clamp_min(1.0)
    return None


class SupervisedEvalHook:
    def __init__(
        self,
        *,
        eval_cfg: DictConfig,
        dataset: object,
        dataset_val: object | None,
        collate_fn: object,
        train_batch_size: int,
        device: torch.device,
        wandb_run: object | None,
        loss_fn: torch.nn.Module | None,
    ) -> None:
        has_explicit_val_dataset = dataset_val is not None
        eval_source = dataset if dataset_val is None else dataset_val
        eval_dataset = _unwrap_dataset_for_eval(
            eval_source,
            use_base_dataset=bool(eval_cfg.get("use_base_dataset", False)),
        )
        if not hasattr(eval_dataset, "__len__") or not hasattr(
            eval_dataset, "__getitem__"
        ):
            raise ValueError(
                "supervised_eval hook requires an indexable dataset with __len__ and __getitem__."
            )

        dataset_size = int(len(eval_dataset))
        if dataset_size < 2:
            raise ValueError(
                "supervised_eval hook requires at least 2 samples to build a validation split."
            )

        if has_explicit_val_dataset:
            val_size = dataset_size
            val_subset = eval_dataset
            self._val_source = "dataset_val"
        else:
            val_fraction = float(eval_cfg.get("val_fraction", 0.1))
            if not (0.0 < val_fraction < 1.0):
                raise ValueError(
                    f"supervised_eval.val_fraction must be in (0, 1), got {val_fraction}."
                )
            val_size = int(round(dataset_size * val_fraction))
            val_size = max(1, min(val_size, dataset_size - 1))
            rng = torch.Generator()
            rng.manual_seed(int(eval_cfg.get("random_seed", 42)))
            indices = torch.randperm(dataset_size, generator=rng).tolist()
            val_indices = indices[:val_size]
            val_subset = Subset(eval_dataset, val_indices)
            self._val_source = "train_split"

        raw_batch_size = eval_cfg.get("batch_size", None)
        eval_batch_size = (
            train_batch_size if raw_batch_size is None else max(1, int(raw_batch_size))
        )
        raw_workers = eval_cfg.get("num_workers", 0)
        eval_workers = 0 if raw_workers is None else max(0, int(raw_workers))
        self.loader = DataLoader(
            val_subset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_workers,
            collate_fn=collate_fn,
        )
        self.device = device
        self.wandb_run = wandb_run
        self.loss_fn = loss_fn
        self.log_prefix = str(eval_cfg.get("log_prefix", "val"))
        self.every_n_steps = _step_interval(eval_cfg)
        self.every_n_epochs = max(1, int(eval_cfg.get("every_n_epochs", 1)))
        self.run_on_final_epoch = bool(eval_cfg.get("run_on_final_epoch", True))
        raw_max_batches = eval_cfg.get("max_batches", None)
        self.max_batches = (
            None if raw_max_batches is None else max(1, int(raw_max_batches))
        )
        self.progress_bar = bool(eval_cfg.get("progress_bar", True))
        raw_progress_every = eval_cfg.get("progress_every_batches", 100)
        if raw_progress_every is None:
            self.progress_every_batches = None
        else:
            self.progress_every_batches = max(1, int(raw_progress_every))
        self.last_eval_step: int | None = None
        self._val_size = val_size
        self._dataset_size = dataset_size

    def _prediction_for_metric(
        self,
        *,
        key: str,
        outputs: object,
        batch: object,
    ) -> torch.Tensor | None:
        pred = _resolve_prediction(outputs, key)
        if not torch.is_tensor(pred):
            return None
        if self.loss_fn is None:
            return pred
        denorm = getattr(self.loss_fn, "denormalize_prediction", None)
        if not callable(denorm):
            return pred
        converted = denorm(key, pred, data=batch)
        return converted if torch.is_tensor(converted) else pred

    def on_train_start(
        self,
        *,
        start_epoch: int,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None:
        _ = start_epoch, epochs, global_step, model
        print(
            "supervised_eval: enabled "
            f"(source={self._val_source}, val_size={self._val_size}/{self._dataset_size}, "
            f"batch_size={self.loader.batch_size}, every_n_steps={self.every_n_steps}, "
            f"every_n_epochs={self.every_n_epochs}, progress_bar={self.progress_bar})"
        )

    def _should_run_epoch(self, *, epoch: int, epochs: int) -> bool:
        is_periodic = ((epoch + 1) % self.every_n_epochs) == 0
        is_final = (epoch + 1) == int(epochs)
        return bool(is_periodic or (self.run_on_final_epoch and is_final))

    def _accumulate_mae(
        self,
        *,
        outputs: object,
        batch: object,
        mae_totals: dict[str, float],
        mae_counts: dict[str, int],
    ) -> None:
        energy_pred = self._prediction_for_metric(
            key="energy", outputs=outputs, batch=batch
        )
        energy_true = getattr(batch, "energy", None)
        if torch.is_tensor(energy_pred) and torch.is_tensor(energy_true):
            pred = energy_pred.reshape(-1)
            target = energy_true.to(device=pred.device, dtype=pred.dtype).reshape(-1)
            mask = torch.isfinite(target)
            if mask.any():
                mae_totals["energy"] += float(
                    torch.abs(pred[mask] - target[mask]).sum().item()
                )
                mae_counts["energy"] += int(mask.sum().item())
                num_atoms = _resolve_num_atoms_per_graph(batch=batch, ref=pred)
                if num_atoms is not None and num_atoms.numel() == pred.numel():
                    masked_num_atoms = num_atoms[mask].clamp_min(1.0)
                    pred_pa = pred[mask] / masked_num_atoms
                    target_pa = target[mask] / masked_num_atoms
                    mae_totals["energy_per_atom"] += float(
                        torch.abs(pred_pa - target_pa).sum().item()
                    )
                    mae_counts["energy_per_atom"] += int(mask.sum().item())

        forces_pred = self._prediction_for_metric(
            key="forces", outputs=outputs, batch=batch
        )
        forces_true = getattr(batch, "forces", None)
        if torch.is_tensor(forces_pred) and torch.is_tensor(forces_true):
            pred = forces_pred
            target = forces_true.to(device=pred.device, dtype=pred.dtype)
            if pred.shape == target.shape and pred.ndim == 2 and pred.size(-1) == 3:
                mask = torch.isfinite(target).all(dim=-1)
                if mask.any():
                    row_mae = torch.abs(pred[mask] - target[mask]).mean(dim=-1)
                    mae_totals["forces"] += float(row_mae.sum().item())
                    mae_counts["forces"] += int(mask.sum().item())

        stress_pred = self._prediction_for_metric(
            key="stress", outputs=outputs, batch=batch
        )
        stress_true = getattr(batch, "stress", None)
        if torch.is_tensor(stress_pred) and torch.is_tensor(stress_true):
            pred = _as_voigt6(stress_pred).reshape(-1, 6)
            target = _as_voigt6(
                stress_true.to(device=pred.device, dtype=pred.dtype)
            ).reshape(-1, 6)
            if pred.shape == target.shape:
                mask = torch.isfinite(target).all(dim=-1)
                if mask.any():
                    row_mae = torch.abs(pred[mask] - target[mask]).mean(dim=-1)
                    mae_totals["stress"] += float(row_mae.sum().item())
                    mae_counts["stress"] += int(mask.sum().item())

    def _run_eval(
        self,
        *,
        model: torch.nn.Module,
        epoch: int,
        step: int,
        trigger: str,
    ) -> None:
        total_batches: int | None = None
        try:
            total_batches = int(len(self.loader))
        except TypeError:
            total_batches = None
        if self.max_batches is not None:
            if total_batches is None:
                total_batches = int(self.max_batches)
            else:
                total_batches = min(total_batches, int(self.max_batches))

        print(
            f"epoch={epoch + 1} step={step} {self.log_prefix} {trigger} started "
            f"batches={total_batches if total_batches is not None else 'unknown'}"
        )

        was_training = model.training
        model.eval()
        grad_enabled = bool(getattr(model, "requires_grad_for_inference", False))

        loss_total = 0.0
        loss_count = 0
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        mae_totals = {
            "energy": 0.0,
            "energy_per_atom": 0.0,
            "forces": 0.0,
            "stress": 0.0,
        }
        mae_counts = {"energy": 0, "energy_per_atom": 0, "forces": 0, "stress": 0}
        processed_batches = 0

        eval_iterator = self.loader
        progress = None
        if self.progress_bar:
            progress = tqdm(
                self.loader,
                desc=f"{self.log_prefix} {trigger} e{epoch + 1}",
                total=total_batches,
                leave=False,
            )
            eval_iterator = progress

        try:
            with torch.set_grad_enabled(grad_enabled):
                for batch_idx, batch in enumerate(eval_iterator, start=1):
                    if self.max_batches is not None and batch_idx > self.max_batches:
                        break
                    processed_batches = batch_idx
                    batch = batch.to(self.device)
                    outputs = model(batch, training=False)
                    self._accumulate_mae(
                        outputs=outputs,
                        batch=batch,
                        mae_totals=mae_totals,
                        mae_counts=mae_counts,
                    )

                    if self.loss_fn is not None:
                        batch_loss, batch_metrics = self.loss_fn(batch, outputs)
                        loss_total += float(batch_loss.detach().cpu().item())
                        loss_count += 1
                        for key, value in batch_metrics.items():
                            if isinstance(value, torch.Tensor):
                                if value.numel() == 1:
                                    metric_value = float(value.item())
                                else:
                                    metric_value = float(value.mean().item())
                            elif isinstance(value, (int, float)):
                                metric_value = float(value)
                            else:
                                continue
                            metric_sums[key] = metric_sums.get(key, 0.0) + metric_value
                            metric_counts[key] = metric_counts.get(key, 0) + 1

                    if progress is not None and (
                        batch_idx == 1
                        or (
                            self.progress_every_batches is not None
                            and (batch_idx % self.progress_every_batches) == 0
                        )
                    ):
                        progress_loss = (
                            loss_total / float(loss_count)
                            if loss_count > 0
                            else float("nan")
                        )
                        progress.set_postfix(loss=f"{progress_loss:.4f}")
                    elif (
                        progress is None
                        and self.progress_every_batches is not None
                        and (batch_idx % self.progress_every_batches) == 0
                    ):
                        print(
                            f"epoch={epoch + 1} step={step} {self.log_prefix} {trigger} "
                            f"progress batches={batch_idx}"
                        )
        finally:
            if progress is not None:
                progress.close()

        if was_training:
            model.train()

        payload: dict[str, float | int | str] = {
            "train/epoch": int(epoch + 1),
            "train/global_step": int(step),
            f"{self.log_prefix}/trigger": trigger,
        }
        if loss_count > 0:
            payload[f"{self.log_prefix}/loss"] = loss_total / float(loss_count)
            for key, total in metric_sums.items():
                payload[f"{self.log_prefix}/{key}"] = total / float(
                    max(metric_counts.get(key, 1), 1)
                )
        for name in ("energy", "energy_per_atom", "forces", "stress"):
            count = mae_counts[name]
            if count > 0:
                payload[f"{self.log_prefix}/mae_{name}"] = mae_totals[name] / float(
                    count
                )

        mae_summary = ", ".join(
            f"{k}={payload[f'{self.log_prefix}/mae_{k}']:.6f}"
            for k in ("energy", "energy_per_atom", "forces", "stress")
            if f"{self.log_prefix}/mae_{k}" in payload
        )
        if not mae_summary:
            mae_summary = "none"
        print(
            f"epoch={epoch + 1} step={step} {self.log_prefix} {trigger} "
            f"finished batches={processed_batches} "
            f"loss={payload.get(f'{self.log_prefix}/loss', 'n/a')} mae={mae_summary}"
        )
        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=step)
        self.last_eval_step = step

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
        if self.last_eval_step == global_step:
            return
        self._run_eval(model=model, epoch=epoch, step=global_step, trigger="step")

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
        if self.last_eval_step == global_step:
            return
        self._run_eval(model=model, epoch=epoch, step=global_step, trigger="epoch")

    def on_train_end(
        self,
        *,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None:
        _ = epochs, global_step, model
