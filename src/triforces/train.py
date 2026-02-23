from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from triforces.losses import ContrastiveLoss


def _to_float_metrics(metrics: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                out[key] = float(value.item())
            else:
                out[key] = float(value.mean().item())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _normalize_metrics_dict(metrics: dict[str, float]) -> dict[str, float]:
    """Normalize metric keys for cleaner logging namespaces."""
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        name = str(key).strip()
        if not name:
            continue
        if name.startswith("loss/") or "/" in name:
            normalized[name] = float(value)
            continue
        if name == "total_loss":
            normalized["loss/total"] = float(value)
            continue
        if name.endswith("_loss"):
            normalized[f"loss/{name[:-5]}"] = float(value)
            continue
        normalized[name] = float(value)
    return normalized


def _is_target_cfg(value: object) -> bool:
    return isinstance(value, DictConfig) and value.get("_target_") is not None


def _maybe_init_wandb(cfg: DictConfig) -> object | None:
    wandb_cfg = cfg.get("logger")
    if wandb_cfg is None:
        return None

    enabled = wandb_cfg.get("enabled", True)
    if not bool(enabled):
        return None

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    return wandb.init(
        project=wandb_cfg.get("project", "triforces"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        group=wandb_cfg.get("group"),
        tags=wandb_cfg.get("tags"),
        job_type=wandb_cfg.get("job_type"),
        mode=wandb_cfg.get("mode"),
        config=resolved_cfg,
    )


def _get_train_value(train_cfg: DictConfig | None, key: str, default: object) -> object:
    if not isinstance(train_cfg, DictConfig):
        return default
    value = train_cfg.get(key, default)
    if value is None:
        return default
    return value


def _build_loss(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    loss_cfg = cfg.get("loss")
    if _is_target_cfg(loss_cfg):
        loss = instantiate(loss_cfg, _convert_="object")
        return loss.to(device) if hasattr(loss, "to") else loss

    loss = ContrastiveLoss(
        temperature_node=(
            loss_cfg.get("temperature_node", 0.07)
            if isinstance(loss_cfg, DictConfig)
            else 0.07
        ),
        temperature_graph=(
            loss_cfg.get("temperature_graph", 0.1)
            if isinstance(loss_cfg, DictConfig)
            else 0.1
        ),
        lambda_node=(
            loss_cfg.get("lambda_node", 0.0)
            if isinstance(loss_cfg, DictConfig)
            else 0.0
        ),
        lambda_graph=(
            loss_cfg.get("lambda_graph", 1.0)
            if isinstance(loss_cfg, DictConfig)
            else 1.0
        ),
        max_negatives=(
            loss_cfg.get("max_negatives", 1024)
            if isinstance(loss_cfg, DictConfig)
            else 1024
        ),
        similarity_metric=(
            loss_cfg.get("similarity_metric", "cosine")
            if isinstance(loss_cfg, DictConfig)
            else "cosine"
        ),
    )
    return loss.to(device)


def _checkpoint_state_for_loss(loss_fn: torch.nn.Module) -> dict[str, Any] | None:
    get_state = getattr(loss_fn, "get_checkpoint_state", None)
    if callable(get_state):
        state = get_state()
        if isinstance(state, dict):
            return state
    return None


def _load_checkpoint_state_for_loss(
    loss_fn: torch.nn.Module, state: dict[str, Any] | None
) -> None:
    load_state = getattr(loss_fn, "load_checkpoint_state", None)
    if callable(load_state):
        load_state(state or {})


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epoch: int,
    global_step: int,
    best_metric: float | None,
) -> None:
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "loss_state_dict": (
            loss_fn.state_dict() if hasattr(loss_fn, "state_dict") else {}
        ),
        "best_metric": best_metric,
    }
    loss_checkpoint_state = _checkpoint_state_for_loss(loss_fn)
    if loss_checkpoint_state:
        payload["loss_checkpoint_state"] = loss_checkpoint_state
    torch.save(payload, path)


def _trim_epoch_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    epoch_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
    excess = len(epoch_files) - keep_last_n
    for path in epoch_files[: max(excess, 0)]:
        path.unlink(missing_ok=True)


def _load_checkpoint_weights(
    *,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    mode: str = "full",
    strict: bool = True,
) -> None:
    source_state = checkpoint.get("model_state_dict")
    if source_state is None:
        raise KeyError("Checkpoint is missing `model_state_dict`.")

    mode = str(mode)
    if mode == "full":
        model.load_state_dict(source_state, strict=bool(strict))
        return

    if mode != "backbone":
        raise ValueError(f"Unsupported init mode {mode!r}. Use 'full' or 'backbone'.")

    target_state = model.state_dict()
    source_backbone = {
        key: value for key, value in source_state.items() if key.startswith("backbone.")
    }
    if not source_backbone:
        raise ValueError(
            "Backbone init mode requested, but no `backbone.*` keys were found in the checkpoint."
        )
    target_state.update(source_backbone)
    incompatible = model.load_state_dict(target_state, strict=False)
    if strict and incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys while loading backbone weights: "
            f"{incompatible.unexpected_keys}"
        )


def _resolve_monitored_value(
    *,
    monitor_key: str,
    epoch_loss: float,
    avg_metrics: dict[str, float],
) -> float:
    key = str(monitor_key).strip()
    if key in {"loss", "loss_epoch", "train/loss_epoch"}:
        return float(epoch_loss)
    if key in avg_metrics:
        return float(avg_metrics[key])
    prefixed = f"loss/{key}"
    if prefixed in avg_metrics:
        return float(avg_metrics[prefixed])
    return float(epoch_loss)


def _load_training_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    strict: bool,
) -> tuple[int, int, float | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=bool(strict))
    if "optimizer_state_dict" in payload:
        optim.load_state_dict(payload["optimizer_state_dict"])
    if "loss_state_dict" in payload and hasattr(loss_fn, "load_state_dict"):
        loss_fn.load_state_dict(payload["loss_state_dict"], strict=False)
    if "loss_checkpoint_state" in payload:
        _load_checkpoint_state_for_loss(loss_fn, payload["loss_checkpoint_state"])

    epoch = int(payload.get("epoch", -1))
    global_step = int(payload.get("global_step", 0))
    best_metric = payload.get("best_metric")
    if best_metric is not None:
        best_metric = float(best_metric)
    return epoch + 1, global_step, best_metric


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    global_step: int,
    log_every: int,
    wandb_run: object | None,
    use_tqdm: bool,
):
    model.train()

    total = 0.0
    n = 0
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    iterator = loader
    if use_tqdm:
        iterator = tqdm(
            loader, desc=f"train {epoch + 1}/{num_epochs}", total=len(loader)
        )
    for batch in iterator:
        batch = batch.to(device)
        out = model(batch, training=True)

        loss, metrics = loss_fn(batch, out)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        loss_value = float(loss.detach().cpu().item())
        batch_metrics = _to_float_metrics(metrics or {})
        batch_metrics["loss"] = loss_value
        batch_metrics = _normalize_metrics_dict(batch_metrics)
        total += loss_value
        n += 1
        global_step += 1
        if wandb_run is not None and (global_step == 1 or global_step % log_every == 0):
            wandb_payload = {f"train/{k}": v for k, v in batch_metrics.items()}
            wandb_payload["train/epoch"] = epoch + 1
            wandb_run.log(wandb_payload, step=global_step)
        if use_tqdm and (n == 1 or n % log_every == 0):
            iterator.set_postfix(loss=f"{loss_value:.4f}")
        for key, value in batch_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            metric_counts[key] = metric_counts.get(key, 0) + 1

    avg_metrics = {}
    for key, total_value in metric_sums.items():
        count = metric_counts.get(key, 1)
        avg_metrics[key] = total_value / max(count, 1)
    return total / max(n, 1), global_step, avg_metrics


def run(cfg: DictConfig) -> int:
    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_cfg = cfg.get("train")
    epochs = int(_get_train_value(train_cfg, "epochs", 1))
    batch_size = int(_get_train_value(train_cfg, "batch_size", 1))
    lr = float(_get_train_value(train_cfg, "lr", 1e-3))

    dataset = instantiate(cfg.dataset, _convert_="object")

    collate_fn = instantiate(cfg.collate, _convert_="object")
    model = instantiate(cfg.model, _convert_="object").to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    loss_fn = _build_loss(cfg, device)

    optim = torch.optim.AdamW(list(model.parameters()), lr=lr)

    checkpoint_cfg = train_cfg.get("checkpoint") if isinstance(train_cfg, DictConfig) else None
    checkpoint_enabled = bool(
        checkpoint_cfg is not None and checkpoint_cfg.get("enabled", False)
    )
    checkpoint_dir: Path | None = None
    save_every_epochs = 1
    save_last = True
    save_best = True
    keep_last_n = 0
    monitor = "loss"
    monitor_mode = "min"
    if checkpoint_enabled:
        checkpoint_dir = Path(
            str(checkpoint_cfg.get("dir") or Path.cwd() / "checkpoints")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_every_epochs = int(checkpoint_cfg.get("save_every_epochs", 1))
        if save_every_epochs < 1:
            save_every_epochs = 1
        save_last = bool(checkpoint_cfg.get("save_last", True))
        save_best = bool(checkpoint_cfg.get("save_best", True))
        keep_last_n = int(checkpoint_cfg.get("keep_last_n", 0) or 0)
        monitor = str(checkpoint_cfg.get("monitor", "loss"))
        monitor_mode = str(checkpoint_cfg.get("mode", "min")).lower()
        if monitor_mode not in {"min", "max"}:
            monitor_mode = "min"

    log_every = int(_get_train_value(train_cfg, "log_every", 10))
    if log_every < 1:
        log_every = 1
    use_tqdm = bool(_get_train_value(train_cfg, "tqdm", True))

    wandb_run = _maybe_init_wandb(cfg)
    if wandb_run is not None:
        num_params = sum(p.numel() for p in model.parameters())
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
        wandb_run.config.update(
            {"num_parameters": num_params, "dataset_size": dataset_size},
            allow_val_change=True,
        )

    start_epoch = 0
    global_step = 0
    best_metric: float | None = None

    if checkpoint_enabled:
        resume_from = checkpoint_cfg.get("resume_from")
        if resume_from:
            start_epoch, global_step, best_metric = _load_training_checkpoint(
                path=Path(str(resume_from)),
                model=model,
                optim=optim,
                loss_fn=loss_fn,
                strict=bool(checkpoint_cfg.get("resume_strict", True)),
            )
        else:
            init_from = checkpoint_cfg.get("init_from")
            if init_from:
                payload = torch.load(
                    Path(str(init_from)), map_location="cpu", weights_only=False
                )
                _load_checkpoint_weights(
                    model=model,
                    checkpoint=payload,
                    mode=str(checkpoint_cfg.get("init_mode", "full")),
                    strict=bool(checkpoint_cfg.get("init_strict", False)),
                )
                loss_checkpoint_state = payload.get("loss_checkpoint_state")
                if isinstance(loss_checkpoint_state, dict):
                    _load_checkpoint_state_for_loss(loss_fn, loss_checkpoint_state)

    for epoch in range(start_epoch, epochs):
        avg, global_step, avg_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            loader=loader,
            optim=optim,
            device=device,
            epoch=epoch,
            num_epochs=epochs,
            global_step=global_step,
            log_every=log_every,
            wandb_run=wandb_run,
            use_tqdm=use_tqdm,
        )
        print(f"epoch={epoch + 1} loss={avg:.4f}")
        if wandb_run is not None:
            epoch_payload = {
                "train/loss_epoch": avg,
                "train/epoch": epoch + 1,
            }
            for key, value in avg_metrics.items():
                if key == "loss":
                    continue
                epoch_payload[f"train_epoch/{key}"] = value
            wandb_run.log(epoch_payload, step=global_step)

        if checkpoint_enabled and checkpoint_dir is not None:
            monitored_value = _resolve_monitored_value(
                monitor_key=monitor,
                epoch_loss=avg,
                avg_metrics=avg_metrics,
            )
            improved = False
            if best_metric is None:
                improved = True
            elif monitor_mode == "min":
                improved = monitored_value < best_metric
            else:
                improved = monitored_value > best_metric
            if improved:
                best_metric = monitored_value

            if (epoch + 1) % save_every_epochs == 0:
                _save_checkpoint(
                    path=checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                    model=model,
                    optim=optim,
                    loss_fn=loss_fn,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                )
                _trim_epoch_checkpoints(checkpoint_dir, keep_last_n)

            if save_last:
                _save_checkpoint(
                    path=checkpoint_dir / "last.pt",
                    model=model,
                    optim=optim,
                    loss_fn=loss_fn,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                )

            if save_best and improved:
                _save_checkpoint(
                    path=checkpoint_dir / "best.pt",
                    model=model,
                    optim=optim,
                    loss_fn=loss_fn,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                )

    if wandb_run is not None:
        wandb_run.finish()

    return 0
