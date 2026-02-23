from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from triforces.evaluation.hooks import TrainHook, build_train_hooks
from triforces.losses import ContrastiveLoss

logger = logging.getLogger("triforces.train")
CHECKPOINT_SCHEMA_VERSION = 2


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


def _is_debug_enabled() -> bool:
    raw = os.getenv("DEBUG", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _configure_debug_logging(*, enabled: bool) -> None:
    if not enabled:
        return
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    root_logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)


def _cfg_target_name(value: object, fallback: str = "unknown") -> str:
    if isinstance(value, DictConfig):
        target = value.get("_target_")
        if target:
            return str(target)
    return fallback


def _resolve_wandb_name(wandb_cfg: DictConfig, *, run_name: str | None) -> str | None:
    explicit_name = wandb_cfg.get("name")
    if explicit_name is not None:
        name = str(explicit_name).strip()
        if name:
            return name
    if run_name is None:
        return None
    inferred = str(run_name).strip()
    return inferred or None


def _maybe_init_wandb(cfg: DictConfig, *, run_name: str | None = None) -> object | None:
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
        name=_resolve_wandb_name(wandb_cfg, run_name=run_name),
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


def _safe_path_component(value: str | None) -> str | None:
    if value is None:
        return None
    compact = re.sub(r"[^\w.-]+", "_", str(value).strip())
    compact = compact.strip("._")
    return compact or None


def _resolve_checkpoint_dir(
    checkpoint_cfg: DictConfig | None, *, run_name: str | None = None
) -> Path:
    raw_dir = (
        checkpoint_cfg.get("dir") if isinstance(checkpoint_cfg, DictConfig) else None
    )
    if raw_dir:
        path = Path(str(raw_dir))
        return path.resolve()

    default_root = Path.cwd() / "checkpoints"
    run_component = _safe_path_component(run_name)
    path = (default_root / run_component) if run_component else default_root
    return path.resolve()


def _path_from_optional_string(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _resolved_section_dict(cfg: DictConfig, key: str) -> dict[str, Any] | None:
    section = cfg.get(key)
    if section is None:
        return None
    resolved = OmegaConf.to_container(section, resolve=True)
    if isinstance(resolved, dict):
        return resolved
    return None


def _validate_resume_data_pipeline_consistency(
    *,
    launch_cfg: DictConfig,
    checkpoint_cfg: DictConfig,
    allow_override: bool,
) -> None:
    launch_dataset = _resolved_section_dict(launch_cfg, "dataset")
    launch_collate = _resolved_section_dict(launch_cfg, "collate")
    checkpoint_dataset = _resolved_section_dict(checkpoint_cfg, "dataset")
    checkpoint_collate = _resolved_section_dict(checkpoint_cfg, "collate")

    dataset_matches = launch_dataset == checkpoint_dataset
    collate_matches = launch_collate == checkpoint_collate
    if dataset_matches and collate_matches:
        return

    launch_dataset_target = (
        None if launch_dataset is None else launch_dataset.get("_target_")
    )
    checkpoint_dataset_target = (
        None if checkpoint_dataset is None else checkpoint_dataset.get("_target_")
    )
    launch_collate_target = (
        None if launch_collate is None else launch_collate.get("_target_")
    )
    checkpoint_collate_target = (
        None if checkpoint_collate is None else checkpoint_collate.get("_target_")
    )
    summary = (
        "dataset:"
        f" launch={launch_dataset_target!r} checkpoint={checkpoint_dataset_target!r}; "
        "collate:"
        f" launch={launch_collate_target!r} checkpoint={checkpoint_collate_target!r}"
    )
    if allow_override:
        logger.warning(
            "Resume data pipeline differs from checkpoint and override is enabled; "
            "using launch config. %s",
            summary,
        )
        return
    raise RuntimeError(
        "Resume data pipeline mismatch between launch config and checkpoint "
        "(dataset/collate, including graph creation settings). "
        f"{summary}. To override intentionally, set "
        "`train.checkpoint.allow_data_pipeline_override=true`."
    )


def _validate_collate_config(collate_cfg: object) -> None:
    if not isinstance(collate_cfg, DictConfig):
        raise ValueError(
            "Invalid `collate` config: expected a Hydra config with `_target_`."
        )
    target = collate_cfg.get("_target_")
    if not target:
        raise ValueError(
            "Invalid `collate` config: missing `_target_`. "
            "You likely set only `collate.contrastive=false` without choosing a collate preset."
        )


def _model_cfg_with_backbone_from_checkpoint(
    *,
    model_cfg: object,
    checkpoint_payload: dict[str, Any],
    checkpoint_path: Path,
) -> DictConfig:
    checkpoint_model_cfg = checkpoint_payload.get("model_config_resolved")
    if not isinstance(checkpoint_model_cfg, dict):
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is missing `model_config_resolved`."
        )
    checkpoint_backbone = checkpoint_model_cfg.get("backbone")
    if not isinstance(checkpoint_backbone, dict):
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is missing `model_config_resolved.backbone`."
        )
    if isinstance(model_cfg, DictConfig):
        model_container = OmegaConf.to_container(model_cfg, resolve=False)
    elif isinstance(model_cfg, dict):
        model_container = dict(model_cfg)
    else:
        raise RuntimeError("Model config must be a mapping.")
    if not isinstance(model_container, dict):
        raise RuntimeError("Resolved model config must be a dictionary.")
    model_container["backbone"] = checkpoint_backbone
    wrapped = OmegaConf.create({"model": model_container})
    updated = wrapped.get("model")
    if not isinstance(updated, DictConfig):
        raise RuntimeError("Failed to build model config using checkpoint backbone.")
    return updated


def _validate_checkpoint_settings(
    checkpoint_cfg: DictConfig | None, *, enabled: bool
) -> None:
    if not enabled or not isinstance(checkpoint_cfg, DictConfig):
        return

    resume_path = _path_from_optional_string(checkpoint_cfg.get("resume_from"))
    init_path = _path_from_optional_string(checkpoint_cfg.get("init_from"))
    if resume_path is not None and init_path is not None:
        raise ValueError(
            "Checkpoint config conflict: set only one of `train.checkpoint.resume_from` "
            "or `train.checkpoint.init_from`."
        )

    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(
            f"`train.checkpoint.resume_from` does not exist: {resume_path}"
        )
    if resume_path is not None and not resume_path.is_file():
        raise ValueError(
            f"`train.checkpoint.resume_from` must point to a checkpoint file: {resume_path}"
        )
    if init_path is not None and not init_path.exists():
        raise FileNotFoundError(
            f"`train.checkpoint.init_from` does not exist: {init_path}"
        )
    if init_path is not None and not init_path.is_file():
        raise ValueError(
            f"`train.checkpoint.init_from` must point to a checkpoint file: {init_path}"
        )

    init_mode = str(checkpoint_cfg.get("init_mode", "full")).strip().lower()
    if init_mode not in {"full", "backbone"}:
        raise ValueError(
            f"Invalid `train.checkpoint.init_mode={init_mode!r}`. "
            "Use `full` or `backbone`."
        )
    init_use_backbone_config = bool(
        checkpoint_cfg.get("init_use_backbone_config", False)
    )
    if init_use_backbone_config and init_path is None:
        raise ValueError(
            "`train.checkpoint.init_use_backbone_config=true` requires "
            "`train.checkpoint.init_from`."
        )
    if init_use_backbone_config and init_mode != "backbone":
        raise ValueError(
            "`train.checkpoint.init_use_backbone_config=true` requires "
            "`train.checkpoint.init_mode=backbone`."
        )

    monitor_mode = str(checkpoint_cfg.get("mode", "min")).strip().lower()
    if monitor_mode not in {"min", "max"}:
        raise ValueError(
            f"Invalid `train.checkpoint.mode={monitor_mode!r}`. Use `min` or `max`."
        )

    save_every_epochs = int(checkpoint_cfg.get("save_every_epochs", 1))
    if save_every_epochs < 1:
        raise ValueError(
            "`train.checkpoint.save_every_epochs` must be >= 1. "
            "Set `train.checkpoint.enabled=false` to disable checkpointing."
        )

    save_last_every_steps = int(checkpoint_cfg.get("save_last_every_steps", 1))
    if save_last_every_steps < 1:
        raise ValueError(
            "`train.checkpoint.save_last_every_steps` must be >= 1. "
            "Set `train.checkpoint.save_last=false` to disable last-step checkpointing."
        )

    keep_last_n = int(checkpoint_cfg.get("keep_last_n", 0) or 0)
    if keep_last_n < 0:
        raise ValueError("`train.checkpoint.keep_last_n` must be >= 0.")


def _build_loss(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    loss_cfg = cfg.get("loss")
    return _build_loss_from_cfg(loss_cfg, device)


def _build_loss_from_cfg(loss_cfg: object, device: torch.device) -> torch.nn.Module:
    if isinstance(loss_cfg, dict):
        loss_cfg = OmegaConf.create(loss_cfg)
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


def _load_checkpoint_payload(*, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint at {path} is not a dictionary payload.")
    return payload


def _config_from_checkpoint_payload(
    *, payload: dict[str, Any], path: Path
) -> DictConfig:
    resolved_cfg = payload.get("config_resolved")
    if not isinstance(resolved_cfg, dict):
        raise RuntimeError(
            f"Checkpoint at {path} is missing `config_resolved` metadata. "
            "Legacy checkpoints are not supported."
        )
    cfg = OmegaConf.create(resolved_cfg)
    if not isinstance(cfg, DictConfig):
        raise RuntimeError(
            f"Checkpoint at {path} has invalid `config_resolved` metadata."
        )
    return cfg


def _checkpoint_metadata_from_cfg(
    *,
    cfg: DictConfig,
    model_cfg: object | None = None,
    loss_cfg: object | None = None,
) -> dict[str, Any]:
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        raise RuntimeError("Resolved training config must be a dictionary.")
    if model_cfg is not None:
        if isinstance(model_cfg, dict):
            resolved_model_cfg = dict(model_cfg)
        else:
            resolved_model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        if not isinstance(resolved_model_cfg, dict):
            raise RuntimeError("Resolved model config must be a dictionary.")
        resolved_cfg["model"] = resolved_model_cfg
    if loss_cfg is not None:
        if isinstance(loss_cfg, dict):
            resolved_loss_cfg = dict(loss_cfg)
        else:
            resolved_loss_cfg = OmegaConf.to_container(loss_cfg, resolve=True)
        if not isinstance(resolved_loss_cfg, dict):
            raise RuntimeError("Resolved loss config must be a dictionary.")
        resolved_cfg["loss"] = resolved_loss_cfg
    resolved_model = resolved_cfg.get("model")
    if not isinstance(resolved_model, dict):
        raise RuntimeError(
            "Resolved training config is missing a valid `model` section."
        )
    resolved_loss = resolved_cfg.get("loss")
    if not isinstance(resolved_loss, dict):
        raise RuntimeError(
            "Resolved training config is missing a valid `loss` section."
        )
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_resolved": resolved_cfg,
        "model_config_resolved": resolved_model,
        "loss_config_resolved": resolved_loss,
    }


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
    extra_payload: dict[str, Any] | None = None,
) -> None:
    # Be resilient to external cleanup while training (e.g. deleted checkpoint dir).
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, path)
    logger.debug(
        "Saved checkpoint path=%s epoch=%d global_step=%d best_metric=%s",
        path,
        epoch,
        global_step,
        best_metric,
    )


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
        logger.debug("Loaded checkpoint weights mode=full strict=%s", bool(strict))
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
    logger.debug(
        "Loaded checkpoint weights mode=backbone strict=%s backbone_keys=%d",
        bool(strict),
        len(source_backbone),
    )
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
    payload: dict[str, Any] | None = None,
) -> tuple[int, int, float | None]:
    if payload is None:
        payload = _load_checkpoint_payload(path=path)
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
    logger.debug(
        "Loaded training checkpoint path=%s start_epoch=%d global_step=%d best_metric=%s",
        path,
        epoch + 1,
        global_step,
        best_metric,
    )
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
    checkpoint_last_path: Path | None = None,
    checkpoint_best_metric: float | None = None,
    checkpoint_last_every_steps: int = 1,
    checkpoint_extra_payload: dict[str, Any] | None = None,
    train_hooks: list[TrainHook] | None = None,
    debug_enabled: bool = False,
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
        if (
            checkpoint_last_path is not None
            and checkpoint_last_every_steps > 0
            and (global_step % checkpoint_last_every_steps) == 0
        ):
            _save_checkpoint(
                path=checkpoint_last_path,
                model=model,
                optim=optim,
                loss_fn=loss_fn,
                epoch=epoch,
                global_step=global_step,
                best_metric=checkpoint_best_metric,
                extra_payload=checkpoint_extra_payload,
            )
        if wandb_run is not None and (global_step == 1 or global_step % log_every == 0):
            wandb_payload = {f"train/{k}": v for k, v in batch_metrics.items()}
            wandb_payload["train/epoch"] = epoch + 1
            wandb_run.log(wandb_payload, step=global_step)
        if debug_enabled and (global_step == 1 or global_step % log_every == 0):
            logger.debug(
                "train_step epoch=%d/%d global_step=%d loss=%.6f metrics=%s",
                epoch + 1,
                num_epochs,
                global_step,
                loss_value,
                batch_metrics,
            )
        if train_hooks:
            for hook in train_hooks:
                hook.on_step_end(
                    epoch=epoch,
                    epochs=num_epochs,
                    global_step=global_step,
                    model=model,
                )
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
    debug_enabled = _is_debug_enabled()
    _configure_debug_logging(enabled=debug_enabled)
    if debug_enabled:
        logger.debug("DEBUG mode enabled for train run")

    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_cfg = cfg.get("train")
    run_name_raw = _get_train_value(train_cfg, "run_name", None)
    run_name = None if run_name_raw is None else str(run_name_raw).strip() or None
    epochs = int(_get_train_value(train_cfg, "epochs", 1))
    batch_size = int(_get_train_value(train_cfg, "batch_size", 1))
    lr = float(_get_train_value(train_cfg, "lr", 1e-3))
    if debug_enabled:
        logger.debug(
            "Train config device=%s epochs=%d batch_size=%d lr=%s run_name=%s",
            device,
            epochs,
            batch_size,
            lr,
            run_name,
        )
        logger.debug(
            "Config targets dataset=%s dataset_val=%s collate=%s model=%s loss=%s",
            _cfg_target_name(cfg.get("dataset")),
            _cfg_target_name(cfg.get("dataset_val"), fallback="none"),
            _cfg_target_name(cfg.get("collate")),
            _cfg_target_name(cfg.get("model")),
            _cfg_target_name(cfg.get("loss")),
        )

    checkpoint_cfg = (
        train_cfg.get("checkpoint") if isinstance(train_cfg, DictConfig) else None
    )
    checkpoint_enabled = bool(
        checkpoint_cfg is not None and checkpoint_cfg.get("enabled", False)
    )
    _validate_checkpoint_settings(checkpoint_cfg, enabled=checkpoint_enabled)
    resume_payload: dict[str, Any] | None = None
    init_payload: dict[str, Any] | None = None
    resume_path: Path | None = None
    init_path: Path | None = None
    model_cfg_for_instantiation = cfg.get("model")
    loss_cfg_for_instantiation = cfg.get("loss")
    if checkpoint_enabled and checkpoint_cfg is not None:
        resume_path = _path_from_optional_string(checkpoint_cfg.get("resume_from"))
        init_path = _path_from_optional_string(checkpoint_cfg.get("init_from"))
        init_mode = str(checkpoint_cfg.get("init_mode", "full")).strip().lower()
        init_use_backbone_config = bool(
            checkpoint_cfg.get("init_use_backbone_config", False)
        )
        if resume_path is not None:
            if debug_enabled:
                logger.debug(
                    "Loading checkpoint metadata for resume path=%s", resume_path
                )
            resume_payload = _load_checkpoint_payload(path=resume_path)
            checkpoint_cfg_resolved = _config_from_checkpoint_payload(
                payload=resume_payload, path=resume_path
            )
            model_cfg_candidate = checkpoint_cfg_resolved.get("model")
            if not isinstance(model_cfg_candidate, DictConfig):
                raise RuntimeError(
                    f"Checkpoint at {resume_path} has invalid `config_resolved.model`."
                )
            loss_cfg_candidate = checkpoint_cfg_resolved.get("loss")
            if not isinstance(loss_cfg_candidate, DictConfig):
                raise RuntimeError(
                    f"Checkpoint at {resume_path} has invalid `config_resolved.loss`."
                )
            _validate_resume_data_pipeline_consistency(
                launch_cfg=cfg,
                checkpoint_cfg=checkpoint_cfg_resolved,
                allow_override=bool(
                    checkpoint_cfg.get("allow_data_pipeline_override", False)
                ),
            )
            model_cfg_for_instantiation = model_cfg_candidate
            loss_cfg_for_instantiation = loss_cfg_candidate
            if debug_enabled:
                logger.debug(
                    "Using model/loss configs from checkpoint for instantiation path=%s",
                    resume_path,
                )
        elif init_path is not None and init_use_backbone_config:
            if debug_enabled:
                logger.debug(
                    "Loading checkpoint backbone config for init path=%s mode=%s",
                    init_path,
                    init_mode,
                )
            init_payload = _load_checkpoint_payload(path=init_path)
            model_cfg_for_instantiation = _model_cfg_with_backbone_from_checkpoint(
                model_cfg=model_cfg_for_instantiation,
                checkpoint_payload=init_payload,
                checkpoint_path=init_path,
            )
            if debug_enabled:
                logger.debug(
                    "Using checkpoint backbone config for model instantiation path=%s",
                    init_path,
                )

    dataset = instantiate(cfg.dataset, _convert_="object")
    dataset_val_cfg = cfg.get("dataset_val")
    dataset_val = (
        instantiate(dataset_val_cfg, _convert_="object")
        if isinstance(dataset_val_cfg, DictConfig)
        else None
    )

    _validate_collate_config(cfg.get("collate"))
    collate_fn = instantiate(cfg.collate, _convert_="object")
    model = instantiate(model_cfg_for_instantiation, _convert_="object").to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    loss_fn = _build_loss_from_cfg(loss_cfg_for_instantiation, device)

    optim = torch.optim.AdamW(list(model.parameters()), lr=lr)
    checkpoint_extra_payload: dict[str, Any] | None = None
    if checkpoint_enabled:
        checkpoint_extra_payload = _checkpoint_metadata_from_cfg(
            cfg=cfg,
            model_cfg=model_cfg_for_instantiation,
            loss_cfg=loss_cfg_for_instantiation,
        )
    checkpoint_dir: Path | None = None
    save_every_epochs = 1
    save_last = True
    save_last_every_steps = 1
    save_best = True
    keep_last_n = 0
    monitor = "loss"
    monitor_mode = "min"
    if checkpoint_enabled:
        checkpoint_dir = _resolve_checkpoint_dir(checkpoint_cfg, run_name=run_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"checkpoint_dir={checkpoint_dir}")
        save_every_epochs = int(checkpoint_cfg.get("save_every_epochs", 1))
        save_last = bool(checkpoint_cfg.get("save_last", True))
        save_last_every_steps = int(checkpoint_cfg.get("save_last_every_steps", 1))
        save_best = bool(checkpoint_cfg.get("save_best", True))
        keep_last_n = int(checkpoint_cfg.get("keep_last_n", 0) or 0)
        monitor = str(checkpoint_cfg.get("monitor", "loss"))
        monitor_mode = str(checkpoint_cfg.get("mode", "min")).lower()
        if debug_enabled:
            logger.debug(
                "Checkpoint config dir=%s save_every_epochs=%d save_last=%s "
                "save_last_every_steps=%d save_best=%s keep_last_n=%d monitor=%s mode=%s",
                checkpoint_dir,
                save_every_epochs,
                save_last,
                save_last_every_steps,
                save_best,
                keep_last_n,
                monitor,
                monitor_mode,
            )
        checkpoint_action = "fresh"
        if resume_path is not None:
            checkpoint_action = f"resume:{resume_path}"
        elif init_path is not None:
            backbone_config_note = (
                " checkpoint_backbone_config=true"
                if bool(checkpoint_cfg.get("init_use_backbone_config", False))
                else ""
            )
            checkpoint_action = (
                f"init:{init_path} mode={str(checkpoint_cfg.get('init_mode', 'full'))}"
                f"{backbone_config_note}"
            )
        print(
            "checkpoint_policy="
            f"action={checkpoint_action} "
            f"save_last_every_steps={save_last_every_steps} "
            f"save_every_epochs={save_every_epochs} save_best={save_best}"
        )

    log_every = int(_get_train_value(train_cfg, "log_every", 10))
    if log_every < 1:
        log_every = 1
    use_tqdm = bool(_get_train_value(train_cfg, "tqdm", True))

    wandb_run = _maybe_init_wandb(cfg, run_name=run_name)
    if wandb_run is not None:
        num_params = sum(p.numel() for p in model.parameters())
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
        dataset_val_size = (
            len(dataset_val)
            if dataset_val is not None and hasattr(dataset_val, "__len__")
            else None
        )
        wandb_run.config.update(
            {
                "num_parameters": num_params,
                "dataset_size": dataset_size,
                "dataset_val_size": dataset_val_size,
                "run_name": run_name,
                "checkpoint_enabled": checkpoint_enabled,
                "checkpoint_dir": (
                    str(checkpoint_dir)
                    if checkpoint_enabled and checkpoint_dir is not None
                    else None
                ),
            },
            allow_val_change=True,
        )

    start_epoch = 0
    global_step = 0
    best_metric: float | None = None

    if checkpoint_enabled:
        if resume_path is not None:
            if debug_enabled:
                logger.debug("Resuming training from checkpoint path=%s", resume_path)
            start_epoch, global_step, best_metric = _load_training_checkpoint(
                path=resume_path,
                model=model,
                optim=optim,
                loss_fn=loss_fn,
                strict=bool(checkpoint_cfg.get("resume_strict", True)),
                payload=resume_payload,
            )
        else:
            if init_path is not None:
                if debug_enabled:
                    logger.debug(
                        "Initializing model weights from checkpoint path=%s", init_path
                    )
                payload = (
                    init_payload
                    if init_payload is not None
                    else _load_checkpoint_payload(path=init_path)
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

    train_hooks = build_train_hooks(
        train_cfg=train_cfg,
        dataset=dataset,
        dataset_val=dataset_val,
        collate_fn=collate_fn,
        train_batch_size=batch_size,
        device=device,
        wandb_run=wandb_run,
        loss_fn=loss_fn,
    )
    if debug_enabled:
        logger.debug("Built %d train hooks", len(train_hooks))
    for hook in train_hooks:
        hook.on_train_start(
            start_epoch=start_epoch,
            epochs=epochs,
            global_step=global_step,
            model=model,
        )

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
            checkpoint_last_path=(
                checkpoint_dir / "last.pt"
                if checkpoint_enabled and checkpoint_dir is not None and save_last
                else None
            ),
            checkpoint_best_metric=best_metric,
            checkpoint_last_every_steps=save_last_every_steps,
            checkpoint_extra_payload=checkpoint_extra_payload,
            train_hooks=train_hooks,
            debug_enabled=debug_enabled,
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
        for hook in train_hooks:
            hook.on_epoch_end(
                epoch=epoch,
                epochs=epochs,
                global_step=global_step,
                model=model,
            )

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
                    extra_payload=checkpoint_extra_payload,
                )
                _trim_epoch_checkpoints(checkpoint_dir, keep_last_n)

            if save_last:
                last_already_saved_at_step = (
                    save_last_every_steps > 0
                    and global_step > 0
                    and (global_step % save_last_every_steps) == 0
                )
                if not last_already_saved_at_step:
                    _save_checkpoint(
                        path=checkpoint_dir / "last.pt",
                        model=model,
                        optim=optim,
                        loss_fn=loss_fn,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        extra_payload=checkpoint_extra_payload,
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
                    extra_payload=checkpoint_extra_payload,
                )

    for hook in train_hooks:
        hook.on_train_end(
            epochs=epochs,
            global_step=global_step,
            model=model,
        )

    if wandb_run is not None:
        wandb_run.finish()

    return 0
