from __future__ import annotations

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


def _is_target_cfg(value: object) -> bool:
    return isinstance(value, DictConfig) and value.get("_target_") is not None


def _maybe_init_wandb(cfg: DictConfig) -> object | None:
    wandb_cfg = cfg.get("wandb")
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


def _get_train_value(
    cfg: DictConfig, train_cfg: DictConfig | None, key: str, default: object
) -> object:
    if train_cfg is None:
        return cfg.get(key, default)
    value = train_cfg.get(key, default)
    if value is None:
        return cfg.get(key, default)
    return value


def _build_loss(cfg: DictConfig, device: torch.device) -> ContrastiveLoss:
    loss_cfg = cfg.get("loss")
    if _is_target_cfg(loss_cfg):
        loss = instantiate(loss_cfg, _convert_="object")
        return loss.to(device) if hasattr(loss, "to") else loss

    def _loss_value(key: str, default: object) -> object:
        if isinstance(loss_cfg, DictConfig):
            value = loss_cfg.get(key, default)
            if value is not None:
                return value
        return cfg.get(key, default)

    loss = ContrastiveLoss(
        temperature_node=_loss_value("temperature_node", 0.07),
        temperature_graph=_loss_value("temperature_graph", 0.1),
        lambda_node=_loss_value("lambda_node", 0.0),
        lambda_graph=_loss_value("lambda_graph", 1.0),
        max_negatives=_loss_value("max_negatives", 1024),
        similarity_metric=_loss_value("similarity_metric", "cosine"),
    )
    return loss.to(device)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loss_fn: ContrastiveLoss,
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
    epochs = int(_get_train_value(cfg, train_cfg, "epochs", 1))
    batch_size = int(_get_train_value(cfg, train_cfg, "batch_size", 1))
    lr = float(_get_train_value(cfg, train_cfg, "lr", 1e-3))

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

    log_every = int(_get_train_value(cfg, train_cfg, "log_every", 10))
    if log_every < 1:
        log_every = 1
    use_tqdm = bool(_get_train_value(cfg, train_cfg, "tqdm", True))

    wandb_run = _maybe_init_wandb(cfg)
    if wandb_run is not None:
        num_params = sum(p.numel() for p in model.parameters())
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
        wandb_run.config.update(
            {"num_parameters": num_params, "dataset_size": dataset_size},
            allow_val_change=True,
        )

    global_step = 0
    for epoch in range(epochs):
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

    if wandb_run is not None:
        wandb_run.finish()

    return 0
