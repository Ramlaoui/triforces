# TriForces

Triforces extends atomistic models with self-supervised training for transferable representations.

## Install

```bash
uv sync --extra orb
```

## Basic commands

### 1) Pretrain (Orb)

```bash
triforces train \
  -cn experiments/pretraining/orb/main_triforces \
  train.epochs=10 \
  train.batch_size=16
```

### 2) Supervised from scratch (Orb)

```bash
triforces train \
  -cn experiments/supervised/orb/energy_conserving \
  train.epochs=10 \
  train.batch_size=16
```

### 3) Supervised initialized from a pretrained backbone

```bash
triforces train \
  -cn experiments/supervised/orb/energy_conserving \
  train.checkpoint.init_from=/absolute/path/to/pretrain/best.pt \
  train.checkpoint.init_mode=backbone \
  train.checkpoint.init_use_backbone_config=true \
  train.checkpoint.init_strict=false
```

### 4) Resume an interrupted run

```bash
triforces train \
  -cn experiments/pretraining/orb/main_triforces \
  train.checkpoint.resume_from=/absolute/path/to/run/last.pt
```

## Checkpoint UX

Use these keys under `train.checkpoint`:

- `enabled`: turn checkpointing on/off.
- `save_last_every_steps`: frequency for updating `last.pt`.
- `save_every_epochs`: frequency for `epoch_XXXX.pt`.
- `save_best`: save `best.pt` from monitored metric.
- `monitor`: metric key to monitor (default: `loss`).
- `mode`: `min` or `max`.
- `init_from`: load weights before training (`full` or `backbone`).
- `init_use_backbone_config`: when `true` with `init_mode=backbone`, instantiate using
  the backbone config stored in checkpoint metadata.
- `resume_from`: continue optimizer/model/loss states from a previous run.
- `allow_data_pipeline_override`: if `false` (default), `resume_from` requires dataset+collate
  config to match checkpoint metadata.

Notes:

- `resume_from` and `init_from` are mutually exclusive.
- `resume_from` expects a new-format checkpoint containing `config_resolved`.
