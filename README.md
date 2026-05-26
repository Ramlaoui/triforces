# TriForces

Triforces extends atomistic models with self-supervised training for transferable representations.

Paper: [TriForces: Augmenting Atomistic GNNs for Transferable Representations](https://arxiv.org/abs/2605.20581)

## Install

```bash
uv sync --extra orb
```

## Basic commands

### 1) Pretrain

```bash
triforces train \
  -cn experiments/pretraining/orb/main_triforces \
  train.epochs=10 \
  train.batch_size=16
```

### 2) Supervised from scratch
  
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
  train.checkpoint.init_from=</absolute/path/to/pretrain/best.pt> \
  train.checkpoint.init_mode=backbone \
  train.checkpoint.init_use_backbone_config=true \
  train.checkpoint.init_strict=false
```

### 4) Resume an interrupted run

```bash
triforces train \
  -cn experiments/pretraining/orb/main_triforces \
  train.checkpoint.resume_from=</absolute/path/to/run/last.pt>
```

## Hydra overrides

TriForces uses Hydra configs from `src/triforces/configs`.
Pick a config with `-cn ...`, then override any key inline:

```bash
triforces train \
  -cn experiments/supervised/orb/energy_conserving \
  train.epochs=50 \
  train.batch_size=32 \
  backbone.interaction.model_type=orb-v3-direct
```

You can override dataset, model, loss, logger, and trainer values the same way.

## Wrapping an existing model

The minimal wiring is:

1. Create your own backbone module with `forward(batch, training=False, transform=None)`.
2. Inside that `forward`, pass `batch` to your internal model (and optionally modify/enrich `batch` first if your model needs it).
3. Return `BackboneOutputs` (`node_feats`, `graph_feats`, optional `extras`).
4. Use `AdapterModel` to connect that backbone to one or more TriForces heads.
5. Use `collate` to build the graph/tensors your backbone expects; this is the right place to adapt graph creation for your model.

Example backbone adapter:

```python
import torch
import torch.nn as nn
from triforces.models.outputs import BackboneOutputs


class MyBackbone(nn.Module):
    def __init__(self, wrapped_model: nn.Module, output_dim: int):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.output_dim = output_dim  # lets heads infer their input_dim

    def forward(self, batch, training: bool = False, transform=None):
        # You can adapt/augment batch here before calling your model.
        node_feats = self.wrapped_model(batch)  # [num_nodes, output_dim]
        batch_idx = batch.batch
        num_graphs = int(batch.num_graphs)
        graph_feats = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
        graph_feats.index_add_(0, batch_idx, node_feats)
        counts = torch.bincount(batch_idx, minlength=num_graphs).clamp_min(1)
        graph_feats = graph_feats / counts.to(graph_feats.dtype).unsqueeze(1)
        return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats, extras={})
```

Hydra wiring example:

```yaml
model:
  _target_: triforces.models.adapter_model.AdapterModel
  backbone:
    _target_: mypkg.models.MyBackbone
    wrapped_model:
      _target_: mypkg.models.MyExistingModel
    output_dim: 256
  heads:
    proj:
      _target_: triforces.models.heads.DirectSupervisedHead
      _partial_: true
      predict_energy: true
      predict_forces: true
```

`collate` constructs graph inputs from raw structures. You can replace/customize
its target/config so your backbone receives the exact fields it needs
(`edge_index`, neighbor lists, edge features, etc.).

## Citation

If you use this repository, please cite our paper: [TriForces: Augmenting Atomistic GNNs for Transferable Representations](https://arxiv.org/abs/2605.20581).

BibTeX:

```bibtex
@article{ramlaoui2026triforces,
  title   = {TriForces: Augmenting Atomistic GNNs for Transferable Representations},
  author  = {Ramlaoui, Ali and Duval, Alexandre and Bull, Hannah and Schmidt, Victor and Talbot, Hugues and Malliaros, Fragkiskos D. and Musielewicz, Joseph},
  journal = {arXiv preprint arXiv:2605.20581},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.20581}
}
```

Example citation:

> Ramlaoui, A., Duval, A., Bull, H., Schmidt, V., Talbot, H., Malliaros, F. D., & Musielewicz, J. (2026). *TriForces: Augmenting Atomistic GNNs for Transferable Representations*. arXiv preprint arXiv:2605.20581.
