# triforces

Utilities for contrastive / self-supervised learning on crystal graphs, including:

- Crystal augmentations (noise, strain, vacancies, supercells, group substitution)
- Contrastive datasets for multi-view sampling
- Losses: InfoNCE-style contrastive, Barlow Twins, iBOT/DINO-style components

## Install

```bash
pip install -e .
```

## Minimal training example (CIF folder)

Requires PyTorch + ASE (+ `spglib` for some augmentations):

```bash
triforces data_path=/path/to/cifs dataset=cif epochs=10 batch_size=16
# or a simplified TriForces backbone:
triforces data_path=/path/to/cifs dataset=cif backbone=triforces_esen
```

## Minimal training example (Atompack)

```bash
triforces data_path=/path/to/data.atp dataset=atompack
# or a folder of .atp files:
triforces data_path=/path/to/atp_folder dataset=atompack atompack_mmap=true
```

Defaults live in `src/triforces/configs/train_contrastive.yaml`; override any field with
Hydra-style `key=value` arguments.

## Quick import

```python
from triforces.losses import ContrastiveLoss, BarlowTwinsLoss
from triforces.augmentations import CrystalNoiseAugmentation
```

## Adapting new backbones (no duplication)

Use the preset factory:

```python
from triforces.models import create_backbone

built = create_backbone("triforces_esen", embed_dim=256)
backbone = built.model
```

Or wrap any model to match the `node_features`/`graph_features` API:

```python
from triforces.models import TriForcesAdapter

backbone = TriForcesAdapter(your_model, node_key="node_features", graph_key="graph_features")
```
