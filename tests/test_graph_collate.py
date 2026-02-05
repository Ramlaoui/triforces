import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data

from triforces.data.ase_contrastive import AtomsSample
from triforces.data.graph_collate import graph_contrastive_collate
from triforces.data.simple_graph import SimpleGraph


class DummyTransform:
    def __call__(self, atoms: Atoms, **kwargs) -> Data:
        pos = torch.as_tensor(atoms.positions, dtype=torch.float32)
        z = torch.as_tensor(atoms.numbers, dtype=torch.long)
        return Data(pos=pos, z=z, **kwargs)


def test_graph_contrastive_collate_builds_batch_with_pairs_and_noise():
    atoms1 = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [1, 0, 0]])
    atoms1.info["node_correspondence"] = np.array([0, 1], dtype=np.int64)
    atoms1.arrays["noise_displacement"] = np.array(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32
    )

    atoms2 = Atoms(numbers=[1, 1], positions=[[0, 0, 1], [1, 0, 1]])
    atoms2.info["node_correspondence"] = np.array([0, 1], dtype=np.int64)

    samples = [
        AtomsSample(atoms=atoms1, pair_id=0),
        AtomsSample(atoms=atoms2, pair_id=0),
    ]

    batch = graph_contrastive_collate(samples, transform=DummyTransform())

    assert batch.pair_id.shape[0] == 2
    assert batch.pair_idx1.numel() == 1
    assert batch.pair_idx2.numel() == 1

    assert hasattr(batch, "noise_displacement")
    assert batch.noise_displacement.shape == (4, 3)
    assert hasattr(batch, "noise_mask")
    assert batch.noise_mask.shape == (4,)
    assert batch.noise_mask[:2].all()
    assert (~batch.noise_mask[2:]).all()

    assert hasattr(batch, "node_correspondence")
    assert batch.node_correspondence.shape == (4,)
    assert hasattr(batch, "node_pair_idx1")
    assert hasattr(batch, "node_pair_idx2")


def test_graph_collate_applies_graph_overrides():
    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [1.5, 0, 0]])
    atoms.info["graph_radius"] = 2.0

    samples = [AtomsSample(atoms=atoms, pair_id=0)]

    transform = SimpleGraph(radius=1.0, max_num_neighbors=8)
    batch = graph_contrastive_collate(samples, transform=transform)

    assert batch.edge_index.shape[1] == 2
