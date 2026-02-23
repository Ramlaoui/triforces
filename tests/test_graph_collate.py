from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data

from triforces.data.ase_contrastive import AtomsSample
from triforces.data.graph_collate import (
    graph_contrastive_collate,
    graph_supervised_collate,
    pyg_collate,
)


class DummyTransform:
    def __call__(self, atoms: Atoms, **kwargs) -> Data:
        pos = torch.as_tensor(atoms.positions, dtype=torch.float32)
        z = torch.as_tensor(atoms.numbers, dtype=torch.long)
        n = pos.size(0)
        if n <= 1:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_vec = pos.new_zeros((0, 3))
        else:
            src = []
            dst = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    src.append(i)
                    dst.append(j)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        edge_dist = edge_vec.norm(dim=-1)
        return Data(
            pos=pos,
            z=z,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_dist=edge_dist,
            **kwargs,
        )


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


def test_graph_collate_does_not_validate_edges():
    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [1.5, 0, 0]])

    samples = [AtomsSample(atoms=atoms, pair_id=0)]

    class MissingEdgeVecTransform:
        def __call__(self, atoms: Atoms, **kwargs) -> Data:
            pos = torch.as_tensor(atoms.positions, dtype=torch.float32)
            z = torch.as_tensor(atoms.numbers, dtype=torch.long)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            return Data(pos=pos, z=z, edge_index=edge_index, **kwargs)

    batch = graph_contrastive_collate(samples, transform=MissingEdgeVecTransform())
    assert hasattr(batch, "edge_index")
    assert not hasattr(batch, "edge_vec")


def test_graph_supervised_collate_does_not_require_pair_id():
    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [1.5, 0, 0]])
    samples = [SimpleNamespace(atoms=atoms)]

    batch = graph_supervised_collate(samples, transform=DummyTransform())
    assert batch.num_graphs == 1
    assert not hasattr(batch, "pair_idx1")
    assert not hasattr(batch, "pair_id")


def test_graph_collate_propagates_masking_arrays():
    atoms = Atoms(numbers=[6, 8, 14], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms.arrays["original_numbers"] = np.array([6, 8, 14], dtype=np.int64)
    atoms.arrays["atom_mask"] = np.array([True, False, True], dtype=bool)
    samples = [SimpleNamespace(atoms=atoms)]

    batch = graph_supervised_collate(samples, transform=DummyTransform())
    assert hasattr(batch, "original_numbers")
    assert batch.original_numbers.dtype == torch.long
    assert torch.equal(batch.original_numbers, torch.tensor([6, 8, 14], dtype=torch.long))
    assert hasattr(batch, "atom_mask")
    assert batch.atom_mask.dtype == torch.bool
    assert torch.equal(batch.atom_mask, torch.tensor([True, False, True], dtype=torch.bool))


def test_pyg_collate_can_disable_contrastive_mode():
    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [1.5, 0, 0]])
    samples = [SimpleNamespace(atoms=atoms)]
    batch = pyg_collate(samples, graph=DummyTransform(), contrastive=False)
    assert batch.num_graphs == 1
    assert not hasattr(batch, "pair_idx1")
