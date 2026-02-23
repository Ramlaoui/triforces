import importlib.util

import pytest
import torch
from ase import Atoms

from triforces.data.simple_graph import SimpleGraph


def test_simple_graph_transform_builds_edges():
    if importlib.util.find_spec("torch_cluster") is None:
        pytest.skip("torch-cluster is required for radius_graph in SimpleGraph.")

    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0.5, 0, 0]])
    transform = SimpleGraph(radius=1.0, max_num_neighbors=8)
    data = transform(atoms)

    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] >= 2
    assert torch.allclose(data.pos, torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))
