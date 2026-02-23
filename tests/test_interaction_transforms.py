import numpy as np
import pytest
import torch
from ase import Atoms


def test_orb_graph_builds_edges():
    pytest.importorskip("orb_models")
    from triforces.models.interaction.orb import OrbGraph

    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0, 0, 0], [0.7, 0, 0]],
        cell=[5, 5, 5],
        pbc=True,
    )
    graph = OrbGraph(radius=1.5, max_num_neighbors=8, device="cpu")
    data = graph(atoms)

    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] > 0
    assert hasattr(data, "atomic_numbers")
    assert data.pos.shape == (2, 3)


def test_mace_graph_builds_edges():
    pytest.importorskip("mace")
    from triforces.models.interaction.mace import MACEGraph

    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0, 0, 0], [0.74, 0, 0]],
        cell=[5, 5, 5],
        pbc=True,
    )
    atoms.arrays["Qs"] = np.zeros(len(atoms), dtype=np.float32)

    graph = MACEGraph(r_max=2.0, charges_key="Qs")
    data = graph(atoms)

    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] > 0
    assert data.natoms == len(atoms)
    assert torch.equal(data.atomic_numbers, torch.tensor([1, 1]))


def test_fairchem_graphs_build_edges():
    pytest.importorskip("fairchem")
    from triforces.models.interaction.esen import UMAGraph, esen_graph

    atoms = Atoms(
        numbers=[8, 8],
        positions=[[0, 0, 0], [1.2, 0, 0]],
        cell=[6, 6, 6],
        pbc=True,
    )

    graph = esen_graph(radius=2.5, max_num_neighbors=16, dataset="omat")
    esen_data = graph(atoms)
    assert esen_data.edge_index.shape[0] == 2
    assert esen_data.edge_index.shape[1] > 0
    assert esen_data.dataset == "omat"

    uma_graph = UMAGraph(radius=2.5, max_num_neighbors=16, dataset="omat")
    uma_data = uma_graph(atoms)
    assert uma_data.edge_index.shape[0] == 2
    assert uma_data.edge_index.shape[1] > 0
    assert uma_data.dataset == "omat"
