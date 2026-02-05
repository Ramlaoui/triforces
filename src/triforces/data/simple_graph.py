"""Minimal ``Atoms`` to PyG graph builder without entaloracle."""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data

from triforces.models.graph_build import radius_graph


class SimpleGraph:
    """Lightweight radius-graph builder for ASE ``Atoms``.

    Parameters
    ----------
    radius : float, default=6.0
        Radius cutoff in Angstroms.
    max_num_neighbors : int or None, default=32
        Maximum neighbors per node. ``None`` disables the limit.
    loop : bool, default=False
        Whether to include self-loops.

    Notes
    -----
    This ignores periodic images and applies a plain radius cutoff on positions.
    """

    def __init__(
        self,
        radius: float = 6.0,
        max_num_neighbors: int | None = 32,
        loop: bool = False,
    ) -> None:
        self.radius = float(radius)
        self.max_num_neighbors = (
            None if max_num_neighbors is None else int(max_num_neighbors)
        )
        self.loop = bool(loop)

    def __call__(self, atoms: Atoms, **kwargs) -> Data:
        pos = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float32)
        z = torch.as_tensor(atoms.numbers, dtype=torch.long)

        edge_index = radius_graph(
            pos=pos,
            batch=None,
            r=self.radius,
            max_num_neighbors=self.max_num_neighbors,
            loop=self.loop,
        )

        data = Data(z=z, atomic_numbers=z, pos=pos, edge_index=edge_index, **kwargs)

        cell = getattr(atoms, "cell", None)
        if cell is not None and hasattr(cell, "array"):
            cell_tensor = torch.as_tensor(
                np.asarray(cell.array), dtype=torch.float32
            ).view(1, 3, 3)
            data.cell = cell_tensor

        pbc = getattr(atoms, "pbc", None)
        if pbc is not None:
            data.pbc = torch.as_tensor(np.asarray(pbc), dtype=torch.bool)

        return data


def simple_graph(
    *,
    radius: float = 6.0,
    max_num_neighbors: int | None = 32,
    loop: bool = False,
) -> SimpleGraph:
    return SimpleGraph(radius=radius, max_num_neighbors=max_num_neighbors, loop=loop)


__all__ = ["SimpleGraph", "simple_graph"]
