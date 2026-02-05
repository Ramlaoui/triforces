"""Random subgraph augmentations for crystal structures."""

import random

import numpy as np
from ase import Atoms
from ase.geometry import find_mic


class RandomSubgraphWithPBC:
    """Extract k-hop neighborhood subgraphs maintaining connectivity and PBC.

    Parameters
    ----------
    fraction_range : list[float], default=[0.2, 0.8]
        Target fraction of atoms to select ``[min, max]``.
    min_atoms : int, default=10
        Minimum number of atoms in the subgraph.
    cutoff : float, default=6.0
        Distance cutoff for neighbor determination (Angstroms).
    max_hops : int, optional
        Maximum hops from the seed. If ``None``, grows until fraction is reached.
    """

    def __init__(
        self,
        fraction_range: list[float] = [0.2, 0.8],
        min_atoms: int = 10,
        cutoff: float = 6.0,
        max_hops: int = None,
    ):
        """Initialize the subgraph sampler."""
        self.fraction_range = fraction_range
        self.min_atoms = min_atoms
        self.cutoff = cutoff
        self.max_hops = max_hops

    def __call__(self, atoms: Atoms, return_mapping: bool = False):
        """
        Extract subgraph from atoms.

        Parameters
        ----------
        atoms : Atoms
            Input structure.
        return_mapping : bool
            If True, return (sub_atoms, node_correspondence) where
            node_correspondence[i] gives the original atom index for
            subgraph atom i.

        Returns
        -------
        sub_atoms : Atoms
            Subgraph structure.
        node_correspondence : np.ndarray, optional
            Mapping from subgraph indices to original indices.
            Only returned if return_mapping=True.
        """
        if len(atoms) <= self.min_atoms:
            if return_mapping:
                return atoms.copy(), np.arange(len(atoms))
            return atoms.copy()

        # Build neighbor list with PBC
        adjacency = self._build_adjacency(atoms)

        # Select k-hop neighborhood
        mask = self._select_khop_neighborhood(adjacency, len(atoms))

        if mask.sum() < self.min_atoms:
            # Fallback: return original
            if return_mapping:
                return atoms.copy(), np.arange(len(atoms))
            return atoms.copy()

        # Get indices of selected atoms (mapping from new to old)
        node_correspondence = np.where(mask)[0]

        sub_atoms = atoms[mask]
        sub_atoms.pbc = atoms.pbc.copy()
        sub_atoms.set_cell(atoms.cell)

        # Always store node correspondence in atoms.info for downstream use
        sub_atoms.info["node_correspondence"] = node_correspondence

        # import os

        # os.makedirs("ignore/debug_subgraphs", exist_ok=True)
        # # original cif
        # atoms.write(f"ignore/debug_subgraphs/original_{len(atoms)}atoms.cif")
        # # subgraph cif
        # sub_atoms.write(
        #     f"ignore/debug_subgraphs/subgraph_{len(atoms)}atoms_{len(sub_atoms)}subatoms.cif"
        # )

        if return_mapping:
            return sub_atoms, node_correspondence
        return sub_atoms

    def _build_adjacency(self, atoms: Atoms) -> dict[int, set[int]]:
        """Build adjacency list using pymatgen (faster, pure C++)."""
        from pymatgen.io.ase import AseAtomsAdaptor

        # Convert to pymatgen Structure (minimal overhead)
        struct = AseAtomsAdaptor.get_structure(atoms)

        # Pure C++ implementation - avoids Python/numpy boundary overhead
        center_indices, point_indices, _, _ = struct.get_neighbor_list(
            r=self.cutoff, numerical_tol=1e-8, exclude_self=True
        )

        # Build adjacency list (dict of sets) - much more memory efficient
        adjacency = {i: set() for i in range(len(atoms))}
        for i, j in zip(center_indices, point_indices):
            adjacency[i].add(j)

        return adjacency

    def _select_khop_neighborhood(
        self, adjacency: dict[int, set[int]], n_atoms: int
    ) -> np.ndarray:
        """Select k-hop neighborhood via BFS."""
        fraction = np.random.uniform(*self.fraction_range)
        target_count = int(n_atoms * fraction)
        # Floor at min_atoms, but cap so we always remove at least 1 atom
        target_count = min(max(target_count, self.min_atoms), n_atoms - 1)

        # Random seed node
        seed = np.random.randint(0, n_atoms)

        # BFS to grow k-hop neighborhood
        selected = {seed}
        current_layer = {seed}
        hop = 0

        while len(selected) < target_count:
            if self.max_hops is not None and hop >= self.max_hops:
                break

            # Get neighbors of current layer
            next_layer = set()
            for node in current_layer:
                # Directly access neighbors from dict (much faster than np.where)
                neighbors = adjacency[node]
                next_layer.update(neighbors - selected)

            if not next_layer:
                break  # No more neighbors

            # Add next layer (or subset if target reached)
            remaining = target_count - len(selected)
            if len(next_layer) <= remaining:
                selected.update(next_layer)
                current_layer = next_layer
            else:
                # Randomly sample from next layer
                to_add = random.sample(list(next_layer), remaining)
                selected.update(to_add)
                break

            hop += 1

        # Create mask
        mask = np.zeros(n_atoms, dtype=bool)
        mask[list(selected)] = True
        return mask


class RandomSubgraphConnected:
    """Extract connected subgraphs from crystal structures.

    Parameters
    ----------
    fraction : float, default=0.5
        Fraction of atoms to keep.
    cutoff_factor : float, default=1.3
        Factor applied to covalent radii to form neighbor cutoffs.
    min_atoms : int, default=10
        Minimum number of atoms in the subgraph.
    maintain_stoichiometry : bool, default=False
        Whether to preserve composition ratios when sampling.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        cutoff_factor: float = 1.3,
        min_atoms: int = 10,
        maintain_stoichiometry: bool = False,
    ):
        self.fraction = fraction
        self.cutoff_factor = cutoff_factor
        self.min_atoms = min_atoms
        self.maintain_stoichiometry = maintain_stoichiometry

        # Covalent radii in Angstroms (simplified subset)
        self.covalent_radii = {
            "H": 0.31,
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
            "F": 0.57,
            "Si": 1.11,
            "P": 1.07,
            "S": 1.05,
            "Cl": 1.02,
            "Fe": 1.32,
            "Co": 1.26,
            "Ni": 1.24,
            "Cu": 1.32,
            "Zn": 1.22,
            "Ga": 1.22,
            "Ge": 1.20,
            "As": 1.19,
            "Se": 1.20,
            "Br": 1.20,
            "Al": 1.21,
            "Mg": 1.41,
            "Ca": 1.76,
            "Ti": 1.60,
            "V": 1.53,
            "Cr": 1.39,
            "Mn": 1.39,
            "Na": 1.66,
            "K": 2.03,
            "Li": 1.28,
        }

    def __call__(self, atoms: Atoms) -> Atoms:
        if len(atoms) <= self.min_atoms:
            return atoms.copy()

        # Build connectivity graph
        adjacency = self._build_adjacency_matrix(atoms)

        # Select connected component
        target_count = max(self.min_atoms, int(len(atoms) * self.fraction))
        mask = self._select_connected_component(adjacency, target_count)

        # Create subgraph
        sub_atoms = atoms[mask]

        # Maintain PBC if original structure had them
        if atoms.pbc.any():
            sub_atoms.pbc = True
            # Simple approach: keep original cell
            # More sophisticated: adjust cell to fit selected atoms
            sub_atoms.set_cell(atoms.cell)

        return sub_atoms

    def _build_adjacency_matrix(self, atoms: Atoms) -> np.ndarray:
        n_atoms = len(atoms)
        adjacency = np.zeros((n_atoms, n_atoms), dtype=bool)

        positions = atoms.positions
        symbols = atoms.get_chemical_symbols()

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Get covalent radii
                r_i = self.covalent_radii.get(symbols[i], 1.5)
                r_j = self.covalent_radii.get(symbols[j], 1.5)
                cutoff = (r_i + r_j) * self.cutoff_factor

                # Calculate distance (with PBC if applicable)
                if atoms.pbc.any() and atoms.cell is not None:
                    vector, distance = find_mic(
                        positions[j] - positions[i], atoms.cell, pbc=atoms.pbc
                    )
                    distance = np.linalg.norm(vector)
                else:
                    distance = np.linalg.norm(positions[j] - positions[i])

                if distance <= cutoff:
                    adjacency[i, j] = True
                    adjacency[j, i] = True

        return adjacency

    def _select_connected_component(
        self, adjacency: np.ndarray, target_count: int
    ) -> np.ndarray:
        n_atoms = len(adjacency)
        visited = np.zeros(n_atoms, dtype=bool)

        # Start from random atom
        start = np.random.randint(0, n_atoms)

        # BFS to grow connected component
        queue = [start]
        selected = []

        while queue and len(selected) < target_count:
            current = queue.pop(0)
            if visited[current]:
                continue

            visited[current] = True
            selected.append(current)

            # Add neighbors
            neighbors = np.where(adjacency[current] & ~visited)[0]
            queue.extend(neighbors)

        # Create mask
        mask = np.zeros(n_atoms, dtype=bool)
        mask[selected] = True

        return mask
