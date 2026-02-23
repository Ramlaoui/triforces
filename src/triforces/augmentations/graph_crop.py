"""Fast graph-native spatial cropping for contrastive learning.

This module provides optimized spatial cropping that works directly on
PyTorch Geometric graphs without ASE conversion, making it ideal for
batched contrastive learning.
"""

import random
from typing import Union

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph


class SpatialBallCropASE:
    """
    ASE-compatible spatial ball crop for use in augmentation pipelines.

    This is a lightweight wrapper that works on ASE Atoms objects,
    making it compatible with existing augmentation configs that expect
    Atoms → Atoms transformations.

    For graph-level cropping (faster), use SpatialBallCrop instead.

    Parameters
    ----------
    radius_range : tuple of float, default=(5.0, 10.0)
        Range of ball radii (Angstroms) to sample from
    min_atoms : int, default=5
        Minimum atoms in crop
    max_retries : int, default=10
        Maximum retries if crop is too small
    center_strategy : str, default="atom"
        How to pick center: "atom" (random atom) or "uniform" (random point)
    handle_pbc : bool, default=True
        Whether to handle periodic boundaries
    remove_pbc : bool, default=True
        Whether to remove PBC from output (treats as fragment)
    """

    def __init__(
        self,
        radius_range: tuple[float, float] = (5.0, 10.0),
        min_atoms: int = 5,
        max_retries: int = 10,
        center_strategy: str = "atom",
        handle_pbc: bool = True,
        remove_pbc: bool = True,
    ):
        self.radius_range = radius_range
        self.min_atoms = min_atoms
        self.max_retries = max_retries
        self.center_strategy = center_strategy
        self.handle_pbc = handle_pbc
        self.remove_pbc = remove_pbc

    def __call__(
        self, atoms: Atoms, return_mapping: bool = False
    ) -> Union[Atoms, tuple[Atoms, np.ndarray]]:
        """
        Extract ball crop from atoms.

        Parameters
        ----------
        atoms : Atoms
            Input structure
        return_mapping : bool
            If True, return (cropped_atoms, node_indices)

        Returns
        -------
        Atoms or (Atoms, ndarray)
            Cropped structure, optionally with node indices
        """
        n_atoms = len(atoms)

        # Skip if too small
        if n_atoms <= self.min_atoms:
            if return_mapping:
                return atoms.copy(), np.arange(n_atoms)
            return atoms.copy()

        # Sample ball radius
        radius = random.uniform(*self.radius_range)

        # Try to find valid crop
        for attempt in range(self.max_retries):
            # Pick center
            center = self._pick_center(atoms)

            # Find atoms within radius
            if self.handle_pbc and atoms.pbc.any() and atoms.cell is not None:
                subset = self._find_atoms_in_ball_pbc(atoms, center, radius)
            else:
                subset = self._find_atoms_in_ball(atoms, center, radius)

            # Check minimum size
            if len(subset) >= self.min_atoms:
                break
        else:
            # Fallback: return random subset
            subset = np.random.permutation(n_atoms)[: self.min_atoms]

        # Extract subgraph atoms
        cropped_atoms = atoms[subset]

        # Remove PBC if requested
        if self.remove_pbc:
            cropped_atoms.set_pbc(False)
            # Create a large bounding box cell instead of None
            # (pymatgen needs a valid cell matrix)
            positions = cropped_atoms.positions
            mins = positions.min(axis=0)
            maxs = positions.max(axis=0)
            # Add padding to avoid atoms at cell boundaries
            padding = 10.0  # Angstroms
            box_size = maxs - mins + 2 * padding
            # Center the atoms in the box
            cropped_atoms.positions = positions - mins + padding
            # Set orthogonal box cell
            cropped_atoms.set_cell([box_size[0], box_size[1], box_size[2]])
        else:
            # Preserve PBC and cell
            cropped_atoms.set_pbc(atoms.pbc)
            cropped_atoms.set_cell(atoms.cell)

        # Store node correspondence in atoms.info for downstream use
        cropped_atoms.info["node_correspondence"] = subset

        if return_mapping:
            return cropped_atoms, subset
        return cropped_atoms

    def _pick_center(self, atoms: Atoms) -> np.ndarray:
        """Pick center point for ball crop."""
        if self.center_strategy == "atom":
            # Random atom position
            idx = np.random.randint(0, len(atoms))
            return atoms.positions[idx]

        elif self.center_strategy == "uniform":
            # Uniform random point in cell
            if atoms.pbc.any() and atoms.cell is not None:
                # Random point in unit cell
                fracs = np.random.rand(3)
                return fracs @ atoms.cell
            else:
                # Random point in bounding box
                mins = atoms.positions.min(axis=0)
                maxs = atoms.positions.max(axis=0)
                return mins + np.random.rand(3) * (maxs - mins)

        else:
            raise ValueError(f"Unknown center_strategy: {self.center_strategy}")

    def _find_atoms_in_ball(
        self, atoms: Atoms, center: np.ndarray, radius: float
    ) -> np.ndarray:
        """Find atoms within radius of center (no PBC)."""
        distances = np.linalg.norm(atoms.positions - center, axis=1)
        return np.where(distances <= radius)[0]

    def _find_atoms_in_ball_pbc(
        self, atoms: Atoms, center: np.ndarray, radius: float
    ) -> np.ndarray:
        """Find atoms within radius of center (with PBC using minimum image)."""
        from ase.geometry import find_mic

        # Compute displacement vectors
        deltas = atoms.positions - center

        # Apply minimum image convention
        vectors, distances = find_mic(deltas, atoms.cell, pbc=atoms.pbc)

        return np.where(distances <= radius)[0]


class FastSpatialCrop:
    """
    Fast spatial cropping directly on PyG graphs (no ASE conversion).

    This is optimized for contrastive learning pipelines where:
    1. You already have a graph with computed edges
    2. You want multiple random crops per sample (batched)
    3. You don't want to recompute neighbor lists in ASE

    Key optimizations:
    - Works directly on edge_index (no neighbor list recomputation)
    - Pure tensor operations (GPU-compatible)
    - No ASE conversion overhead
    - Supports both k-hop (connectivity-aware) and spatial (distance-aware) crops

    Parameters
    ----------
    fraction_range : tuple of float, default=(0.2, 0.8)
        Range of atoms to keep [min, max]
    min_atoms : int, default=10
        Minimum atoms in crop
    mode : str, default="khop"
        Cropping mode:
        - "khop": k-hop neighborhood (connectivity-based, fastest)
        - "spatial": spatial distance cutoff (requires pos)
        - "random": random selection (baseline)
    max_hops : int, optional
        Maximum hops for k-hop mode. If None, grows until fraction reached
    spatial_cutoff : float, default=6.0
        Distance cutoff for spatial mode (Angstroms)
    remove_pbc : bool, default=True
        Whether to remove periodic boundary after cropping
        (treats crop as molecular fragment)
    """

    def __init__(
        self,
        fraction_range: tuple[float, float] = (0.2, 0.8),
        min_atoms: int = 10,
        mode: str = "khop",
        max_hops: int | None = None,
        spatial_cutoff: float = 6.0,
        remove_pbc: bool = True,
    ):
        self.fraction_range = fraction_range
        self.min_atoms = min_atoms
        self.mode = mode
        self.max_hops = max_hops
        self.spatial_cutoff = spatial_cutoff
        self.remove_pbc = remove_pbc

    def __call__(
        self, data: Data, return_subset: bool = False
    ) -> Union[Data, tuple[Data, torch.Tensor]]:
        """
        Extract spatial crop from graph.

        Parameters
        ----------
        data : Data
            Input graph (must have edge_index)
        return_subset : bool
            If True, return (cropped_data, node_indices)

        Returns
        -------
        Data or (Data, Tensor)
            Cropped graph, optionally with node indices
        """
        n_nodes = data.num_nodes

        # Skip if too small
        if n_nodes <= self.min_atoms:
            if return_subset:
                return data, torch.arange(n_nodes, device=data.edge_index.device)
            return data

        # Sample target size
        fraction = random.uniform(*self.fraction_range)
        target_size = max(self.min_atoms, int(n_nodes * fraction))
        target_size = min(target_size, n_nodes - 1)  # Always remove at least 1

        # Select subset based on mode
        if self.mode == "khop":
            subset = self._khop_subset(data, target_size)
        elif self.mode == "spatial":
            subset = self._spatial_subset(data, target_size)
        elif self.mode == "random":
            subset = torch.randperm(n_nodes, device=data.edge_index.device)[
                :target_size
            ]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Extract subgraph
        cropped_data = self._extract_subgraph(data, subset)

        # Remove PBC if requested (treat as molecular fragment)
        if self.remove_pbc and hasattr(cropped_data, "cell"):
            cropped_data.cell = None
            if hasattr(cropped_data, "pbc"):
                cropped_data.pbc = None

        if return_subset:
            return cropped_data, subset
        return cropped_data

    def _khop_subset(self, data: Data, target_size: int) -> torch.Tensor:
        """Select k-hop neighborhood subset."""
        n_nodes = data.num_nodes
        device = data.edge_index.device

        # Random seed node
        seed = torch.randint(0, n_nodes, (1,), device=device).item()

        if self.max_hops is not None:
            # Fixed k-hop
            subset, _, _, _ = k_hop_subgraph(
                node_idx=seed,
                num_hops=self.max_hops,
                edge_index=data.edge_index,
                num_nodes=n_nodes,
            )
        else:
            # Grow until target size
            subset = torch.tensor([seed], dtype=torch.long, device=device)

            for k in range(1, 20):  # Upper limit
                subset_k, _, _, _ = k_hop_subgraph(
                    node_idx=seed,
                    num_hops=k,
                    edge_index=data.edge_index,
                    num_nodes=n_nodes,
                )

                if len(subset_k) >= target_size:
                    subset = subset_k
                    break
                subset = subset_k

        # Randomly sample if too large
        if len(subset) > target_size:
            perm = torch.randperm(len(subset), device=device)[:target_size]
            subset = subset[perm]

        # Ensure minimum size
        if len(subset) < self.min_atoms:
            # Fallback: random nodes
            subset = torch.randperm(n_nodes, device=device)[:target_size]

        return subset

    def _spatial_subset(self, data: Data, target_size: int) -> torch.Tensor:
        """Select spatial neighborhood subset (BFS with distance awareness)."""
        if not hasattr(data, "pos"):
            # Fallback to k-hop if no positions
            return self._khop_subset(data, target_size)

        n_nodes = data.num_nodes
        device = data.edge_index.device

        # Random seed
        seed = torch.randint(0, n_nodes, (1,), device=device).item()

        # BFS with distance filtering
        selected = {seed}
        current_layer = {seed}

        # Build adjacency list for faster neighbor lookup
        adj = {}
        for i in range(n_nodes):
            adj[i] = set()
        for i, j in data.edge_index.t().tolist():
            adj[i].add(j)

        while len(selected) < target_size:
            next_layer = set()

            for node in current_layer:
                # Get neighbors within spatial cutoff
                neighbors = adj[node]
                for neighbor in neighbors:
                    if neighbor not in selected:
                        # Check distance
                        dist = torch.norm(data.pos[neighbor] - data.pos[node])
                        if dist <= self.spatial_cutoff:
                            next_layer.add(neighbor)

            if not next_layer:
                break

            # Add next layer
            remaining = target_size - len(selected)
            if len(next_layer) <= remaining:
                selected.update(next_layer)
                current_layer = next_layer
            else:
                # Random sample
                to_add = random.sample(list(next_layer), remaining)
                selected.update(to_add)
                break

        return torch.tensor(sorted(selected), dtype=torch.long, device=device)

    def _extract_subgraph(self, data: Data, subset: torch.Tensor) -> Data:
        """
        Extract subgraph with all relevant attributes.

        Uses PyG's subgraph utility for efficient extraction.
        """
        # Get subgraph edges
        edge_index, edge_attr = subgraph(
            subset=subset,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

        # Create new data object
        new_data = Data(edge_index=edge_index)

        # Copy node-level attributes
        for key in data.keys():
            if key in ["edge_index", "edge_attr"]:
                continue

            value = data[key]

            # Check if node-level attribute
            if isinstance(value, torch.Tensor) and value.size(0) == data.num_nodes:
                new_data[key] = value[subset]
            # Check if edge-level attribute
            elif key == "edge_attr" and edge_attr is not None:
                new_data[key] = edge_attr
            # Graph-level attributes
            elif not isinstance(value, torch.Tensor) or value.numel() == 1:
                new_data[key] = value
            # Special case: cell, pbc (keep as-is for now)
            elif key in ["cell", "pbc"]:
                new_data[key] = value

        # Store node correspondence for contrastive learning
        new_data.node_correspondence = subset

        return new_data


class BatchedSpatialCrop:
    """
    Batched spatial crop for efficient multi-view generation.

    Generates multiple spatial crops from a single graph in parallel,
    optimized for contrastive learning with high batch sizes.

    Parameters
    ----------
    n_crops : int, default=4
        Number of crops to generate per sample
    **crop_kwargs
        Arguments passed to FastSpatialCrop

    Examples
    --------
    >>> crop_fn = BatchedSpatialCrop(n_crops=4, fraction_range=(0.2, 0.5))
    >>> crops = crop_fn(data)  # Returns list of 4 cropped graphs
    """

    def __init__(self, n_crops: int = 4, **crop_kwargs):
        self.n_crops = n_crops
        self.crop_fn = FastSpatialCrop(**crop_kwargs)

    def __call__(self, data: Data) -> list[Data]:
        """Generate multiple crops from input graph."""
        return [self.crop_fn(data) for _ in range(self.n_crops)]


class SpatialBallCrop:
    """
    Pure spatial ball crop - picks a random center and takes all atoms within radius.

    This is the fastest possible spatial crop because:
    1. No graph dependency - works directly on positions
    2. No BFS/traversal needed - simple distance calculation
    3. Pure tensor operations - fully GPU compatible
    4. O(n) complexity for distance checks

    Perfect for contrastive learning when you want:
    - Ultra-fast random crops
    - Guaranteed spherical locality
    - No connectivity constraints

    Parameters
    ----------
    radius_range : tuple of float, default=(5.0, 10.0)
        Range of ball radii (Angstroms) to sample from
    min_atoms : int, default=10
        Minimum atoms in crop (will retry if too small)
    max_retries : int, default=10
        Maximum retries if crop is too small
    center_strategy : str, default="atom"
        How to pick center:
        - "atom": Pick random atom position
        - "uniform": Uniform random point in cell
        - "weighted": Weighted by local density
    handle_pbc : bool, default=True
        Whether to handle periodic boundaries properly
    remove_pbc : bool, default=True
        Whether to remove PBC from cropped structure
    """

    def __init__(
        self,
        radius_range: tuple[float, float] = (5.0, 10.0),
        min_atoms: int = 10,
        max_retries: int = 10,
        center_strategy: str = "atom",
        handle_pbc: bool = True,
        remove_pbc: bool = True,
    ):
        self.radius_range = radius_range
        self.min_atoms = min_atoms
        self.max_retries = max_retries
        self.center_strategy = center_strategy
        self.handle_pbc = handle_pbc
        self.remove_pbc = remove_pbc

    def __call__(
        self, data: Data, return_subset: bool = False
    ) -> Union[Data, tuple[Data, torch.Tensor]]:
        """
        Extract ball crop from structure.

        Parameters
        ----------
        data : Data
            Input graph (must have pos)
        return_subset : bool
            If True, return (cropped_data, node_indices)

        Returns
        -------
        Data or (Data, Tensor)
            Cropped graph, optionally with node indices
        """
        if not hasattr(data, "pos"):
            raise ValueError("SpatialBallCrop requires position data (data.pos)")

        n_nodes = data.num_nodes
        device = data.pos.device

        # Skip if too small
        if n_nodes <= self.min_atoms:
            if return_subset:
                return data, torch.arange(n_nodes, device=device)
            return data

        # Sample ball radius
        radius = random.uniform(*self.radius_range)

        # Try to find valid crop
        for attempt in range(self.max_retries):
            # Pick center
            center = self._pick_center(data)

            # Find atoms within radius
            if self.handle_pbc and hasattr(data, "cell") and data.cell is not None:
                subset = self._find_atoms_in_ball_pbc(
                    data.pos, center, radius, data.cell
                )
            else:
                subset = self._find_atoms_in_ball(data.pos, center, radius)

            # Check minimum size
            if len(subset) >= self.min_atoms:
                break
        else:
            # Fallback: return random subset
            subset = torch.randperm(n_nodes, device=device)[: self.min_atoms]

        # Extract subgraph
        cropped_data = self._extract_subgraph(data, subset)

        # Store crop metadata
        cropped_data.crop_center = center
        cropped_data.crop_radius = radius

        # Remove PBC if requested
        if self.remove_pbc and hasattr(cropped_data, "cell"):
            cropped_data.cell = None
            if hasattr(cropped_data, "pbc"):
                cropped_data.pbc = None

        if return_subset:
            return cropped_data, subset
        return cropped_data

    def _pick_center(self, data: Data) -> torch.Tensor:
        """Pick center point for ball crop."""
        if self.center_strategy == "atom":
            # Random atom position
            idx = torch.randint(0, data.num_nodes, (1,), device=data.pos.device).item()
            return data.pos[idx]

        elif self.center_strategy == "uniform":
            # Uniform random point in cell
            if hasattr(data, "cell") and data.cell is not None:
                # Random point in unit cell
                fracs = torch.rand(3, device=data.pos.device)
                return fracs @ data.cell
            else:
                # Random point in bounding box
                mins = data.pos.min(dim=0)[0]
                maxs = data.pos.max(dim=0)[0]
                return mins + torch.rand(3, device=data.pos.device) * (maxs - mins)

        elif self.center_strategy == "weighted":
            # Weighted by local density (approximate with inverse distances)
            # This biases toward denser regions
            sample_size = min(100, data.num_nodes)
            sample_idx = torch.randperm(data.num_nodes, device=data.pos.device)[
                :sample_size
            ]
            sample_pos = data.pos[sample_idx]

            # Compute pairwise distances
            dists = torch.cdist(sample_pos, data.pos)
            densities = 1.0 / (dists.mean(dim=0) + 1e-6)

            # Sample weighted by density
            probs = densities / densities.sum()
            idx = torch.multinomial(probs, 1).item()
            return data.pos[idx]

        else:
            raise ValueError(f"Unknown center_strategy: {self.center_strategy}")

    def _find_atoms_in_ball(
        self, pos: torch.Tensor, center: torch.Tensor, radius: float
    ) -> torch.Tensor:
        """Find atoms within radius of center (no PBC)."""
        distances = torch.norm(pos - center, dim=1)
        mask = distances <= radius
        return torch.where(mask)[0]

    def _find_atoms_in_ball_pbc(
        self, pos: torch.Tensor, center: torch.Tensor, radius: float, cell: torch.Tensor
    ) -> torch.Tensor:
        """Find atoms within radius of center (with PBC using minimum image convention)."""
        # Compute displacement vectors
        delta = pos - center  # (n_atoms, 3)

        # Apply minimum image convention
        # Convert to fractional coordinates
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)  # (1, 3, 3)

        # Inverse cell matrix
        inv_cell = torch.linalg.inv(cell.squeeze(0))  # (3, 3)

        # Convert to fractional
        frac_delta = delta @ inv_cell.T  # (n_atoms, 3)

        # Wrap to [-0.5, 0.5]
        frac_delta = frac_delta - torch.round(frac_delta)

        # Convert back to Cartesian
        cart_delta = frac_delta @ cell.squeeze(0)  # (n_atoms, 3)

        # Compute distances
        distances = torch.norm(cart_delta, dim=1)
        mask = distances <= radius

        return torch.where(mask)[0]

    def _extract_subgraph(self, data: Data, subset: torch.Tensor) -> Data:
        """Extract subgraph with all relevant attributes."""
        from torch_geometric.utils import subgraph

        # Get subgraph edges
        edge_index, edge_attr = subgraph(
            subset=subset,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

        # Create new data object
        new_data = Data(edge_index=edge_index)

        # Copy node-level attributes
        for key in data.keys():
            if key in ["edge_index", "edge_attr"]:
                continue

            value = data[key]

            # Node-level attributes
            if isinstance(value, torch.Tensor) and value.size(0) == data.num_nodes:
                new_data[key] = value[subset]
            # Edge-level attributes
            elif key == "edge_attr" and edge_attr is not None:
                new_data[key] = edge_attr
            # Graph-level attributes
            elif not isinstance(value, torch.Tensor) or value.numel() == 1:
                new_data[key] = value
            elif key in ["cell", "pbc"]:
                new_data[key] = value

        # Store node correspondence
        new_data.node_correspondence = subset

        return new_data


class AdaptiveSpatialCrop(FastSpatialCrop):
    """
    Adaptive spatial crop that adjusts crop size based on graph properties.

    Automatically scales crop size based on:
    - Graph size (smaller crops for larger graphs)
    - Connectivity (more connected = smaller crops needed)
    - Density (denser = smaller crops)

    This ensures crops are semantically meaningful across diverse structures.

    Parameters
    ----------
    base_fraction_range : tuple of float, default=(0.3, 0.7)
        Base fraction range before adaptive scaling
    scale_with_size : bool, default=True
        Scale crop size inversely with graph size
    scale_with_density : bool, default=True
        Scale crop size inversely with edge density
    **crop_kwargs
        Arguments passed to FastSpatialCrop
    """

    def __init__(
        self,
        base_fraction_range: tuple[float, float] = (0.3, 0.7),
        scale_with_size: bool = True,
        scale_with_density: bool = True,
        **crop_kwargs,
    ):
        self.base_fraction_range = base_fraction_range
        self.scale_with_size = scale_with_size
        self.scale_with_density = scale_with_density

        # Initialize parent with dummy range (will be overridden)
        super().__init__(fraction_range=base_fraction_range, **crop_kwargs)

    def __call__(self, data: Data, return_subset: bool = False):
        """Extract adaptive crop."""
        # Compute adaptive fraction range
        self.fraction_range = self._compute_adaptive_range(data)

        # Call parent
        return super().__call__(data, return_subset)

    def _compute_adaptive_range(self, data: Data) -> tuple[float, float]:
        """Compute adaptive fraction range based on graph properties."""
        base_min, base_max = self.base_fraction_range

        # Start with base range
        scale_factor = 1.0

        # Scale with size (smaller crops for larger graphs)
        if self.scale_with_size:
            n_nodes = data.num_nodes
            # Logarithmic scaling: larger graphs get relatively smaller crops
            size_scale = max(
                0.5, 1.0 - 0.1 * torch.log10(torch.tensor(n_nodes / 50.0)).item()
            )
            scale_factor *= size_scale

        # Scale with density (smaller crops for denser graphs)
        if self.scale_with_density:
            n_nodes = data.num_nodes
            n_edges = data.edge_index.size(1)
            density = n_edges / (n_nodes * n_nodes)
            # Denser graphs need smaller crops to maintain diversity
            density_scale = max(0.6, 1.0 - 10.0 * density)
            scale_factor *= density_scale

        # Apply scaling
        adapted_min = max(0.05, base_min * scale_factor)
        adapted_max = max(adapted_min + 0.1, base_max * scale_factor)
        adapted_max = min(0.95, adapted_max)  # Cap at 95%

        return (adapted_min, adapted_max)


class RandomResizeCropASE:
    """
    Random resize crop for crystal structures - analogous to RandomResizedCrop in vision.

    Pipeline:
    1. **Crop**: Extract a spatial ball (random center, random radius)
    2. **Resize**: Scale the cell to the original volume, preserving angles

    This effectively "zooms in" on a local region - fewer atoms spread out to fill
    the original cell volume, changing the density while preserving the lattice shape.

    Parameters
    ----------
    radius_range : tuple of float, default=(4.0, 8.0)
        Range of ball radii (Angstroms) to sample from for cropping
    min_atoms : int, default=5
        Minimum atoms in crop
    max_retries : int, default=10
        Maximum retries if crop is too small
    center_strategy : str, default="atom"
        How to pick center: "atom" (random atom) or "uniform" (random point)
    preserve_volume : bool, default=True
        If True, scale cell to original volume after crop (the "resize" step)
    handle_pbc : bool, default=True
        Whether to handle periodic boundaries during crop

    Examples
    --------
    >>> augment = RandomResizeCropASE(radius_range=(4.0, 8.0))
    >>> cropped = augment(atoms)  # Cropped + resized to original volume
    """

    def __init__(
        self,
        radius_range: tuple[float, float] = (4.0, 8.0),
        min_atoms: int = 5,
        max_retries: int = 10,
        center_strategy: str = "atom",
        preserve_volume: bool = True,
        handle_pbc: bool = True,
    ):
        self.radius_range = radius_range
        self.min_atoms = min_atoms
        self.max_retries = max_retries
        self.center_strategy = center_strategy
        self.preserve_volume = preserve_volume
        self.handle_pbc = handle_pbc

    def __call__(
        self,
        atoms: Atoms,
        return_mapping: bool = False,
    ) -> Union[Atoms, tuple[Atoms, np.ndarray]]:
        """
        Apply random resize crop to structure.

        Parameters
        ----------
        atoms : Atoms
            Input structure
        return_mapping : bool
            If True, return (cropped_atoms, node_indices)

        Returns
        -------
        Atoms or (Atoms, ndarray)
            Cropped and resized structure, optionally with node indices
        """

        n_atoms = len(atoms)

        # Skip if too small
        if n_atoms <= self.min_atoms:
            if return_mapping:
                return atoms.copy(), np.arange(n_atoms)
            return atoms.copy()

        # Store original volume for resize step
        original_volume = atoms.get_volume()

        # Sample ball radius
        radius = random.uniform(*self.radius_range)

        # Try to find valid crop
        for attempt in range(self.max_retries):
            # Pick center
            center = self._pick_center(atoms)

            # Find atoms within radius
            if self.handle_pbc and atoms.pbc.any() and atoms.cell is not None:
                subset = self._find_atoms_in_ball_pbc(atoms, center, radius)
            else:
                subset = self._find_atoms_in_ball(atoms, center, radius)

            # Check minimum size
            if len(subset) >= self.min_atoms:
                break
        else:
            # Fallback: return random subset
            subset = np.random.permutation(n_atoms)[: self.min_atoms]

        # Extract cropped atoms - keep original cell and PBC
        cropped_atoms = atoms[subset]
        cropped_atoms.set_pbc(atoms.pbc)
        cropped_atoms.set_cell(atoms.cell)

        # Store node correspondence
        cropped_atoms.info["node_correspondence"] = subset

        # Resize: scale cell to original volume (preserving angles)
        if self.preserve_volume:
            cropped_atoms = self._scale_to_volume(cropped_atoms, original_volume)

        # import os
        # import uuid

        # uid = str(uuid.uuid4())[:8]
        # os.makedirs("ignore/debug_subgraphs", exist_ok=True)
        # # original cif
        # atoms.write(f"ignore/debug_subgraphs/original.cif")
        # # subgraph cif
        # cropped_atoms.write(f"ignore/debug_subgraphs/cropped.cif")

        if return_mapping:
            return cropped_atoms, subset
        return cropped_atoms

    def _pick_center(self, atoms: Atoms) -> np.ndarray:
        """Pick center point for ball crop."""
        if self.center_strategy == "atom":
            idx = np.random.randint(0, len(atoms))
            return atoms.positions[idx]
        elif self.center_strategy == "uniform":
            if atoms.pbc.any() and atoms.cell is not None:
                fracs = np.random.rand(3)
                return fracs @ atoms.cell
            else:
                mins = atoms.positions.min(axis=0)
                maxs = atoms.positions.max(axis=0)
                return mins + np.random.rand(3) * (maxs - mins)
        else:
            raise ValueError(f"Unknown center_strategy: {self.center_strategy}")

    def _find_atoms_in_ball(
        self, atoms: Atoms, center: np.ndarray, radius: float
    ) -> np.ndarray:
        """Find atoms within radius of center (no PBC)."""
        distances = np.linalg.norm(atoms.positions - center, axis=1)
        return np.where(distances <= radius)[0]

    def _find_atoms_in_ball_pbc(
        self, atoms: Atoms, center: np.ndarray, radius: float
    ) -> np.ndarray:
        """Find atoms within radius of center (with PBC using minimum image)."""
        from ase.geometry import find_mic

        deltas = atoms.positions - center
        vectors, distances = find_mic(deltas, atoms.cell, pbc=atoms.pbc)
        return np.where(distances <= radius)[0]

    def _scale_to_volume(self, atoms: Atoms, target_volume: float) -> Atoms:
        """
        Scale cell to target volume while preserving angles.

        Uses pymatgen's scale_lattice which uniformly scales all lattice
        vectors to achieve the target volume.
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        # Convert to pymatgen Structure
        struct = AseAtomsAdaptor.get_structure(atoms)

        # Scale lattice to target volume (preserves angles)
        struct.scale_lattice(target_volume)

        # Convert back to ASE
        scaled_atoms = AseAtomsAdaptor.get_atoms(struct)

        # Preserve info dict
        scaled_atoms.info = atoms.info.copy()

        return scaled_atoms
