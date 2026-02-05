"""Crystal augmentations for ASE ``Atoms`` objects."""

import random
from typing import Optional

import numpy as np
import spglib
from ase import Atoms
from ase.build import make_supercell


class CrystalNoiseAugmentation:
    """Add Gaussian noise to crystal positions.

    Parameters
    ----------
    noise_scale : float or list[float], default=0.01
        Noise scale in Angstroms. If a list/tuple of two values is provided, a
        uniform sample in ``[min, max]`` is used per call.
    label_preserving : bool, default=False
        Whether this augmentation preserves labels.
    """

    def __init__(
        self, noise_scale: float | list[float] = 0.01, label_preserving: bool = False
    ):
        self._noise_scale = None
        self._noise_scale_min = None
        self._noise_scale_max = None
        self.noise_scale = noise_scale
        self.label_preserving = label_preserving

    @property
    def noise_scale(self):
        return self._noise_scale

    @noise_scale.setter
    def noise_scale(self, value):
        object.__setattr__(self, "_noise_scale", value)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            object.__setattr__(self, "_noise_scale_min", float(value[0]))
            object.__setattr__(self, "_noise_scale_max", float(value[1]))
        elif value is not None:
            val = float(value)
            object.__setattr__(self, "_noise_scale_min", val)
            object.__setattr__(self, "_noise_scale_max", val)
        else:
            object.__setattr__(self, "_noise_scale_min", None)
            object.__setattr__(self, "_noise_scale_max", None)

    @property
    def noise_scale_min(self) -> Optional[float]:
        return self._noise_scale_min

    @noise_scale_min.setter
    def noise_scale_min(self, value: Optional[float]):
        object.__setattr__(
            self, "_noise_scale_min", None if value is None else float(value)
        )
        self._sync_noise_scale_from_bounds()

    @property
    def noise_scale_max(self) -> Optional[float]:
        return self._noise_scale_max

    @noise_scale_max.setter
    def noise_scale_max(self, value: Optional[float]):
        object.__setattr__(
            self, "_noise_scale_max", None if value is None else float(value)
        )
        self._sync_noise_scale_from_bounds()

    def _sync_noise_scale_from_bounds(self):
        """Keep ``noise_scale`` list in sync when min/max bounds change."""
        min_scale = self._noise_scale_min
        max_scale = self._noise_scale_max

        if min_scale is None and max_scale is None:
            return

        if min_scale is None:
            min_scale = max_scale
        if max_scale is None:
            max_scale = min_scale

        if min_scale == max_scale:
            object.__setattr__(self, "_noise_scale", float(min_scale))
        else:
            low = float(min(min_scale, max_scale))
            high = float(max(min_scale, max_scale))
            object.__setattr__(self, "_noise_scale", [low, high])

    def _sample_noise_scale(self) -> float:
        """Sample the current noise scale respecting schedulable bounds.

        Returns
        -------
        float
            Sampled noise scale.
        """
        min_scale = self._noise_scale_min
        max_scale = self._noise_scale_max

        if min_scale is not None or max_scale is not None:
            if min_scale is None:
                min_scale = max_scale
            if max_scale is None:
                max_scale = min_scale

            if min_scale == max_scale:
                return float(min_scale)

            low = float(min(min_scale, max_scale))
            high = float(max(min_scale, max_scale))
            return float(np.random.uniform(low, high))

        if isinstance(self.noise_scale, (list, tuple)) and len(self.noise_scale) >= 2:
            return float(np.random.uniform(self.noise_scale[0], self.noise_scale[1]))

        return float(self.noise_scale)

    def __call__(self, atoms: Atoms) -> Atoms:
        new_atoms = atoms.copy()
        scale = self._sample_noise_scale()
        noise = np.random.randn(len(atoms), 3).astype(np.float32) * scale
        new_atoms.positions += noise
        new_atoms.arrays["noise_displacement"] = noise
        # Track the applied noise scale and preserve original species metadata
        new_atoms.info["noise_scale"] = float(scale)
        if "original_numbers" not in new_atoms.info:
            new_atoms.info["original_numbers"] = atoms.numbers.copy()
        return new_atoms


class DiffusionNoiseAugmentation:
    """Add noise to crystal positions using diffusion-style schedules.

    Parameters
    ----------
    sigma_min : float, default=0.01
        Minimum noise level in Angstroms.
    sigma_max : float, default=0.5
        Maximum noise level in Angstroms.
    schedule : str, default="edm"
        Noise schedule type: ``"uniform"``, ``"log_uniform"``, ``"edm"``, or ``"cosine"``.
    store_timestep : bool, default=True
        Whether to store the diffusion timestep in ``atoms.info``.
    label_preserving : bool, default=False
        Whether this augmentation preserves labels.

    Notes
    -----
    Supported schedules:
    - ``"uniform"``: Uniform sampling (same as ``CrystalNoiseAugmentation``).
    - ``"log_uniform"``: Log-uniform sampling.
    - ``"edm"``: ``sigma_min * (sigma_max / sigma_min) ** t``.
    - ``"cosine"``: Cosine schedule from improved DDPM.
    The timestep ``t`` is stored in ``atoms.info`` for optional conditioning.
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 0.5,
        schedule: str = "edm",
        store_timestep: bool = True,
        label_preserving: bool = False,
    ):
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self.schedule = schedule
        self.store_timestep = store_timestep
        self.label_preserving = label_preserving

    @property
    def sigma_min(self) -> float:
        return self._sigma_min

    @sigma_min.setter
    def sigma_min(self, value: float):
        self._sigma_min = float(value)

    @property
    def sigma_max(self) -> float:
        return self._sigma_max

    @sigma_max.setter
    def sigma_max(self, value: float):
        self._sigma_max = float(value)

    def _sample_timestep_and_sigma(self) -> tuple[float, float]:
        """Sample timestep and corresponding sigma for the chosen schedule.

        Returns
        -------
        tuple[float, float]
            Tuple of ``(t, sigma)``.
        """
        t = np.random.uniform(0, 1)

        if self.schedule == "uniform":
            # Linear interpolation between sigma_min and sigma_max
            sigma = self._sigma_min + t * (self._sigma_max - self._sigma_min)
        elif self.schedule == "log_uniform":
            # Log-uniform: better coverage of noise scales
            log_sigma = np.log(self._sigma_min) + t * (
                np.log(self._sigma_max) - np.log(self._sigma_min)
            )
            sigma = np.exp(log_sigma)
        elif self.schedule == "edm":
            # EDM schedule: sigma = sigma_min * (sigma_max/sigma_min)^t
            sigma = self._sigma_min * (self._sigma_max / self._sigma_min) ** t
        elif self.schedule == "cosine":
            # Cosine schedule (from improved DDPM)
            # alpha_bar = cos((t + 0.008) / 1.008 * pi/2)^2
            # sigma = sqrt((1 - alpha_bar) / alpha_bar) scaled to [sigma_min, sigma_max]
            s = 0.008
            alpha_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            # Map to sigma range using SNR relationship
            snr = alpha_bar / (1 - alpha_bar + 1e-8)
            sigma_normalized = 1.0 / np.sqrt(snr + 1e-8)
            # Scale to desired range
            sigma = self._sigma_min + sigma_normalized * (
                self._sigma_max - self._sigma_min
            )
            sigma = np.clip(sigma, self._sigma_min, self._sigma_max)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return float(t), float(sigma)

    def __call__(self, atoms: Atoms) -> Atoms:
        new_atoms = atoms.copy()
        t, sigma = self._sample_timestep_and_sigma()

        noise = np.random.randn(len(atoms), 3).astype(np.float32) * sigma
        new_atoms.positions += noise

        # Store noise displacement as per-atom array
        # This is the noise in the CURRENT frame (before any subsequent rotation)
        # If rotation is applied later, both positions and this array rotate together
        new_atoms.arrays["noise_displacement"] = noise

        # Track diffusion-specific metadata
        new_atoms.info["noise_scale"] = sigma  # For compatibility with existing code
        new_atoms.info["diffusion_sigma"] = sigma
        new_atoms.info["diffusion_schedule"] = self.schedule

        # Optionally store timestep for model conditioning
        if self.store_timestep:
            new_atoms.info["diffusion_timestep"] = t

        if "original_numbers" not in new_atoms.info:
            new_atoms.info["original_numbers"] = atoms.numbers.copy()

        return new_atoms


class CrystalStrainAugmentation:
    """Apply strain deformation to a crystal structure.

    Parameters
    ----------
    max_strain : float or list[float], default=0.05
        Maximum strain magnitude. If a list/tuple of two values is provided, a
        uniform sample in ``[min, max]`` is used per call.
    label_preserving : bool, default=False
        Whether this augmentation preserves labels.

    Notes
    -----
    Tracks the cell displacement for denoising via ``cell_noise_displacement`` and
    stores ``strain_scale`` in ``atoms.info``.
    """

    def __init__(
        self, max_strain: float | list[float] = 0.05, label_preserving: bool = False
    ):
        self.max_strain = max_strain
        self.label_preserving = label_preserving

    def __call__(self, atoms: Atoms) -> Atoms:
        new_atoms = atoms.copy()

        if atoms.cell is not None and atoms.cell.any():
            # Sample max_strain if it's a list [min, max]
            if isinstance(self.max_strain, list):
                strain_scale = np.random.uniform(self.max_strain[0], self.max_strain[1])
            else:
                strain_scale = self.max_strain

            strain = np.random.randn(3, 3) * strain_scale
            strain = (strain + strain.T) / 2
            strain_matrix = np.eye(3) + strain

            original_cell = atoms.cell.array.copy()
            new_cell = original_cell @ strain_matrix
            new_atoms.set_cell(new_cell, scale_atoms=True)

            # Track cell displacement for denoising (new_cell - original_cell)
            cell_displacement = new_cell - original_cell

            # Accumulate if previous cell displacement exists
            existing_displacement = new_atoms.info.get("cell_noise_displacement")
            if existing_displacement is not None:
                cell_displacement = cell_displacement + np.asarray(
                    existing_displacement
                )

            new_atoms.info["cell_noise_displacement"] = cell_displacement
            new_atoms.info["strain_scale"] = float(strain_scale)
            # Store original cell if not already stored
            if "original_cell" not in new_atoms.info:
                new_atoms.info["original_cell"] = original_cell

        return new_atoms


class RandomUnitCellPerturbation:
    """Randomly perturb unit cell without scaling atoms.

    Parameters
    ----------
    max_strain : float or list[float], default=0.05
        Maximum strain magnitude. If a list/tuple of two values is provided, a
        uniform sample in ``[min, max]`` is used per call.
    label_preserving : bool, default=False
        Whether this augmentation preserves labels.

    Notes
    -----
    This only changes the unit cell, not atomic positions. It still affects PBC
    neighbor distances, so labels are not strictly preserved. The augmentation
    tracks ``cell_noise_displacement`` and ``strain_scale`` in ``atoms.info``.
    """

    def __init__(
        self, max_strain: float | list[float] = 0.05, label_preserving: bool = False
    ):
        self.max_strain = max_strain
        self.label_preserving = label_preserving

    def __call__(self, atoms: Atoms) -> Atoms:
        new_atoms = atoms.copy()

        if atoms.cell is not None and atoms.cell.any():
            # Sample max_strain if it's a list [min, max]
            if isinstance(self.max_strain, list):
                strain_scale = np.random.uniform(self.max_strain[0], self.max_strain[1])
            else:
                strain_scale = self.max_strain

            strain = np.random.randn(3, 3) * strain_scale
            strain = (strain + strain.T) / 2
            I = np.eye(3)

            original_cell = atoms.cell.array.copy()
            new_cell = original_cell @ (I + strain)
            new_atoms.set_cell(new_cell, scale_atoms=False)

            # Track cell displacement for denoising (new_cell - original_cell)
            cell_displacement = new_cell - original_cell

            # Accumulate if previous cell displacement exists
            existing_displacement = new_atoms.info.get("cell_noise_displacement")
            if existing_displacement is not None:
                cell_displacement = cell_displacement + np.asarray(
                    existing_displacement
                )

            new_atoms.info["cell_noise_displacement"] = cell_displacement
            new_atoms.info["strain_scale"] = float(strain_scale)
            # Store original cell if not already stored
            if "original_cell" not in new_atoms.info:
                new_atoms.info["original_cell"] = original_cell

        return new_atoms


class RandomRotation:
    """Apply a random 3D rotation to a crystal structure.

    Parameters
    ----------
    max_angle : float, default=30
        Maximum rotation angle in degrees for each axis.
    label_preserving : bool, default=True
        Whether this augmentation preserves labels.

    Notes
    -----
    Energy is preserved (scalar invariant). Forces are rotated by the same
    rotation matrix when present.
    """

    def __init__(self, max_angle: float = 30, label_preserving: bool = True):
        self.max_angle = max_angle
        self.label_preserving = label_preserving

    def __call__(self, atoms: Atoms) -> Atoms:
        new_atoms = atoms.copy()

        # Sample angles uniformly in [-max_angle, max_angle]
        angles = np.random.uniform(-self.max_angle, self.max_angle, 3) * (np.pi / 180)

        cx, sx = np.cos(angles[0]), np.sin(angles[0])
        cy, sy = np.cos(angles[1]), np.sin(angles[1])
        cz, sz = np.cos(angles[2]), np.sin(angles[2])

        R = np.array(
            [
                [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                [-sy, sx * cy, cx * cy],
            ],
            dtype=np.float32,  # Use float32 to match PyTorch default dtype
        )

        new_atoms.positions = new_atoms.positions @ R.T

        if atoms.cell is not None and atoms.cell.any():
            new_atoms.set_cell(atoms.cell @ R.T, scale_atoms=False)

        # Track total rotation applied so downstream transforms can undo it
        existing_rotation = new_atoms.info.get("rotation_matrix")
        if existing_rotation is not None:
            existing_rotation = np.asarray(existing_rotation, dtype=np.float64)
            combined_rotation = R @ existing_rotation
        else:
            combined_rotation = R
        new_atoms.info["rotation_matrix"] = combined_rotation.astype(np.float32)

        # Rotate forces if they exist (for label preservation in hybrid training)
        if hasattr(atoms, "arrays") and "forces" in atoms.arrays:
            # Rotate forces by same rotation matrix, preserving dtype
            new_atoms.arrays["forces"] = (atoms.arrays["forces"] @ R.T).astype(
                atoms.arrays["forces"].dtype
            )

        # Rotate noise_displacement if it exists (from DiffusionNoiseAugmentation)
        # This keeps the noise vector aligned with the rotated coordinate frame
        if hasattr(new_atoms, "arrays") and "noise_displacement" in new_atoms.arrays:
            new_atoms.arrays["noise_displacement"] = (
                new_atoms.arrays["noise_displacement"] @ R.T
            ).astype(np.float32)

        return new_atoms


class RandomSupercell:
    """Create a random supercell of the structure.

    Parameters
    ----------
    max_repeat : int, default=2
        Maximum repeat factor along each axis.
    p : float, default=0.5
        Probability of applying the augmentation.
    """

    def __init__(self, max_repeat: int = 2, p: float = 0.5):
        self.max_repeat = max_repeat
        self.p = p

    def __call__(self, atoms: Atoms) -> Atoms:
        if random.random() > self.p:
            return atoms.copy()

        if atoms.cell is None or not atoms.cell.any():
            return atoms.copy()

        repeats = np.random.randint(1, self.max_repeat + 1, 3)
        P = np.diag(repeats)

        supercell = make_supercell(atoms, P)

        return supercell


class RandomVacancy:
    """Create random vacancies in the structure.

    Parameters
    ----------
    vacancy_prob : float, default=0.05
        Probability of removing each atom.
    """

    label_preserving = False

    def __init__(self, vacancy_prob: float = 0.05):
        self.vacancy_prob = vacancy_prob

    def __call__(self, atoms: Atoms) -> Atoms:
        keep_mask = np.random.rand(len(atoms)) > self.vacancy_prob
        if keep_mask.sum() == 0:
            keep_mask[0] = True
        return atoms[keep_mask]


class AtomMasking:
    """Mask atoms by replacing atomic numbers with a token or random elements.

    Parameters
    ----------
    mask_prob : float, default=0.15
        Probability of masking each atom.
    mask_token : int, default=0
        Atomic number to use as the mask token (used when ``mask_mode="fixed"``).
    return_mask : bool, default=False
        If True, return ``(masked_atoms, mask_indices)``.
    strategy : str, default="random"
        Masking strategy: ``"random"``, ``"element"``, ``"region"``, or ``"wyckoff"``.
    mask_mode : str, default="fixed"
        Replacement mode: ``"fixed"`` or ``"random_element"``.
    random_element_range : tuple[int, int] or list[int], default=(1, 80)
        Atomic numbers for random element masking. If a 2-tuple/list is provided,
        it expands to the full range; otherwise interpreted as explicit choices.
    mask_neighbor_cutoff : float, default=0.0
        If > 0, also mask atoms within this distance (Angstroms) of masked atoms.
    wyckoff_symmetry_threshold : float, default=1e-5
        Symmetry tolerance for spglib when using the ``"wyckoff"`` strategy.
    wyckoff_group_by_element : bool, default=True
        Whether to separate Wyckoff positions by element type.

    Notes
    -----
    Strategies:
    - ``"random"``: randomly mask atoms.
    - ``"element"``: mask all atoms of randomly selected element(s).
    - ``"region"``: mask atoms in a spatial region.
    - ``"wyckoff"``: mask atoms by Wyckoff positions (symmetry-aware).
    """

    # Metadata for hybrid training
    label_preserving = False  # Changes atomic species

    def __init__(
        self,
        mask_prob: float | tuple[float, float] | list[float] = 0.15,
        mask_token: int = 0,
        return_mask: bool = False,
        strategy: str = "random",
        mask_mode: str = "fixed",
        random_element_range: tuple[int, int] = (1, 80),
        exclude_elements: Optional[list[int]] = [
            84,
            85,
            86,
            87,
            88,
        ],  # Exclude Po, Ra by default
        mask_neighbor_cutoff: float = 0.0,
        wyckoff_symmetry_threshold: float = 1e-5,
        wyckoff_group_by_element: bool = True,
    ):
        self._mask_prob = None
        self._mask_prob_min = None
        self._mask_prob_max = None
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.return_mask = return_mask
        self.strategy = strategy
        self.mask_mode = mask_mode
        self.mask_neighbor_cutoff = mask_neighbor_cutoff
        self.wyckoff_symmetry_threshold = wyckoff_symmetry_threshold
        self.wyckoff_group_by_element = wyckoff_group_by_element

        # Convert [min, max] or (min, max) to full list [min, min+1, ..., max]
        if (
            isinstance(random_element_range, (list, tuple))
            and len(random_element_range) == 2
            and isinstance(random_element_range[0], int)
        ):
            self.random_element_range = list(
                range(random_element_range[0], random_element_range[1] + 1)
            )
        else:
            self.random_element_range = random_element_range

        # remove 84, 88
        self.random_element_range = [
            z for z in self.random_element_range if z not in (exclude_elements or [])
        ]

        if self.mask_mode not in ["fixed", "random_element"]:
            raise ValueError(
                f"mask_mode must be 'fixed' or 'random_element', got '{self.mask_mode}'"
            )

    def _get_spatial_neighbors(
        self, atoms: Atoms, indices: np.ndarray, cutoff: float
    ) -> np.ndarray:
        """Get indices of atoms within a cutoff distance.

        Parameters
        ----------
        atoms : Atoms
            ASE atoms object.
        indices : np.ndarray
            Indices of atoms to find neighbors for.
        cutoff : float
            Distance cutoff in Angstroms.

        Returns
        -------
        np.ndarray
            Unique indices of all atoms within cutoff of any atom in ``indices``,
            including the originals.
        """
        if len(indices) == 0:
            return indices

        positions = atoms.positions
        neighbor_set = set(indices.tolist())

        # Check if we have periodic boundary conditions
        has_pbc = atoms.cell is not None and atoms.cell.any() and any(atoms.pbc)

        if has_pbc:
            # Use ASE's neighbor list for periodic systems
            from ase.neighborlist import neighbor_list

            # Get all pairs within cutoff
            i_indices, j_indices = neighbor_list("ij", atoms, cutoff)

            # Find neighbors of masked atoms
            mask_set = set(indices.tolist())
            for i, j in zip(i_indices, j_indices):
                if i in mask_set:
                    neighbor_set.add(j)
                if j in mask_set:
                    neighbor_set.add(i)
        else:
            # Use scipy KDTree for non-periodic systems
            from scipy.spatial import KDTree

            tree = KDTree(positions)
            for idx in indices:
                neighbors = tree.query_ball_point(positions[idx], cutoff)
                neighbor_set.update(neighbors)

        return np.array(sorted(neighbor_set), dtype=int)

    @property
    def mask_prob(self):
        return self._mask_prob

    @mask_prob.setter
    def mask_prob(self, value):
        object.__setattr__(self, "_mask_prob", value)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            object.__setattr__(self, "_mask_prob_min", float(value[0]))
            object.__setattr__(self, "_mask_prob_max", float(value[1]))
        elif value is not None:
            val = float(value)
            object.__setattr__(self, "_mask_prob_min", val)
            object.__setattr__(self, "_mask_prob_max", val)
        else:
            object.__setattr__(self, "_mask_prob_min", None)
            object.__setattr__(self, "_mask_prob_max", None)

    @property
    def mask_prob_min(self) -> Optional[float]:
        return self._mask_prob_min

    @mask_prob_min.setter
    def mask_prob_min(self, value: Optional[float]):
        object.__setattr__(
            self, "_mask_prob_min", None if value is None else float(value)
        )
        self._sync_mask_prob_from_bounds()

    @property
    def mask_prob_max(self) -> Optional[float]:
        return self._mask_prob_max

    @mask_prob_max.setter
    def mask_prob_max(self, value: Optional[float]):
        object.__setattr__(
            self, "_mask_prob_max", None if value is None else float(value)
        )
        self._sync_mask_prob_from_bounds()

    def _sync_mask_prob_from_bounds(self):
        """Keep ``mask_prob`` list in sync when min/max bounds are updated."""
        min_prob = self._mask_prob_min
        max_prob = self._mask_prob_max

        if min_prob is None and max_prob is None:
            return

        if min_prob is None:
            min_prob = max_prob
        if max_prob is None:
            max_prob = min_prob

        if min_prob == max_prob:
            object.__setattr__(self, "_mask_prob", float(min_prob))
        else:
            low = float(min(min_prob, max_prob))
            high = float(max(min_prob, max_prob))
            object.__setattr__(self, "_mask_prob", [low, high])

    def _sample_mask_probability(self) -> float:
        """Sample the mask probability respecting schedulable bounds.

        Returns
        -------
        float
            Sampled mask probability.
        """
        min_prob = self._mask_prob_min
        max_prob = self._mask_prob_max

        if min_prob is not None or max_prob is not None:
            if min_prob is None:
                min_prob = max_prob
            if max_prob is None:
                max_prob = min_prob

            if min_prob == max_prob:
                return float(min_prob)

            low = float(min(min_prob, max_prob))
            high = float(max(min_prob, max_prob))
            return float(np.random.uniform(low, high))

        # Fallback for legacy paths when only mask_prob is provided
        if isinstance(self.mask_prob, (list, tuple)) and len(self.mask_prob) >= 2:
            return float(np.random.uniform(self.mask_prob[0], self.mask_prob[1]))

        return float(self.mask_prob)

    def _get_wyckoff_positions(self, atoms: Atoms) -> Optional[np.ndarray]:
        """Extract Wyckoff position labels for each atom using spglib.

        Parameters
        ----------
        atoms : Atoms
            ASE atoms object.

        Returns
        -------
        np.ndarray or None
            Array of Wyckoff position indices for each atom. Returns ``None`` if
            symmetry analysis fails.
        """
        if atoms.cell is None or not atoms.cell.any():
            return None

        # Prepare structure for spglib
        cell = (
            atoms.cell.array,
            atoms.get_scaled_positions(),
            atoms.numbers,
        )

        try:
            # Get symmetry dataset from spglib
            dataset = spglib.get_symmetry_dataset(
                cell, symprec=self.wyckoff_symmetry_threshold
            )

            if dataset is None:
                return None

            # dataset.equivalent_atoms gives us the mapping we need
            # Atoms with the same value are symmetrically equivalent (same Wyckoff position)
            return dataset.equivalent_atoms

        except Exception:
            # If spglib fails, fall back to None
            return None

    def __call__(self, atoms: Atoms) -> Atoms | tuple[Atoms, np.ndarray]:
        new_atoms = atoms.copy()
        n_atoms = len(atoms)
        original_numbers = atoms.numbers.copy()

        # Sample mask probability (supports schedulable min/max bounds)
        mask_prob = self._sample_mask_probability()

        if self.strategy == "random":
            mask_indices = np.where(np.random.rand(n_atoms) < mask_prob)[0]

        elif self.strategy == "element":
            unique_elements = np.unique(atoms.numbers)
            n_elements_to_mask = max(1, int(len(unique_elements) * mask_prob))
            elements_to_mask = np.random.choice(
                unique_elements,
                size=min(n_elements_to_mask, len(unique_elements)),
                replace=False,
            )
            mask_indices = np.where(np.isin(atoms.numbers, elements_to_mask))[0]

        elif self.strategy == "region":
            positions = atoms.positions
            if len(positions) > 0:
                center_idx = np.random.randint(len(positions))
                center = positions[center_idx]

                distances = np.linalg.norm(positions - center, axis=1)

                sorted_distances = np.sort(distances)
                n_mask = max(1, int(n_atoms * mask_prob))
                radius = sorted_distances[min(n_mask, len(sorted_distances) - 1)]
                mask_indices = np.where(distances <= radius)[0]
            else:
                mask_indices = np.array([])

        elif self.strategy == "wyckoff":
            # Get Wyckoff positions for all atoms
            wyckoff_labels = self._get_wyckoff_positions(atoms)

            if wyckoff_labels is None:
                # Fallback to random masking if symmetry detection fails
                mask_indices = np.where(np.random.rand(n_atoms) < mask_prob)[0]
            else:
                # Group atoms by Wyckoff position (and optionally by element)
                if self.wyckoff_group_by_element:
                    # Create composite labels: (wyckoff_position, element)
                    # This allows masking different elements in the same Wyckoff position separately
                    composite_labels = np.array(
                        [(wyckoff_labels[i], atoms.numbers[i]) for i in range(n_atoms)]
                    )
                    # Convert to unique integer labels
                    unique_groups, group_indices = np.unique(
                        composite_labels, axis=0, return_inverse=True
                    )
                    effective_labels = group_indices
                else:
                    # Use only Wyckoff positions
                    effective_labels = wyckoff_labels

                unique_positions = np.unique(effective_labels)
                n_positions = len(unique_positions)

                # Sample Wyckoff positions to mask until we reach target mask_prob
                target_n_masked = int(n_atoms * mask_prob)
                mask_indices = []

                # Shuffle positions for random selection
                shuffled_positions = unique_positions.copy()
                np.random.shuffle(shuffled_positions)

                n_masked = 0
                for pos in shuffled_positions:
                    if n_masked >= target_n_masked:
                        break

                    # Add all atoms from this Wyckoff position (+ element group)
                    pos_indices = np.where(effective_labels == pos)[0]
                    mask_indices.extend(pos_indices)
                    n_masked += len(pos_indices)

                mask_indices = np.array(mask_indices, dtype=int)

        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")

        # Expand mask to include neighbors if cutoff is set
        # This prevents the model from cheating by looking at unmasked neighbors
        if self.mask_neighbor_cutoff > 0 and len(mask_indices) > 0:
            mask_indices = self._get_spatial_neighbors(
                atoms, mask_indices, self.mask_neighbor_cutoff
            )

        if len(mask_indices) > 0:
            if (
                hasattr(new_atoms, "arrays")
                and "original_numbers" not in new_atoms.arrays
            ):
                new_atoms.arrays["original_numbers"] = atoms.numbers.copy()

            # Apply masking based on mode
            if self.mask_mode == "random_element":
                # Replace each masked atom with a random element from the list
                for idx in mask_indices:
                    random_z = random.choice(self.random_element_range)
                    new_atoms.numbers[idx] = random_z
            else:
                # Fixed mask token (original behavior)
                new_atoms.numbers[mask_indices] = self.mask_token

        new_atoms.info["original_numbers"] = original_numbers

        if self.return_mask:
            return new_atoms, mask_indices
        return new_atoms


class FullAtomMasking:
    """Replace every atom with the mask token to remove composition cues.

    Parameters
    ----------
    mask_token : int, default=54
        Atomic number to use as the mask token.
    """

    label_preserving = False

    def __init__(self, mask_token: int = 54):
        self.mask_token = mask_token

    def __call__(self, atoms: Atoms) -> Atoms:
        new_atoms = atoms.copy()
        if "original_numbers" not in new_atoms.info:
            new_atoms.info["original_numbers"] = atoms.numbers.copy()
        masked = np.full(len(atoms), self.mask_token, dtype=int)
        new_atoms.set_atomic_numbers(masked)
        new_atoms.info["full_masking"] = True
        return new_atoms


class CompositeAugmentation:
    """Apply multiple augmentations in sequence.

    Parameters
    ----------
    augmentations : list
        Sequence of augmentation callables.
    probabilities : list, optional
        Per-augmentation application probabilities. Defaults to all 1.0.
    """

    def __init__(self, augmentations: list, probabilities: Optional[list] = None):
        self.augmentations = augmentations
        self.probabilities = probabilities or [1.0] * len(augmentations)

    def __call__(self, atoms: Atoms) -> Atoms:
        result = atoms.copy()

        for aug, prob in zip(self.augmentations, self.probabilities):
            if random.random() < prob:
                result = aug(result)

        return result


class RandomAugmentation:
    """Randomly select and apply one augmentation from a list.

    Parameters
    ----------
    augmentations : list
        Sequence of augmentation callables.
    weights : list, optional
        Sampling weights for each augmentation.
    """

    def __init__(self, augmentations: list, weights: Optional[list] = None):
        self.augmentations = augmentations
        self.weights = weights

    def __call__(self, atoms: Atoms) -> Atoms:
        if self.weights:
            aug = random.choices(self.augmentations, weights=self.weights, k=1)[0]
        else:
            aug = random.choice(self.augmentations)

        return aug(atoms)


class AtomSwap:
    """Swap positions of two atoms in the structure.

    Parameters
    ----------
    swap_fraction : float or tuple[float, float], default=0.5
        Fraction of atom pairs to swap. Can be a single float or a tuple ``(min, max)``
        for random sampling. Scales with structure size.

    Notes
    -----
    This augmentation changes composition and is not label-preserving.
    """

    def __init__(
        self,
        swap_fraction: float | tuple[float, float] = 0.5,
    ):
        self.swap_fraction = swap_fraction
        self.label_preserving = False

    def _get_swap_pairs(self, atoms: Atoms, n_swaps: int) -> list[tuple[int, int]]:
        """Generate random pairs of atom indices to swap.

        Parameters
        ----------
        atoms : Atoms
            ASE atoms object.
        n_swaps : int
            Maximum number of pairs to generate.

        Returns
        -------
        list[tuple[int, int]]
            List of index pairs.
        """
        n_atoms = len(atoms)
        if n_atoms < 2:
            return []

        # Randomly select pairs without replacement
        available_indices = set(range(n_atoms))
        pairs = []

        for _ in range(min(n_swaps, n_atoms // 2)):
            if len(available_indices) < 2:
                break
            idx1, idx2 = random.sample(list(available_indices), 2)
            pairs.append((idx1, idx2))
            available_indices.discard(idx1)
            available_indices.discard(idx2)

        return pairs

    def __call__(self, atoms: Atoms) -> Atoms:
        new_atoms = atoms.copy()

        # Get all possible pairs
        max_pairs = len(atoms) // 2
        all_pairs = self._get_swap_pairs(atoms, max_pairs)

        if not all_pairs:
            return new_atoms

        # Determine fraction of pairs to swap
        if isinstance(self.swap_fraction, tuple):
            fraction = np.random.uniform(self.swap_fraction[0], self.swap_fraction[1])
        else:
            fraction = self.swap_fraction

        # Calculate number of pairs to swap
        n_pairs_to_swap = max(1, int(len(all_pairs) * fraction))

        # Randomly select pairs to swap
        pairs_to_swap = random.sample(all_pairs, min(n_pairs_to_swap, len(all_pairs)))

        # Perform swaps (positions only, not forces or other properties)
        for idx1, idx2 in pairs_to_swap:
            new_atoms.positions[[idx1, idx2]] = new_atoms.positions[[idx2, idx1]]

        return new_atoms


class PeriodicGroupSubstitution:
    """Substitute atoms with chemically similar elements.

    Parameters
    ----------
    sub_prob : float, default=0.3
        Probability of substituting each atom.
    strategy : str, default="same_group"
        Substitution strategy: ``"same_group"``, ``"similar_radius"``,
        ``"similar_electronegativity"``, or ``"same_period"``.
    radius_tolerance : float, default=0.3
        Tolerance for radius-based substitutions.
    en_tolerance : float, default=0.5
        Tolerance for electronegativity-based substitutions.
    """

    label_preserving = False

    # Exclude radioactive/unstable elements (Z > 83) and rare elements
    EXCLUDED_ELEMENTS = {
        43,
        61,
        84,
        85,
        86,
        87,
        88,
        89,
    }  # Tc, Pm, Po, At, Rn, Fr, Ra, Ac and beyond

    PERIODIC_GROUPS = {
        1: [3, 11, 19, 37, 55],
        2: [4, 12, 20, 38, 56],
        3: [21, 39, 57],
        4: [22, 40, 72],
        5: [23, 41, 73],
        6: [24, 42, 74],
        7: [25, 75],
        8: [26, 44, 76],
        9: [27, 45, 77],
        10: [28, 46, 78],
        11: [29, 47, 79],
        12: [30, 48, 80],
        13: [5, 13, 31, 49, 81],
        14: [6, 14, 32, 50, 82],
        15: [7, 15, 33, 51, 83],
        16: [8, 16, 34, 52],
        17: [9, 17, 35, 53],
        18: [2, 10, 18, 36, 54],
    }

    ATOMIC_RADII = {
        1: 0.31,
        3: 1.28,
        4: 0.96,
        5: 0.84,
        6: 0.76,
        7: 0.71,
        8: 0.66,
        9: 0.57,
        11: 1.66,
        12: 1.41,
        13: 1.21,
        14: 1.11,
        15: 1.07,
        16: 1.05,
        17: 1.02,
        19: 2.03,
        20: 1.76,
        21: 1.70,
        22: 1.60,
        23: 1.53,
        24: 1.39,
        25: 1.39,
        26: 1.32,
        27: 1.26,
        28: 1.24,
        29: 1.32,
        30: 1.22,
        31: 1.22,
        32: 1.20,
        33: 1.19,
        34: 1.20,
        35: 1.20,
        37: 2.20,
        38: 1.95,
        39: 1.90,
        40: 1.75,
        41: 1.64,
        42: 1.54,
        44: 1.46,
        45: 1.42,
        46: 1.39,
        47: 1.45,
        48: 1.44,
        49: 1.42,
        50: 1.39,
        51: 1.39,
        52: 1.38,
        53: 1.39,
        55: 2.44,
        56: 2.15,
        57: 2.07,
        72: 1.75,
        73: 1.70,
        74: 1.62,
        75: 1.51,
        76: 1.44,
        77: 1.41,
        78: 1.36,
        79: 1.36,
        80: 1.32,
        81: 1.45,
        82: 1.46,
        83: 1.48,
    }

    ELECTRONEGATIVITY = {
        1: 2.20,
        3: 0.98,
        4: 1.57,
        5: 2.04,
        6: 2.55,
        7: 3.04,
        8: 3.44,
        9: 3.98,
        11: 0.93,
        12: 1.31,
        13: 1.61,
        14: 1.90,
        15: 2.19,
        16: 2.58,
        17: 3.16,
        19: 0.82,
        20: 1.00,
        21: 1.36,
        22: 1.54,
        23: 1.63,
        24: 1.66,
        25: 1.55,
        26: 1.83,
        27: 1.88,
        28: 1.91,
        29: 1.90,
        30: 1.65,
        31: 1.81,
        32: 2.01,
        33: 2.18,
        34: 2.55,
        35: 2.96,
        37: 0.82,
        38: 0.95,
        39: 1.22,
        40: 1.33,
        41: 1.6,
        42: 2.16,
        44: 2.2,
        45: 2.28,
        46: 2.20,
        47: 1.93,
        48: 1.69,
        49: 1.78,
        50: 1.96,
        51: 2.05,
        52: 2.1,
        53: 2.66,
        55: 0.79,
        56: 0.89,
        57: 1.10,
        72: 1.3,
        73: 1.5,
        74: 2.36,
        75: 1.9,
        76: 2.2,
        77: 2.20,
        78: 2.28,
        79: 2.54,
        80: 2.00,
        81: 1.62,
        82: 2.33,
        83: 2.02,
    }

    def __init__(
        self,
        sub_prob: float = 0.3,
        strategy: str = "same_group",
        radius_tolerance: float = 0.3,
        en_tolerance: float = 0.5,
    ):
        self.sub_prob = sub_prob
        self.strategy = strategy
        self.radius_tolerance = radius_tolerance
        self.en_tolerance = en_tolerance
        self.z_to_group = {z: g for g, els in self.PERIODIC_GROUPS.items() for z in els}

    def _get_candidates(self, z: int) -> list[int]:
        if self.strategy == "same_group":
            group = self.z_to_group.get(z)
            return (
                [
                    e
                    for e in self.PERIODIC_GROUPS.get(group, [])
                    if e != z and e not in self.EXCLUDED_ELEMENTS
                ]
                if group
                else []
            )
        elif self.strategy == "similar_radius":
            if z not in self.ATOMIC_RADII:
                return []
            r = self.ATOMIC_RADII[z]
            return [
                oz
                for oz, or_ in self.ATOMIC_RADII.items()
                if oz != z
                and oz not in self.EXCLUDED_ELEMENTS
                and abs(or_ - r) <= self.radius_tolerance
            ]
        elif self.strategy == "similar_electronegativity":
            if z not in self.ELECTRONEGATIVITY:
                return []
            en = self.ELECTRONEGATIVITY[z]
            return [
                oz
                for oz, oen in self.ELECTRONEGATIVITY.items()
                if oz != z
                and oz not in self.EXCLUDED_ELEMENTS
                and abs(oen - en) <= self.en_tolerance
            ]
        elif self.strategy == "same_period":
            periods = [(1, 3), (3, 11), (11, 19), (19, 37), (37, 55), (55, 84)]
            for start, end in periods:
                if start <= z < end:
                    return [
                        oz
                        for oz in range(start, end)
                        if oz != z and oz not in self.EXCLUDED_ELEMENTS
                    ]
        return []

    def __call__(self, atoms: Atoms) -> Atoms:
        new_atoms = atoms.copy()
        sub_mask = np.random.rand(len(atoms)) < self.sub_prob

        for i in np.where(sub_mask)[0]:
            candidates = self._get_candidates(new_atoms.numbers[i])
            if candidates:
                new_atoms.numbers[i] = random.choice(candidates)

        return new_atoms
