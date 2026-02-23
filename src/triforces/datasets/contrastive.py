"""Contrastive datasets for generating augmented views."""

import random
from typing import Any, Dict

from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    """Generate multiple augmented views per sample for contrastive learning.

    Parameters
    ----------
    dataset : Dataset
        Base dataset providing samples.
    augmentations : dict[str, Any], optional
        Mapping of augmentation names to callables.
    augmentation_probabilities : dict[str, float], optional
        Optional per-augmentation application probabilities in ``[0, 1]``.
        Defaults to ``1.0`` per augmentation.
    n_augmentation_views : int, default=2
        Number of augmented views per sample.
    n_opposite_views : int, default=0
        Number of "opposite" views (uses ``_apply_opposite_augmentation``).
    include_original : bool, default=False
        Whether to include the unmodified sample as a view.
    apply_augmentations_prob : float, default=1.0
        Probability of applying augmentations for a view.
    return_pairs : bool, default=True
        Whether ``__getitem__`` returns all views for a sample (pair mode).

    Notes
    -----
    When ``return_pairs=True``, ``__getitem__`` returns a tuple/list of views for a
    base sample so the collate function can keep pairs together under shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        augmentations: Dict[str, Any] | None = None,
        augmentation_probabilities: Dict[str, float] | None = None,
        n_augmentation_views: int = 2,
        n_opposite_views: int = 0,
        include_original: bool = False,
        apply_augmentations_prob: float = 1.0,
        return_pairs: bool = True,
    ):
        self.dataset = dataset
        self.augmentations = augmentations or {}
        self.augmentation_probabilities = {
            str(k): float(v) for k, v in (augmentation_probabilities or {}).items()
        }
        self.n_augmentation_views = n_augmentation_views
        self.n_opposite_views = n_opposite_views
        self.include_original = include_original
        self.apply_augmentations_prob = apply_augmentations_prob
        self.return_pairs = bool(return_pairs)
        self._validate_augmentation_config()

        # Calculate total views per sample
        self.total_views = n_augmentation_views + n_opposite_views
        if include_original:
            self.total_views += 1

        # Each sample gets expanded to multiple views
        if self.total_views < 1:
            raise ValueError(
                "ContrastiveDataset requires at least one view per sample."
            )
        self._length = (
            len(dataset) if self.return_pairs else len(dataset) * self.total_views
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Any:
        if self.return_pairs:
            sample_idx = int(idx)
            base_sample = self.dataset[sample_idx]
            pair_id = getattr(base_sample, "pair_id", sample_idx)
            views: list[Any] = []
            for view_idx in range(self.total_views):
                sample = base_sample if view_idx == 0 else self.dataset[sample_idx]
                views.append(self._build_view(sample, view_idx, pair_id))
            if len(views) == 2:
                return (views[0], views[1])
            return views

        # Convert linear index to (sample_idx, view_idx)
        sample_idx = idx // self.total_views
        view_idx = idx % self.total_views

        sample = self.dataset[sample_idx]
        pair_id = getattr(sample, "pair_id", sample_idx)
        return self._build_view(sample, view_idx, pair_id)

    def _build_view(self, sample: Any, view_idx: int, pair_id: int) -> Any:
        current_view = 0

        if self.include_original:
            if view_idx == current_view:
                sample.pair_id = pair_id
                return sample
            current_view += 1

        should_apply_augmentations = random.random() <= self.apply_augmentations_prob
        if self.augmentations and should_apply_augmentations:
            if view_idx < current_view + self.n_augmentation_views:
                sample = self._apply_augmentations(sample)
            else:
                sample = self._apply_opposite_augmentation(sample)

        sample.pair_id = pair_id
        return sample

    def _validate_augmentation_config(self) -> None:
        if not 0.0 <= float(self.apply_augmentations_prob) <= 1.0:
            raise ValueError("apply_augmentations_prob must be in [0, 1].")

        for name, prob in self.augmentation_probabilities.items():
            if name not in self.augmentations:
                raise ValueError(
                    "augmentation_probabilities contains unknown augmentation "
                    f"name '{name}'."
                )
            if not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"augmentation probability for '{name}' must be in [0, 1]."
                )

    def _apply_to_sample(self, sample: Any, augmentation: Any) -> Any:
        """Apply one augmentation callable to a sample."""
        if hasattr(sample, "atoms"):
            sample.atoms = augmentation(sample.atoms)
            return sample
        if hasattr(sample, "structure"):
            sample.structure = augmentation(sample.structure)
            return sample
        return augmentation(sample)

    def _apply_augmentations(self, sample: Any) -> Any:
        """Apply augmentations according to per-augmentation probabilities."""
        if not self.augmentations:
            return sample

        for name in self.augmentations.keys():
            prob = float(self.augmentation_probabilities.get(name, 1.0))
            if random.random() < prob:
                sample = self._apply_to_sample(sample, self.augmentations[name])
        return sample

    def _apply_opposite_augmentation(self, sample: Any) -> Any:
        return self._apply_augmentations(sample)


class AugmentationDataset(ContrastiveDataset):
    """Canonical name for the dataset that generates augmented views."""
