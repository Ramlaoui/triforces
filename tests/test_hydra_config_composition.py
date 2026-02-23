from __future__ import annotations

from hydra import compose, initialize_config_module


def test_train_contrastive_orb_barlow_composes() -> None:
    with initialize_config_module(config_module="triforces.configs", version_base=None):
        cfg = compose(config_name="train_contrastive_orb_barlow")

    assert cfg.model._target_ == "triforces.models.adapter_model.AdapterModel"
    assert (
        cfg.model.heads.proj._target_
        == "triforces.models.heads.BarlowTwinsProjectionHead"
    )
    assert cfg.loss._target_ == "triforces.losses.BarlowTwinsLoss"


def test_train_supervised_composes() -> None:
    with initialize_config_module(config_module="triforces.configs", version_base=None):
        cfg = compose(config_name="train_supervised")

    assert cfg.model._target_ == "triforces.models.adapter_model.AdapterModel"
    assert (
        cfg.model.heads.proj._target_ == "triforces.models.heads.DirectSupervisedHead"
    )
    assert cfg.collate.contrastive is False
    assert cfg.collate._target_ == "triforces.data.pyg_collate"
    assert cfg.loss._target_ == "triforces.losses.SupervisedLoss"
    assert cfg.train.lr == 0.001
    assert cfg.logger.enabled is True


def test_train_contrastive_uses_trainer_and_wandb_presets() -> None:
    with initialize_config_module(config_module="triforces.configs", version_base=None):
        cfg = compose(config_name="train_contrastive")

    assert cfg.train.batch_size == 8
    assert cfg.train.log_every == 10
    assert cfg.train.checkpoint.enabled is True
    assert cfg.train.checkpoint.monitor == "loss"
    assert cfg.train.hooks.linear_probe.enabled is False
    assert cfg.logger.project == "triforces"
    assert cfg.logger.enabled is True


def test_orb_supervised_direct_experiment_composes() -> None:
    with initialize_config_module(config_module="triforces.configs", version_base=None):
        cfg = compose(config_name="experiments/supervised/orb/direct")

    assert (
        cfg.model.backbone.interaction._target_
        == "triforces.models.interaction.orb.Orb"
    )
    assert (
        cfg.model.heads.proj._target_ == "triforces.models.heads.DirectSupervisedHead"
    )
    assert cfg.model.backbone.interaction.compute_displacement is False
    assert cfg.collate.contrastive is False
    assert list(cfg.dataset.add_targets) == ["energy", "forces"]


def test_orb_supervised_energy_experiment_composes() -> None:
    with initialize_config_module(config_module="triforces.configs", version_base=None):
        cfg = compose(config_name="experiments/supervised/orb/energy_conserving")

    assert (
        cfg.model.backbone.interaction._target_
        == "triforces.models.interaction.orb.Orb"
    )
    assert (
        cfg.model.heads.proj._target_ == "triforces.models.heads.EnergyConservingHead"
    )
    assert cfg.model.backbone.interaction.compute_displacement is True
    assert cfg.collate.contrastive is False
    assert list(cfg.dataset.add_targets) == ["energy", "forces"]


def test_orb_pretraining_experiment_uses_reconstruction_lejepa_loss() -> None:
    with initialize_config_module(config_module="triforces.configs", version_base=None):
        cfg = compose(config_name="experiments/pretraining/orb/main_triforces")

    assert cfg.loss._target_ == "triforces.losses.ReconstructionLoss"
    assert cfg.loss.base_loss._target_ == "triforces.losses.LeJEPALoss"
    assert cfg.dataset._target_ == "triforces.datasets.AugmentationDataset"
    assert cfg.dataset.dataset._target_ == "triforces.data.lemat_bulk.LeMatBulkDataset"
    assert cfg.model.heads.proj._target_ == "triforces.models.heads.ProjectionHead"
    assert cfg.model.heads.denoise._target_ == "triforces.models.heads.DirectVectorHead"
    assert cfg.model.heads.denoise.output_dim == 3
    assert (
        cfg.model.heads.unmask._target_ == "triforces.models.heads.ClassificationHead"
    )
    assert list(cfg.dataset.augmentations.keys()) == ["noise", "rotate", "mask"]
    assert cfg.dataset.augmentation_probabilities.noise == 1.0
    assert cfg.dataset.augmentation_probabilities.rotate == 0.0
    assert cfg.dataset.augmentation_probabilities.mask == 0.0
    assert float(cfg.loss.base_weight) == 0.0
    assert cfg.loss.atom_type_weight == 1.0


def test_esen_pretraining_experiment_uses_reconstruction_lejepa_loss() -> None:
    with initialize_config_module(config_module="triforces.configs", version_base=None):
        cfg = compose(config_name="experiments/pretraining/esen/main_triforces")

    assert cfg.loss._target_ == "triforces.losses.ReconstructionLoss"
    assert cfg.loss.base_loss._target_ == "triforces.losses.LeJEPALoss"
    assert cfg.dataset._target_ == "triforces.datasets.AugmentationDataset"
    assert cfg.dataset.dataset._target_ == "triforces.data.lemat_bulk.LeMatBulkDataset"
    assert cfg.model._target_ == "triforces.models.adapter_model.AdapterModel"
    assert (
        cfg.model.backbone.interaction._target_
        == "triforces.models.interaction.esen.eSEN"
    )
    assert cfg.model.heads.proj._target_ == "triforces.models.heads.ProjectionHead"
    assert (
        cfg.model.heads.denoise._target_
        == "triforces.models.heads.EquivariantVectorHead"
    )
    assert cfg.model.heads.denoise.output_dim == 3
    assert cfg.model.heads.denoise.use_tensor_product is True
    assert (
        cfg.model.heads.unmask._target_ == "triforces.models.heads.ClassificationHead"
    )
    assert list(cfg.dataset.augmentations.keys()) == ["noise", "rotate", "mask"]
    assert cfg.dataset.augmentation_probabilities.noise == 1.0
    assert cfg.dataset.augmentation_probabilities.rotate == 1.0
    assert cfg.dataset.augmentation_probabilities.mask == 1.0
    assert float(cfg.loss.base_weight) == 0.1
    assert cfg.loss.atom_type_weight == 1.0


def test_mace_pretraining_experiment_uses_reconstruction_lejepa_loss() -> None:
    with initialize_config_module(config_module="triforces.configs", version_base=None):
        cfg = compose(config_name="experiments/pretraining/mace/main_triforces")

    assert cfg.loss._target_ == "triforces.losses.ReconstructionLoss"
    assert cfg.loss.base_loss._target_ == "triforces.losses.LeJEPALoss"
    assert cfg.dataset._target_ == "triforces.datasets.AugmentationDataset"
    assert cfg.dataset.dataset._target_ == "triforces.data.lemat_bulk.LeMatBulkDataset"
    assert cfg.model._target_ == "triforces.models.adapter_model.AdapterModel"
    assert (
        cfg.model.backbone.interaction._target_
        == "triforces.models.interaction.mace.MACE"
    )
    assert cfg.model.heads.proj._target_ == "triforces.models.heads.ProjectionHead"
    assert (
        cfg.model.heads.denoise._target_
        == "triforces.models.heads.EnergyConservingHead"
    )
    assert cfg.model.heads.denoise.forces_output_key == "noise_displacement"
    assert cfg.model.heads.denoise.include_energy is False
    assert (
        cfg.model.heads.unmask._target_ == "triforces.models.heads.ClassificationHead"
    )
    assert list(cfg.dataset.augmentations.keys()) == ["noise", "rotate", "mask"]
    assert cfg.dataset.augmentation_probabilities.noise == 1.0
    assert cfg.dataset.augmentation_probabilities.rotate == 1.0
    assert cfg.dataset.augmentation_probabilities.mask == 1.0
    assert float(cfg.loss.base_weight) == 0.1
    assert cfg.loss.atom_type_weight == 1.0
