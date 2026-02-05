from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import partial
import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .chemistry import stress_to_voigt_6
from .model_outputs import ModelOutputs
from .normalization import NormalizationState, NormalizationType

logger = logging.getLogger("triforces")


def get_nested_attr(obj, attr_path: str):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def make_hook(fn, hook_features, m, input_embeddings, output_embeddings):
    return fn(m, input_embeddings, output_embeddings, hook_features)


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        targets: Optional[list[str]] = None,
        hook_fns: Optional[dict[str, Callable]] = None,
        freeze_weights: Optional[list[str]] = None,
        set_targets: bool = True,
    ):
        super().__init__()
        if set_targets:
            self.set_targets(targets)
        self.requires_grad_for_inference = False

        self.hook_fns = hook_fns or {}
        self.hook_features = {}
        self.freeze_weights = freeze_weights or []

    def _post_init(self) -> None:
        self.hook_handlers = {}
        for hook_module, hook_fn in self.hook_fns.items():
            logger.debug(f"Registering hook for {hook_module}")
            self.hook_handlers[hook_module] = get_nested_attr(
                self, hook_module
            ).register_forward_hook(partial(make_hook, hook_fn, self.hook_features))

        for weight_path in self.freeze_weights:
            for name, param in get_nested_attr(self, weight_path).named_parameters():
                param.requires_grad = False
                logger.info(f"Freezing {weight_path}.{name}")

    def disable_heads(self, disable_attributes: list[str] | None = None):
        if disable_attributes is None:
            return
        logger.warning(
            f"disable_heads is not implemented for this model, skipping for {self.get_model_name()}"
        )

    def add_hook_features(
        self, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for key, value in self.hook_features.items():
            if key in outputs:
                logger.warning("Hook feature %s already in outputs; overwriting.", key)
            outputs[key] = value
        return outputs

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        training: bool = False,
        transform: Optional[Callable[..., object]] = None,
    ) -> dict:
        raise NotImplementedError

    @property
    def possible_targets(self):
        return ["energy", "forces"]

    @classmethod
    def get_model_name(cls):
        return cls.__name__

    def set_targets(self, targets: Optional[list[str]] = None):
        possible_targets = self.possible_targets
        self.targets = list(set(possible_targets) & set(targets or []))

    def _format_model_outputs(
        self,
        batch: Batch,
        raw_model_outputs: dict[str, torch.Tensor],
        model_outputs: ModelOutputs,
        add_keys: list[str] = [],
    ) -> ModelOutputs:
        for key in add_keys:
            model_outputs[key] = raw_model_outputs[key]
        return model_outputs

    def format_model_outputs(
        self,
        batch: Batch,
        raw_model_outputs: dict[str, torch.Tensor],
        add_keys: list[str] = [],
    ) -> ModelOutputs:
        batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        kwargs = {
            "batch": batch.batch if hasattr(batch, "batch") else None,
            "batch_size": batch_size,
            "ptr": batch.ptr if hasattr(batch, "ptr") else None,
            "attributes": {},
        }

        normalization_state = NormalizationState()

        if "energy" in raw_model_outputs and raw_model_outputs["energy"] is not None:
            kwargs["energy"] = raw_model_outputs["energy"]
            normalization_state.add_transform(
                NormalizationType.MODEL_OUTPUT, params={"key": "energy"}
            )

        if "forces" in raw_model_outputs and raw_model_outputs["forces"] is not None:
            kwargs["forces"] = raw_model_outputs["forces"]
            normalization_state.add_transform(
                NormalizationType.MODEL_OUTPUT, params={"key": "forces"}
            )

        for key, value in raw_model_outputs.items():
            if key not in ["energy", "forces"] and value is not None:
                if key == "stress":
                    value = stress_to_voigt_6(value)
                kwargs["attributes"][key] = value
                normalization_state.add_transform(
                    NormalizationType.MODEL_OUTPUT, params={"key": key}
                )

        kwargs["normalization_state"] = normalization_state

        model_outputs = ModelOutputs(**kwargs)
        return self._format_model_outputs(
            batch, raw_model_outputs, model_outputs, add_keys
        )

    def get_backbone_outputs(
        self, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        add_keys = [
            "pos",
            "positions",
            "cell",
            "lattice",
            "displacement",
            "unit_shifts",
        ]
        return {
            key: outputs[key]
            for key in outputs
            if key.startswith("backbone_") or key in add_keys
        }

    def get_readout(self) -> tuple[nn.Module, int]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_log_params(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "targets": list(self.targets),
            "requires_grad_for_inference": self.requires_grad_for_inference,
        }
