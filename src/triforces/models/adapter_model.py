from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from .base import Model
from .utils import build_output_batch, BackboneOutputs

from torch_geometric.data import Batch

logger = logging.getLogger("triforces")


class AdapterModel(Model):
    """Wrap a backbone and apply multiple heads with a strict interface.

    Parameters
    ----------
    backbone : nn.Module
        Backbone model returning ``BackboneOutputs``.
    heads : dict[str, nn.Module], optional
        Mapping of head names to head modules.
    use_model_readout : bool or dict[str, bool], optional
        Whether to call ``get_model_readout`` on each head using the backbone.
    disable_heads : list[str] or None, optional
        Head names to disable in the backbone, if supported.
    prefix_outputs : bool, optional
        Whether to prefix head outputs with the head name.
    **kwargs : Any
        Additional keyword arguments forwarded to ``Model.__init__``.

    Notes
    -----
    Backbone forward signature
        ``forward(batch: Batch, training: bool = False, transform: Any = None)``
        ``-> BackboneOutputs``.
    Head forward signature
        ``forward(backbone_outputs: BackboneOutputs, batch: Batch,``
        ``outputs: dict[str, Any] | None = None, training: bool = False,``
        ``transform: Any = None, **kwargs: Any) -> dict[str, Any]``.
    The ``forward`` method returns a new ``Batch`` with batch metadata and
    output attributes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        heads: dict[str, nn.Module] | None = None,
        use_model_readout: bool | dict[str, bool] = True,
        disable_heads: list[str] | None = None,
        prefix_outputs: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(set_targets=False, **kwargs)

        self.backbone = backbone
        self.prefix_outputs = prefix_outputs

        if disable_heads is not None and hasattr(self.backbone, "disable_heads"):
            self.backbone.disable_heads(disable_heads)

        heads = heads or {}
        if isinstance(use_model_readout, bool):
            use_model_readout = {k: use_model_readout for k in heads}

        for key, head in heads.items():
            head.key_name = key
            if use_model_readout.get(key, False) and hasattr(head, "get_model_readout"):
                logger.info("Using model readout for head %s", key)
                head.get_model_readout(self.backbone)

        self.heads = nn.ModuleDict(heads)

        heads_require_grad = any(
            getattr(head, "requires_grad_for_inference", False)
            for head in self.heads.values()
        )
        self.requires_grad_for_inference = (
            getattr(self.backbone, "requires_grad_for_inference", False)
            or heads_require_grad
        )
        if heads_require_grad:
            self._propagate_requires_grad(self.backbone)

        self.set_targets(self.possible_targets)
        self._post_init()

    def _propagate_requires_grad(self, model: nn.Module) -> None:
        setattr(model, "requires_grad_for_inference", True)
        if hasattr(model, "model"):
            self._propagate_requires_grad(model.model)

    def _post_init(self) -> None:
        super()._post_init()
        if hasattr(self.backbone, "_post_init"):
            self.backbone._post_init()

    def forward(
        self,
        batch: Any,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        backbone_outputs = self.backbone(batch, training=training, transform=transform)
        if not isinstance(backbone_outputs, BackboneOutputs):
            raise TypeError("Backbone must return BackboneOutputs.")

        outputs: dict[str, Any] = {
            "node_feats": backbone_outputs.node_feats,
            "graph_feats": backbone_outputs.graph_feats,
        }
        outputs.update(backbone_outputs.extras)

        accumulated_outputs = dict(outputs)

        for name, head in self.heads.items():
            head_out = head(
                backbone_outputs,
                batch,
                outputs=accumulated_outputs,
                training=training,
                transform=transform,
                **kwargs,
            )
            if not isinstance(head_out, dict):
                raise TypeError("Heads must return a dict of outputs.")
            if self.prefix_outputs:
                head_out = {f"{name}_{k}": v for k, v in head_out.items()}
            outputs.update(head_out)
            accumulated_outputs.update(head_out)

        return build_output_batch(batch, outputs)
