from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import torch.nn as nn

from .base import Model
from .outputs import BackboneOutputs
from .utils import build_output_batch

logger = logging.getLogger("triforces")


class AdapterModel(Model):
    """Wrap a backbone and apply multiple heads with a strict interface.

    Parameters
    ----------
    backbone : nn.Module
        Backbone model returning ``BackboneOutputs``.
    heads : dict[str, nn.Module | Callable[..., nn.Module]], optional
        Mapping of head names to either:
        - pre-built ``nn.Module`` heads, or
        - head factories (typically Hydra partials). Factories can implement
          ``build_from_backbone_info(backbone_info: dict, **kwargs)`` to
          self-configure dimensions from the backbone contract.
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
        heads: dict[str, nn.Module | Callable[..., nn.Module]] | None = None,
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

        head_build_info = self._collect_head_build_info()
        resolved_heads: dict[str, nn.Module] = {}
        for key, head_spec in heads.items():
            head = self._build_head(key, head_spec, head_build_info)
            head.key_name = key
            if use_model_readout.get(key, False) and hasattr(head, "get_model_readout"):
                logger.info("Using model readout for head %s", key)
                head.get_model_readout(self.backbone)
            resolved_heads[key] = head
            get_head_info = getattr(head, "get_head_build_info", None)
            if callable(get_head_info):
                extra_info = get_head_info()
                if isinstance(extra_info, dict):
                    head_build_info.update(extra_info)

        self.heads = nn.ModuleDict(resolved_heads)

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

    def _build_head(
        self,
        name: str,
        head_spec: nn.Module | Callable[..., nn.Module],
        head_build_info: dict[str, object],
    ) -> nn.Module:
        if isinstance(head_spec, nn.Module):
            return head_spec
        if not callable(head_spec):
            raise TypeError(
                f"Head {name!r} must be an nn.Module or callable factory, got {type(head_spec)!r}"
            )

        target_factory = head_spec.func if isinstance(head_spec, partial) else head_spec
        preset_kwargs = (
            dict(head_spec.keywords or {}) if isinstance(head_spec, partial) else {}
        )

        build_from_info = getattr(target_factory, "build_from_backbone_info", None)
        if callable(build_from_info):
            try:
                head = build_from_info(dict(head_build_info), **preset_kwargs)
            except TypeError as exc:
                details = (
                    f"Failed to build head {name!r} using "
                    f"{target_factory.__name__}.build_from_backbone_info(...). "
                    f"backbone_info={head_build_info}, kwargs={preset_kwargs}"
                )
                raise TypeError(details) from exc
            if not isinstance(head, nn.Module):
                raise TypeError(
                    f"{target_factory.__name__}.build_from_backbone_info must return nn.Module, "
                    f"got {type(head)!r}"
                )
            return head

        try:
            head = head_spec()
        except TypeError as exc:
            details = (
                f"Failed to instantiate head {name!r} from factory {head_spec!r}. "
                f"Either provide all constructor args in the factory/partial, "
                f"or implement `build_from_backbone_info(backbone_info, **kwargs)` "
                f"on the head class."
            )
            raise TypeError(details) from exc
        if not isinstance(head, nn.Module):
            raise TypeError(
                f"Head factory {name!r} must return nn.Module, got {type(head)!r}"
            )
        return head

    def _collect_head_build_info(self) -> dict[str, object]:
        info: dict[str, object] = {}
        get_info = getattr(self.backbone, "get_head_build_info", None)
        if callable(get_info):
            collected = get_info()
            if isinstance(collected, dict):
                info.update(collected)

        if "output_dim" not in info:
            output_dim = getattr(self.backbone, "output_dim", None)
            if isinstance(output_dim, int) and output_dim > 0:
                info["output_dim"] = int(output_dim)
        if "stream_dims" not in info:
            stream_dims = getattr(self.backbone, "stream_dims", None)
            if isinstance(stream_dims, dict):
                info["stream_dims"] = dict(stream_dims)
        return info

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
