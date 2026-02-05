from typing import Callable, Optional

import numpy as np
import torch
import torch._functorch.config
import torch.nn as nn
from ase import Atoms
from mace.calculators import mace_mp
from mace.data import AtomicData, KeySpecification, config_from_atoms
from mace.tools.compile import prepare
from mace.tools.scripts_utils import extract_model
from mace.tools.utils import AtomicNumberTable
from torch_geometric.data import Batch, Data

import logging

from triforces.models.base import Model

logger = logging.getLogger(__name__)

torch._functorch.config.donated_buffer = False


class MACE(Model):
    """MACE model implementation.

    Parameters
    ----------
    model_type : str, default="small"
        Type of MACE model to use (e.g., ``"small"``, ``"medium"``, ``"large"``).
    device : str, default="cpu"
        Device to load the model on.
    default_dtype : str, default="float32"
        Default data type for model parameters.
    model_foundation : nn.Module, optional
        Pre-initialized model foundation to use.
    disable_forces : bool, default=False
        If True, disable force predictions.
    disable_stress : bool, default=False
        If True, disable stress predictions.
    targets : list[str], optional
        Targets predicted by the model.
    hook_fns : dict[str, Callable], optional
        Hook functions registered on modules.
    freeze_weights : list[str], optional
        Module paths whose parameters should be frozen.
    **kwargs : Any
        Additional arguments passed to ``mace_mp``.
    """

    def __init__(
        self,
        model_type: str = "small",
        device: str = "cpu",
        default_dtype: str = "float32",
        model_foundation: Optional[nn.Module] = None,
        disable_forces: bool = False,
        disable_stress: bool = False,
        targets: Optional[list[str]] = None,
        hook_fns: Optional[dict[str, Callable]] = None,
        freeze_weights: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            targets=targets, hook_fns=hook_fns, freeze_weights=freeze_weights
        )

        self.requires_grad_for_inference = True
        self.model_type = model_type

        self.model = model_foundation
        if self.model is None:
            # Load the config from the foundation model, but reinitialize the model
            calc = mace_mp(
                model=model_type, device=device, default_dtype=default_dtype, **kwargs
            )
            self.model = build_default_model()

        # Store readout information before `torch.fx.symbolic_trace`
        self.readout_info = {
            "irreps_in_0": self.model.readouts[0].linear.irreps_in,
            "irreps_out_0": self.model.readouts[0].linear.irreps_out,
            "irreps_in_1": self.model.readouts[1].linear_1.irreps_in,
            "irreps_out_1": self.model.readouts[1].linear_1.irreps_out,
            "irreps_in_2": self.model.readouts[1].linear_2.irreps_in,
            "irreps_out_2": self.model.readouts[1].linear_2.irreps_out,
        }

        # Disable TorchScript compilation to avoid issues with distributed training
        try:
            self.model = prepare(extract_model)(model=self.model, map_location=device)
        except Exception as e:
            logger.warning(
                f"Error removing TorchScript from model: {e}, will try again\n"
                "by removing pair_repulsion args."
            )
            del self.model.pair_repulsion_fn.r_max
            del self.model.pair_repulsion_fn.cutoff.r_max
            del self.model.pair_repulsion_fn.cutoff.p
            self.model = prepare(extract_model)(model=self.model, map_location=device)

        self.disable_forces, self.disable_stress = disable_forces, disable_stress

        self._post_init()

    def forward(
        self,
        batch: Batch,
        training: bool = True,
        skip_displacement: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_dict = batch._store._mapping

        # If skip_displacement=True, the caller has already applied displacement
        # to batch.positions and will handle force/stress computation themselves.
        # We must disable MACE's internal force/stress to avoid:
        # 1. Double-applying displacement (MACE checks compute_stress || compute_displacement)
        # 2. Consuming the autograd graph before all streams have run
        if skip_displacement:
            compute_force = False
            compute_stress = False
            compute_displacement = False
        else:
            compute_force = ("forces" in self.targets) and not self.disable_forces
            compute_stress = ("stress" in self.targets) and not self.disable_stress
            compute_displacement = (
                compute_stress  # Only need displacement if computing stress
            )

        out = self.model(
            batch_dict,
            compute_force=compute_force,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            training=training,
        )

        return out

    @classmethod
    def get_model_name(cls):
        return "MACE"

    @classmethod
    def from_pretrained(
        cls,
        model_type: str = "small",
        device: str = "cpu",
        default_dtype: str = "float32",
        enable_cueq: bool = False,
        return_calculator: bool = False,
        **kwargs,
    ):
        calc = mace_mp(
            model=model_type,
            device=device,
            default_dtype=default_dtype,
            enable_cueq=enable_cueq,
        )

        if return_calculator:
            return calc

        model = calc.models[0]
        # Set all parameters to be trainable
        for param in model.parameters():
            param.requires_grad = True

        return cls(model_foundation=model, model_type=model_type, **kwargs)

    def get_backbone_outputs(
        self, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        backbone_outputs = super().get_backbone_outputs(outputs)
        return {
            "node_feats": outputs["node_feats"],
            **backbone_outputs,
        }

    def get_readout(self) -> tuple[nn.Module, int, Optional[nn.Module]]:
        # This is the dimension of the backbone embeddings
        irreps_out = (
            self.readout_info["irreps_in_0"] + self.readout_info["irreps_in_1"]
        ).simplify()

        # When using TorchScript, we could do the following:

        # from e3nn import o3
        # readout = deepcopy(self.model.readouts[1])
        # readout.linear_1 = o3.Linear(irreps_out, self.readout_info["irreps_out_1"])

        # return None, list(self.readout_info["irreps_out_2"])[0].mul, None

        return None, irreps_out, None

    def get_log_params(self) -> dict:
        base_info = super().get_log_params()
        mace_info = {
            "model_type": self.model_type,
        }
        return {**base_info, **mace_info}


CLOSE_ATOM_THRESHOLD = 1e-6
NOISE_SCALE_ANGSTROM = 1e-6


def _stress_to_voigt_6(stress: torch.Tensor | None) -> torch.Tensor | None:
    if stress is None:
        return None

    if stress.shape[-1] == 6:
        return stress

    if stress.shape[-1] == 9:
        stress = stress.reshape(-1, 3, 3)

    if stress.shape[-2:] != (3, 3):
        raise ValueError(
            "Input stress tensor must have shape (..., 3, 3) or (..., 6). "
            f"Got shape {stress.shape}"
        )

    batch_shape = stress.shape[:-2]
    voigt = torch.empty((*batch_shape, 6), dtype=stress.dtype, device=stress.device)

    voigt[..., 0] = stress[..., 0, 0]
    voigt[..., 1] = stress[..., 1, 1]
    voigt[..., 2] = stress[..., 2, 2]
    voigt[..., 3] = (stress[..., 1, 2] + stress[..., 2, 1]) * 0.5
    voigt[..., 4] = (stress[..., 2, 0] + stress[..., 0, 2]) * 0.5
    voigt[..., 5] = (stress[..., 0, 1] + stress[..., 1, 0]) * 0.5

    return voigt


class MACEGraph:
    """Build a PyG graph for MACE from ASE ``Atoms``.

    Parameters
    ----------
    model : nn.Module, optional
        Optional MACE model to infer atomic number table and cutoff.
    charges_key : str, default="Qs"
        ASE array key to use for atomic charges.
    r_max : float, default=6.0
        Cutoff radius for graph construction.
    """

    def __init__(
        self,
        model=None,
        charges_key: str = "Qs",
        r_max: float = 6.0,
    ):
        self.charges_key = charges_key
        self.keyspec = KeySpecification(
            info_keys={}, arrays_keys={"charges": self.charges_key}
        )
        self.r_max = float(r_max)
        self.reload_model(model)
        self.converter_attrs = [
            "edge_index",
            "charges",
            "stress_weight",
            "unit_shifts",
            "virial",
            "virial_weights",
            "weight",
            "energy_weight",
            "forces_weight",
            "dipole",
            "shifts",
            "positions",
            "head",
            "node_attrs",
        ]

    def reload_model(self, model=None):
        if model is None:
            atomic_numbers = np.array(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    71,
                    72,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    83,
                    89,
                    90,
                    91,
                    92,
                    93,
                    94,
                ]
            )
        else:
            atomic_numbers = model.model.atomic_numbers
        self.z_table = AtomicNumberTable([int(z) for z in atomic_numbers])

        if model is None:
            r_maxs = [self.r_max]
        else:
            r_maxs = [model.model.r_max.cpu()]
        r_maxs = np.array(r_maxs)
        if not np.all(r_maxs == r_maxs[0]):
            raise ValueError(f"committee r_max are not all the same {' '.join(r_maxs)}")
        self.r_max = float(r_maxs[0])

    def __call__(self, atoms: Atoms, **kwargs) -> Data:
        config = config_from_atoms(atoms, key_specification=self.keyspec)
        atomic_data = AtomicData.from_config(
            config, z_table=self.z_table, cutoff=self.r_max, heads=["Default"]
        )
        data = Data(**{**atomic_data.__dict__, **kwargs})

        lengths = torch.linalg.norm(
            data["positions"][data["edge_index"][0, :]]
            - data["positions"][data["edge_index"][1, :]],
            dim=1,
        )
        different_sites = data["edge_index"][0, :] != data["edge_index"][1, :]
        same_positions_mask = (lengths < CLOSE_ATOM_THRESHOLD) & different_sites
        if torch.any(same_positions_mask):
            nodes_close = data["edge_index"][:, same_positions_mask]
            data["positions"][nodes_close[0, :]] += (
                torch.randn_like(data["positions"][nodes_close[0, :]])
                * NOISE_SCALE_ANGSTROM
            )

        if hasattr(data, "stress"):
            data.stress = _stress_to_voigt_6(data.stress)
        data.atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        data.natoms = len(atoms)

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, "
            f"charges_key={self.charges_key})"
        )


def mace_graph(
    *,
    r_max: float = 6.0,
    charges_key: str = "Qs",
):
    """Return the MACE graph builder used for collation.

    Parameters
    ----------
    r_max : float, default=6.0
        Cutoff radius for graph construction.
    charges_key : str, default="Qs"
        ASE array key to use for atomic charges.

    Returns
    -------
    MACEGraph
        Graph builder instance.
    """
    return MACEGraph(r_max=r_max, charges_key=charges_key)
