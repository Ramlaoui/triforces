import numpy as np
from ase import Atoms

from triforces.data.lemat_bulk import lemat_item_to_ase


def test_lemat_item_to_ase_adds_targets_and_metadata():
    item = {
        "species_at_sites": ["Si", "O"],
        "cartesian_site_positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "lattice_vectors": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        "energy": -10.0,
        "forces": [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
        "stress_tensor": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        "immutable_id": "abc",
        "nsites": 2,
        "functional": "pbe",
        "database": "lemat",
        "formula": "SiO",
        "elements": ["Si", "O"],
    }

    atoms = lemat_item_to_ase(
        item,
        add_targets=["energy", "energy_per_atom", "forces", "stress"],
        add_metadata=True,
    )

    assert isinstance(atoms, Atoms)
    assert atoms.pbc.all()
    assert atoms.info["immutable_id"] == "abc"
    assert atoms.info["nsites"] == 2
    assert atoms.info["energy"] == -10.0
    assert atoms.info["energy_per_atom"] == -5.0

    forces = atoms.arrays["forces"]
    assert forces.shape == (2, 3)
    assert np.allclose(forces[0], np.array([0.1, 0.0, 0.0], dtype=np.float32))

    stress = atoms.info["stress"]
    assert stress.shape == (6,)
    assert np.allclose(stress[:3], np.array([1.0, 2.0, 3.0], dtype=np.float32))
