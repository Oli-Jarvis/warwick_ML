"""Conformer construction used by the chemical-space descriptors."""
import hashlib
import json

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem


def prepare_dft_geometry(atomic_numbers_text, positions_text):
    """Read DFT coordinates and replace each terminal Au with an S-H cap."""
    atomic_numbers = np.asarray(json.loads(atomic_numbers_text), dtype=int)
    positions = np.asarray(json.loads(positions_text), dtype=float)

    if positions.shape != (len(atomic_numbers), 3):
        raise ValueError(
            f"positions has shape {positions.shape}; expected "
            f"({len(atomic_numbers)}, 3)"
        )

    atomic_numbers = atomic_numbers.copy()
    positions = positions.copy()
    gold_indices = np.flatnonzero(atomic_numbers == 79)
    sulfur_indices = np.flatnonzero(atomic_numbers == 16)

    for gold_index in gold_indices:
        if len(sulfur_indices) == 0:
            raise ValueError("Au atom found without a sulfur atom for capping")
        distances = np.linalg.norm(
            positions[sulfur_indices] - positions[gold_index],
            axis=1,
        )
        sulfur_index = sulfur_indices[np.argmin(distances)]
        direction = positions[gold_index] - positions[sulfur_index]
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Au and S coordinates overlap")

        atomic_numbers[gold_index] = 1
        positions[gold_index] = (
            positions[sulfur_index] + 1.34 * direction / norm
        )

    return atomic_numbers, positions


def conformer_cache_key(smiles, random_state):
    """Return a stable key for one generated conformer."""
    payload = {
        "smiles": str(smiles),
        "random_state": int(random_state),
        "method": "ETKDGv3_then_MMFF_or_UFF_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_conformer_cache(path):
    """Load previously generated SMILES/GGS conformers from an SDF file."""
    if not path.exists():
        return {}

    cached = {}
    invalid = 0
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in supplier:
        if (
            mol is None
            or mol.GetNumConformers() == 0
            or not mol.HasProp("conformer_cache_key")
        ):
            invalid += 1
            continue
        cached[mol.GetProp("conformer_cache_key")] = mol

    print(f"Loaded {len(cached):,} generated conformers from: {path}")
    if invalid:
        print(f"Warning: ignored {invalid:,} invalid conformer-cache records.")
    return cached


def save_conformer_cache(path, cached):
    """Atomically save generated conformers without discarding older entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.sdf")
    writer = Chem.SDWriter(str(temporary_path))
    try:
        for key in sorted(cached):
            writer.write(cached[key])
    finally:
        writer.close()
    temporary_path.replace(path)
    print(f"Saved {len(cached):,} generated conformers: {path}")


def make_3d_molecules(data, random_state, conformer_cache_path):
    """Build molecules/systems, reusing cached generated 3D conformers."""
    molecules = []
    systems = []
    valid_indices = []
    conformer_cache = load_conformer_cache(conformer_cache_path)
    cache_changed = False
    reused_conformers = 0
    generated_conformers = 0

    for index, row in data.iterrows():
        value = row["chemplot_smiles"]
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)

        if row["dataset_label"] == "DFT":
            try:
                atomic_numbers, positions = prepare_dft_geometry(
                    row["atomic_numbers"],
                    row["positions"],
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                print(f"Warning: invalid DFT geometry for molecule {index}: {error}")
                continue

            systems.append(Atoms(numbers=atomic_numbers, positions=positions))
            molecules.append(mol)
            valid_indices.append(index)
            continue

        cache_key = conformer_cache_key(value, random_state)
        cached_mol = conformer_cache.get(cache_key)
        if cached_mol is not None:
            mol = Chem.Mol(cached_mol)
            reused_conformers += 1
        else:
            params = AllChem.ETKDGv3()
            params.randomSeed = int(random_state)

            if AllChem.EmbedMolecule(mol, params) != 0:
                print(
                    f"Warning: conformer generation failed for molecule "
                    f"{index}; skipping."
                )
                continue

            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)

            mol.SetProp("conformer_cache_key", cache_key)
            mol.SetProp("chemplot_smiles", str(value))
            mol.SetProp("random_state", str(int(random_state)))
            conformer_cache[cache_key] = Chem.Mol(mol)
            cache_changed = True
            generated_conformers += 1

        conformer = mol.GetConformer()
        systems.append(
            Atoms(
                symbols=[atom.GetSymbol() for atom in mol.GetAtoms()],
                positions=np.asarray(conformer.GetPositions(), dtype=float),
            )
        )
        molecules.append(mol)
        valid_indices.append(index)

    if len(molecules) < 1:
        raise ValueError("No molecules produced valid 3D conformers.")

    if cache_changed or not conformer_cache_path.exists():
        save_conformer_cache(conformer_cache_path, conformer_cache)
    print(
        f"Generated conformers reused: {reused_conformers:,}; "
        f"newly generated: {generated_conformers:,}."
    )

    return molecules, systems, valid_indices
