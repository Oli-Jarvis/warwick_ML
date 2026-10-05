#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdDetermineBonds
from rdkit.ML.Descriptors import MoleculeDescriptors
from ase import Atoms
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", type=Path, default=Path("."))
    p.add_argument("--smiles-root", type=Path, default=Path("smiles_baseline2"))
    p.add_argument("--ggs-root", type=Path, default=Path("ggs_2"))
    p.add_argument(
        "--dft-file",
        type=Path,
        default=Path("GA_best_candidates_full_history.csv"),
    )
    p.add_argument("--run-pattern", default="run_0_seed_*")
    p.add_argument("--input-name", default="full_history.csv")
    p.add_argument("--delimiter", default="auto")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--drop-duplicate-smiles", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("dim_plots"))
    p.add_argument(
        "--pca-components",
        type=int,
        help="Number of bonding and structural PCA components to save.",
    )
    p.add_argument(
        "--morgan-radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius used for PCA. Default: 2.",
    )
    p.add_argument(
        "--morgan-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint length used for PCA. Default: 2048.",
    )
    return p.parse_args()


def load_data(root, pattern, filename, delimiter, dataset_label):
    frames = []
    sep = None if delimiter.lower() == "auto" else delimiter

    for run in sorted(p for p in root.glob(pattern) if p.is_dir()):
        path = run / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="warn")
        df.columns = df.columns.astype(str).str.strip()
        df["source_run"] = run.name
        df["source_row"] = np.arange(len(df))
        df["dataset_label"] = dataset_label
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No {filename} files found in {root}")

    data = pd.concat(frames, ignore_index=True, sort=False)
    print(
        f"Combined {len(data):,} {dataset_label} rows "
        f"from {len(frames)} runs."
    )
    return data


def find_column(data, requested, alternatives=()):
    lookup = {str(c).strip().lower(): c for c in data.columns}
    for name in (requested, *alternatives):
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise KeyError(
        f"Missing column: {requested}. Available columns: "
        f"{', '.join(map(str, data.columns))}"
    )


def prepare_data(data, smiles_col, drop_duplicates):
    smiles_source = find_column(
        data, smiles_col, ("SMILES", "canonical_smiles", "canonical smiles", "encoding")
    )
    data = data.copy()
    if smiles_source != smiles_col:
        data[smiles_col] = data[smiles_source]
    data = data.dropna(subset=[smiles_col])
    data["chemplot_smiles"] = (
        data[smiles_col]
        .astype(str)
        .str.strip()
        .str.replace("[Au]", "", regex=False)
    )
    data = data[
        data["chemplot_smiles"].map(
            lambda s: bool(s) and Chem.MolFromSmiles(s) is not None
        )
    ]

    if drop_duplicates:
        data = data.drop_duplicates("chemplot_smiles")

    data = data.reset_index(drop=True)
    if len(data) < 5:
        raise ValueError("At least five valid molecules are required.")

    print(f"Usable molecules: {len(data):,}")
    return data


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


def infer_smiles_from_geometry(atomic_numbers, positions):
    """Infer connectivity and bond orders from a neutral 3D structure."""
    last_error = None
    for use_vdw, allow_charged_fragments in (
        (False, True),
        (True, True),
        (False, False),
        (True, False),
    ):
        editable = Chem.RWMol()
        for atomic_number in atomic_numbers:
            atom = Chem.Atom(int(atomic_number))
            atom.SetNoImplicit(True)
            editable.AddAtom(atom)

        molecule = editable.GetMol()
        conformer = Chem.Conformer(len(atomic_numbers))
        conformer.Set3D(True)
        for index, (x, y, z) in enumerate(positions):
            conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
        molecule.AddConformer(conformer)

        try:
            rdDetermineBonds.DetermineBonds(
                molecule,
                charge=0,
                allowChargedFragments=allow_charged_fragments,
                embedChiral=False,
                useVdw=use_vdw,
            )
            Chem.SanitizeMol(molecule)
            return Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True)
        except Exception as error:
            last_error = error

    raise ValueError(f"RDKit bond reconstruction failed: {last_error}")


def load_dft_data(path, delimiter, reference_data, smiles_col):
    """Load DFT geometries and recover SMILES by matching molecule hashes."""
    sep = None if delimiter.lower() == "auto" else delimiter
    data = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="warn")
    data.columns = data.columns.astype(str).str.strip()

    dft_hash_col = find_column(data, "hash_values", ("molcode",))
    reference_hash_col = find_column(
        reference_data,
        "hash_values",
        ("hash", "molcode"),
    )

    reference = reference_data[[reference_hash_col, "chemplot_smiles"]].copy()
    reference["molecule_hash"] = (
        reference[reference_hash_col]
        .astype(str)
        .str.strip()
        .str.replace(r"_aligned\d*$", "", regex=True)
    )
    conflicting = reference.groupby("molecule_hash")["chemplot_smiles"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        raise ValueError(
            "Some molecule hashes map to more than one SMILES string: "
            + ", ".join(conflicting.index[:5])
        )

    smiles_by_hash = (
        reference.drop_duplicates("molecule_hash")
        .set_index("molecule_hash")["chemplot_smiles"]
    )
    data["molecule_hash"] = (
        data[dft_hash_col]
        .astype(str)
        .str.strip()
        .str.replace(r"_aligned\d*$", "", regex=True)
    )
    data["chemplot_smiles"] = data["molecule_hash"].map(smiles_by_hash)

    for column in ("atomic_numbers", "positions"):
        if column not in data.columns:
            raise KeyError(f"The DFT file is missing required column: {column}")

    missing = data["chemplot_smiles"].isna()
    print(
        f"Recovered {(~missing).sum():,} DFT SMILES by matching molecule hashes."
    )
    if missing.any():
        print(
            f"Recovering {missing.sum()} DFT bonding structures from "
            "their atomic coordinates."
        )
        failures = []
        for index in data.index[missing]:
            try:
                atomic_numbers, positions = prepare_dft_geometry(
                    data.at[index, "atomic_numbers"],
                    data.at[index, "positions"],
                )
                data.at[index, "chemplot_smiles"] = infer_smiles_from_geometry(
                    atomic_numbers,
                    positions,
                )
            except Exception as error:
                failures.append((data.at[index, "molecule_hash"], str(error)))

        if failures:
            examples = "; ".join(
                f"{molecule_hash}: {error}"
                for molecule_hash, error in failures[:5]
            )
            raise ValueError(
                f"Could not reconstruct {len(failures)} DFT molecules from "
                f"their coordinates. Examples: {examples}"
            )

    data[smiles_col] = data["chemplot_smiles"]

    data["source_run"] = "DFT"
    data["source_row"] = np.arange(len(data))
    data["dataset_label"] = "DFT"
    data = data.reset_index(drop=True)
    print(f"Loaded {len(data):,} DFT molecules with supplied 3D coordinates.")
    return data


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


def make_soap_matrix(systems):
    """Return averaged SOAP descriptors for generated or supplied conformers."""
    species = set()

    for system in systems:
        species.update(system.get_chemical_symbols())

    soap = SOAP(
        species=sorted(species),
        periodic=False,
        r_cut=5.0,
        n_max=4,
        l_max=3,
        average="inner",
        sparse=False,
    )
    return np.asarray(soap.create(systems), dtype=np.float32)


def make_bonding_matrix(molecules):
    """Return RDKit's complete numerical molecular descriptor set."""
    descriptor_names = [name for name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    matrix = np.asarray(
        [calculator.CalcDescriptors(Chem.RemoveHs(mol)) for mol in molecules],
        dtype=float,
    )

    matrix[~np.isfinite(matrix)] = np.nan
    medians = np.nanmedian(matrix, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    missing_rows, missing_columns = np.where(np.isnan(matrix))
    matrix[missing_rows, missing_columns] = medians[missing_columns]

    nonconstant = np.nanstd(matrix, axis=0) > 0
    return matrix[:, nonconstant], np.asarray(descriptor_names)[nonconstant]


def embed(
    data,
    random_state,
    pca_components,
    morgan_radius,
    morgan_bits,
    output_dir,
):
    # Cache the expensive, method-independent descriptor generation so that
    # repeated PCA runs only need to refit PCA.
    cache_dir = output_dir / "descriptor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    representation_records = []
    for _, row in data.iterrows():
        record = [row["dataset_label"], str(row["chemplot_smiles"])]
        if row["dataset_label"] == "DFT":
            record.extend([str(row["atomic_numbers"]), str(row["positions"])])
        representation_records.append(record)
    representation_hash = hashlib.sha256(
        json.dumps(representation_records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_metadata_path = cache_dir / "metadata.json"
    structural_path = cache_dir / "soap_structural_matrix.npy"
    bonding_path = cache_dir / "bonding_descriptor_matrix.npy"
    bonding_names_path = cache_dir / "bonding_descriptor_names.npy"
    valid_indices_path = cache_dir / "valid_indices.npy"
    conformer_cache_path = output_dir / "generated_conformers.sdf"

    expected_metadata = {
        "representations_sha256": representation_hash,
        "n_input_molecules": len(data),
        "random_state": int(random_state),
        "soap_r_cut": 5.0,
        "soap_n_max": 4,
        "soap_l_max": 3,
        "soap_average": "inner",
        "dft_gold_cap": "Au replaced by H at 1.34 angstrom from nearest S",
        "conformer_cache_version": 1,
    }

    cache_files = [
        cache_metadata_path,
        structural_path,
        bonding_path,
        bonding_names_path,
        valid_indices_path,
        conformer_cache_path,
    ]
    use_cache = all(path.exists() for path in cache_files)

    if use_cache:
        try:
            cached_metadata = json.loads(cache_metadata_path.read_text())
            use_cache = cached_metadata == expected_metadata
        except (OSError, json.JSONDecodeError):
            use_cache = False

    if use_cache:
        print("Loading cached SOAP and bonding descriptors.")
        valid_indices = np.load(valid_indices_path).astype(int).tolist()
        structural_matrix = np.load(structural_path)
        bonding_matrix = np.load(bonding_path)
        bonding_descriptor_names = np.load(
            bonding_names_path, allow_pickle=False
        ).astype(str)
        data = data.iloc[valid_indices].reset_index(drop=True)
        print(f"Cached valid 3D conformers: {len(data):,}")
    else:
        print(
            "Generating conformers for SMILES/GGS and loading supplied "
            "DFT coordinates."
        )
        molecules, systems, valid_indices = make_3d_molecules(
            data,
            random_state=random_state,
            conformer_cache_path=conformer_cache_path,
        )
        data = data.iloc[valid_indices].reset_index(drop=True)
        print(f"Valid 3D conformers: {len(data):,}")

        print("Calculating averaged SOAP structural descriptors.")
        structural_matrix = make_soap_matrix(systems)

        print("Calculating RDKit bonding descriptors.")
        bonding_matrix, bonding_descriptor_names = make_bonding_matrix(molecules)

        np.save(structural_path, structural_matrix)
        np.save(bonding_path, bonding_matrix)
        np.save(bonding_names_path, np.asarray(bonding_descriptor_names, dtype=str))
        np.save(valid_indices_path, np.asarray(valid_indices, dtype=int))
        cache_metadata_path.write_text(
            json.dumps(expected_metadata, indent=2), encoding="utf-8"
        )
        print(f"Saved descriptor cache: {cache_dir}")

    maximum_components = min(
        structural_matrix.shape[0],
        structural_matrix.shape[1],
        bonding_matrix.shape[1],
    )
    actual_components = min(pca_components, maximum_components)

    if actual_components < 1:
        raise ValueError("At least one PCA component is required.")

    if actual_components != pca_components:
        print(
            f"Requested {pca_components} PCA components, but the data permit "
            f"only {actual_components}. Using {actual_components}."
        )

    structural_scaled = StandardScaler().fit_transform(structural_matrix)
    bonding_scaled = StandardScaler().fit_transform(bonding_matrix)

    structural_pca = PCA(
        n_components=actual_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    bonding_pca = PCA(
        n_components=actual_components,
        svd_solver="randomized",
        random_state=random_state,
    )

    structural_coordinates = structural_pca.fit_transform(structural_scaled)
    bonding_coordinates = bonding_pca.fit_transform(bonding_scaled)

    embedded = pd.DataFrame(index=np.arange(len(data)))
    for component in range(actual_components):
        number = component + 1
        embedded[f"Bonding_PC_{number}"] = bonding_coordinates[:, component]
        embedded[f"Structural_PC_{number}"] = structural_coordinates[:, component]

    # Keep the existing plotting/output code unchanged: PCA_1 is the most
    # significant bonding component and PCA_2 is the most significant
    # structural component, matching the axes used in the paper.
    embedded["PCA_1"] = embedded["Bonding_PC_1"]
    embedded["PCA_2"] = embedded["Structural_PC_1"]

    print(f"PCA components saved for each descriptor: {actual_components}")
    print(
        "Cumulative explained variance (bonding): "
        f"{bonding_pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )
    print(
        "Cumulative explained variance (structural): "
        f"{structural_pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )

    result = pd.concat(
        [data.reset_index(drop=True), embedded.reset_index(drop=True)],
        axis=1,
    )
    pca_fit = {
        "bonding": bonding_pca,
        "structural": structural_pca,
        "bonding_descriptor_names": bonding_descriptor_names,
    }
    return result, pca_fit


def save_pca_variance(pca_fit, output_dir):
    for descriptor_type in ("bonding", "structural"):
        fit = pca_fit[descriptor_type]
        ratios = fit.explained_variance_ratio_
        cumulative = np.cumsum(ratios)
        components = np.arange(1, len(ratios) + 1)

        variance_df = pd.DataFrame({
            "PCA_component": components,
            "explained_variance": fit.explained_variance_,
            "explained_variance_ratio": ratios,
            "explained_variance_percent": ratios * 100,
            "cumulative_explained_variance_ratio": cumulative,
            "cumulative_explained_variance_percent": cumulative * 100,
        })

        variance_df.to_csv(
            output_dir / f"{descriptor_type}_pca_explained_variance.csv",
            index=False,
        )

        plt.figure(figsize=(10, 6))
        plt.bar(components, ratios * 100)
        plt.xlabel("PCA component")
        plt.ylabel("Explained variance (%)")
        plt.title(f"{descriptor_type.capitalize()} PCA explained variance")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{descriptor_type}_pca_individual_explained_variance.png",
            dpi=300,
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(components, cumulative * 100, marker="o", markersize=3)
        plt.xlabel("Number of PCA components")
        plt.ylabel("Cumulative explained variance (%)")
        plt.title(f"Cumulative {descriptor_type} PCA explained variance")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{descriptor_type}_pca_cumulative_explained_variance.png",
            dpi=300,
        )
        plt.close()

    pd.DataFrame({
        "bonding_descriptor": pca_fit["bonding_descriptor_names"]
    }).to_csv(output_dir / "bonding_descriptor_names.csv", index=False)

    print("Saved separate bonding and structural PCA variance files.")


def save_outputs(
    data,
    smiles_col,
    output_dir,
    pca_fit=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = "PCA_1", "PCA_2"
    x_label, y_label = "Bonding PC1", "Structural PC1"
    stem = output_dir / "combined_labelled_pca_embedding"

    data.to_csv(stem.with_suffix(".csv"), index=False)

    hover = [
        c for c in [
            smiles_col,
            "source_run",
            "source_row",
            "dataset_label",
            "molcode",
        ]
        if c in data
    ]
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color="dataset_label",
        category_orders={"dataset_label": ["smiles", "ggs", "DFT"]},
        color_discrete_map={"smiles": "#1f77b4", "ggs": "#ff7f0e", "DFT": "#2ca02c"},
        hover_data=hover,
        render_mode="webgl",
        title="Combined molecular chemical space: PCA",
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(template="plotly_white", xaxis_title=x_label, yaxis_title=y_label)
    fig.write_html(stem.with_suffix(".html"), include_plotlyjs="cdn")

    plt.figure(figsize=(10, 8))
    colours = {"smiles": "#1f77b4", "ggs": "#ff7f0e", "DFT": "#2ca02c"}
    for label in ("smiles", "ggs", "DFT"):
        group = data[data["dataset_label"] == label]
        plt.scatter(
            group[x],
            group[y],
            c=colours[label],
            label=label,
            s=8,
            alpha=0.75,
            linewidths=0,
        )
    plt.legend(title="Dataset")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Combined molecular chemical space: PCA")
    plt.tight_layout()
    plt.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close()

    print(f"Saved: {stem}.csv, .html, .png")

    if pca_fit is not None:
        save_pca_variance(pca_fit, output_dir)


def main():
    args = get_args()
    root = args.root.expanduser().resolve()

    if args.pca_components is None:
        while True:
            try:
                pca_components = int(
                    input("Enter the number of PCA components: ").strip()
                )
                if pca_components >= 1:
                    break
                print("Please enter an integer of at least 1.")
            except ValueError:
                print("Please enter a valid integer.")
    else:
        pca_components = args.pca_components

    if pca_components < 1:
        raise ValueError("--pca-components must be at least 1.")

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )

    smiles_root = args.smiles_root if args.smiles_root.is_absolute() else root / args.smiles_root
    ggs_root = args.ggs_root if args.ggs_root.is_absolute() else root / args.ggs_root
    dft_file = args.dft_file if args.dft_file.is_absolute() else root / args.dft_file

    smiles_data = load_data(
        smiles_root,
        args.run_pattern,
        args.input_name,
        args.delimiter,
        "smiles",
    )
    smiles_data = prepare_data(
        smiles_data,
        args.smiles_column,
        args.drop_duplicate_smiles,
    )
    ggs_data = load_data(
        ggs_root,
        args.run_pattern,
        args.input_name,
        args.delimiter,
        "ggs",
    )
    ggs_data = prepare_data(
        ggs_data,
        args.smiles_column,
        args.drop_duplicate_smiles,
    )

    reference_data = pd.concat([smiles_data, ggs_data], ignore_index=True, sort=False)
    dft_data = load_dft_data(
        dft_file,
        args.delimiter,
        reference_data,
        args.smiles_column,
    )
    data = pd.concat([reference_data, dft_data], ignore_index=True, sort=False)
    print("Combined dataset counts:")
    print(data["dataset_label"].value_counts(sort=False).to_string())

    data, pca_fit = embed(
        data,
        args.random_state,
        pca_components,
        args.morgan_radius,
        args.morgan_bits,
        output_dir,
    )
    save_outputs(
        data,
        args.smiles_column,
        output_dir,
        pca_fit,
    )


