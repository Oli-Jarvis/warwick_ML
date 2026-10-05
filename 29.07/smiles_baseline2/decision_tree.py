#!/usr/bin/env python3
"""
Run one fixed decision-tree configuration on full_history.csv using hashed
Morgan fingerprints.

Fixed hyperparameters:
    radius=2
    nbits=2048
    max_depth=20
    min_samples_split=5
    min_samples_leaf=1
    criterion="gini"

Class definitions are rank-based: the requested top or bottom percentage\nof molecules is assigned to target class 1 for each workflow.

The data is split into:
    80% training
    20% untouched test

Example:
    # Run on every run_*/full_history.csv inside the current directory:
    python decision_tree.py

    # Or specify another parent directory:
    python decision_tree.py /path/to/dataset_directory
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import DrawMorganBit, rdMolDraw2D
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


# ---------------------------------------------------------------------------
# Fixed model configuration
# ---------------------------------------------------------------------------

RADIUS = 2
NBITS = 2048
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 1
CRITERION = "gini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and test one fixed decision-tree configuration using "
            "Morgan fingerprints."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=Path("."),
        help=(
            "Directory containing run_*/full_history.csv files, or one "
            "individual CSV/Parquet file. Defaults to the current directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("decision_tree_single_configuration_results"),
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--fitness-column", default="fitness")
    parser.add_argument(
        "--logp-column",
        default="log_P_upconversion",
        help="Column used as the target for the high- and low-log-P workflows.",
    )
    parser.add_argument(
        "--high-fitness-percent",
        type=float,
        default=10.0,
        help=(
            "Percentage of molecules with the highest fitness assigned to "
            "target class 1."
        ),
    )
    parser.add_argument(
        "--low-fitness-percent",
        type=float,
        default=10.0,
        help=(
            "Percentage of molecules with the lowest fitness assigned to "
            "target class 1."
        ),
    )
    parser.add_argument(
        "--high-logp-percent",
        type=float,
        default=10.0,
        help=(
            "Percentage of molecules with the highest log P assigned to "
            "target class 1."
        ),
    )
    parser.add_argument(
        "--low-logp-percent",
        type=float,
        default=10.0,
        help=(
            "Percentage of molecules with the lowest log P assigned to "
            "target class 1."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=1)
    parser.add_argument("--max-fragment-images", type=int, default=12)
    parser.add_argument(
        "--remove-gold",
        action="store_true",
        help="Remove [Au] tokens before fingerprint generation.",
    )
    parser.add_argument(
        "--deduplicate-smiles",
        action="store_true",
        help="Keep one row per canonical SMILES, retaining the highest fitness.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix == ".csv":
        # NMO history files may use a .csv extension while being separated by
        # commas, semicolons, or tabs. Detect the separator from the header.
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline()

        delimiter_counts = {
            ",": header.count(","),
            ";": header.count(";"),
            "\t": header.count("\t"),
        }
        delimiter = max(delimiter_counts, key=delimiter_counts.get)

        if delimiter_counts[delimiter] == 0:
            raise ValueError(
                "Could not detect a comma, semicolon, or tab delimiter in "
                f"the first line of {path}."
            )

        print(f"Detected delimiter: {repr(delimiter)}")
        table = pd.read_csv(
            path,
            sep=delimiter,
            low_memory=False,
            encoding="utf-8-sig",
        )
        table.columns = [str(column).strip() for column in table.columns]
        return table

    raise ValueError("Input file must be CSV or Parquet.")


def find_input_files(input_path: Path) -> list[Path]:
    """Find all run_*/full_history.csv files, or accept one input file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    input_files = sorted(input_path.glob("run_*/full_history.csv"))

    if not input_files:
        input_files = sorted(input_path.rglob("full_history.csv"))

    if not input_files:
        raise FileNotFoundError(
            f"No full_history.csv files found inside: {input_path}"
        )

    return input_files


def load_all_tables(input_path: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Load and concatenate every discovered history file."""
    input_files = find_input_files(input_path)

    print(f"Found {len(input_files):,} input file(s):")
    tables: list[pd.DataFrame] = []

    for path in input_files:
        print(f"  {path}")
        table = load_table(path)
        table["source_file"] = str(path)
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True, sort=False)
    print(f"Combined rows before cleaning: {len(combined):,}")
    return combined, input_files


def clean_smiles_value(value: Any) -> str | None:
    """Return a clean SMILES string, or None for missing/non-text values."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None

    return text


def prepare_data(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = {
        args.smiles_column,
        args.fitness_column,
        args.logp_column,
    }
    missing = required.difference(raw.columns)

    if missing:
        available = ", ".join(map(str, raw.columns))
        raise KeyError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns are: {available}"
        )

    data = raw[
        [
            args.smiles_column,
            args.fitness_column,
            args.logp_column,
        ]
    ].copy()
    data.columns = ["original_smiles", "fitness", "log_p"]
    rows_read = len(data)

    data["fitness"] = pd.to_numeric(data["fitness"], errors="coerce")
    data["log_p"] = pd.to_numeric(data["log_p"], errors="coerce")

    finite_fitness = np.isfinite(
        data["fitness"].to_numpy(dtype=float, na_value=np.nan)
    )
    finite_log_p = np.isfinite(
        data["log_p"].to_numpy(dtype=float, na_value=np.nan)
    )
    valid_targets = finite_fitness & finite_log_p
    invalid_fitness_count = int((~finite_fitness).sum())
    invalid_log_p_count = int((~finite_log_p).sum())
    data = data.loc[valid_targets].copy()

    data["processed_smiles"] = data["original_smiles"].map(clean_smiles_value)
    missing_smiles_count = int(data["processed_smiles"].isna().sum())
    data = data.dropna(subset=["processed_smiles"]).copy()

    if args.remove_gold:
        data["processed_smiles"] = (
            data["processed_smiles"]
            .str.replace("[Au]", "", regex=False)
            .str.strip()
        )
        emptied = data["processed_smiles"].eq("")
        missing_smiles_count += int(emptied.sum())
        data = data.loc[~emptied].copy()

    valid_indices: list[Any] = []
    molecules: list[Chem.Mol] = []
    canonical_smiles: list[str] = []
    invalid_smiles_examples: list[str] = []

    for index, smiles in data["processed_smiles"].items():
        if not isinstance(smiles, str):
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            mol = None

        if mol is None:
            if len(invalid_smiles_examples) < 10:
                invalid_smiles_examples.append(smiles)
            continue

        valid_indices.append(index)
        molecules.append(mol)
        canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))

    invalid_rdkit_count = len(data) - len(valid_indices)
    data = data.loc[valid_indices].copy()
    data["mol"] = molecules
    data["canonical_smiles"] = canonical_smiles

    print(f"Rows read: {rows_read:,}")
    print(f"Rows removed for invalid/missing fitness: {invalid_fitness_count:,}")
    print(f"Rows removed for invalid/missing log P: {invalid_log_p_count:,}")
    print(f"Rows removed for missing/non-text SMILES: {missing_smiles_count:,}")
    print(
        "Rows removed because RDKit could not parse SMILES: "
        f"{invalid_rdkit_count:,}"
    )

    if invalid_smiles_examples:
        print("Examples of unparseable SMILES:")
        for example in invalid_smiles_examples:
            print(f"  {example}")

    if args.deduplicate_smiles:
        before = len(data)
        data = (
            data.sort_values("fitness", ascending=False)
            .drop_duplicates("canonical_smiles", keep="first")
            .reset_index(drop=True)
        )
        print(f"Duplicate molecules removed: {before - len(data):,}")
    else:
        data = data.reset_index(drop=True)

    if len(data) < 10:
        raise ValueError(
            f"Only {len(data)} valid molecules remain; this is too few for training."
        )

    return data


def make_labels(
    values: pd.Series,
    percentage: float,
    direction: str,
) -> tuple[np.ndarray, float, float]:
    """Label exactly the highest or lowest requested percentage as class 1."""
    if not math.isfinite(percentage) or not 0 < percentage < 100:
        raise ValueError("Each target percentage must be greater than 0 and less than 100.")

    numeric_values = values.to_numpy(dtype=float)
    number_of_rows = len(numeric_values)
    number_target = int(round(number_of_rows * percentage / 100.0))
    number_target = max(2, number_target)

    if number_target > number_of_rows - 2:
        raise ValueError(
            f"{percentage:g}% leaves fewer than 2 molecules in the non-target "
            "class. Choose a smaller percentage."
        )

    # Stable sorting makes the choice deterministic when values are tied at
    # the boundary. Exactly number_target molecules are assigned to class 1.
    sorted_indices = np.argsort(numeric_values, kind="mergesort")

    if direction == "high":
        target_indices = sorted_indices[-number_target:]
        boundary_value = float(numeric_values[target_indices].min())
    elif direction == "low":
        target_indices = sorted_indices[:number_target]
        boundary_value = float(numeric_values[target_indices].max())
    else:
        raise ValueError(f"Unknown workflow direction: {direction}")

    y = np.zeros(number_of_rows, dtype=np.int8)
    y[target_indices] = 1

    actual_percentage = 100.0 * number_target / number_of_rows
    return y, actual_percentage, boundary_value


def make_fingerprints(
    molecules: list[Chem.Mol],
) -> tuple[np.ndarray, list[dict[int, list[tuple[int, int]]]]]:
    x = np.zeros((len(molecules), NBITS), dtype=np.int32)
    bit_infos: list[dict[int, list[tuple[int, int]]]] = []

    for row_number, mol in enumerate(molecules):
        bit_info: dict[int, list[tuple[int, int]]] = {}
        fingerprint = rdMolDescriptors.GetHashedMorganFingerprint(
            mol,
            RADIUS,
            nBits=NBITS,
            useFeatures=False,
            bitInfo=bit_info,
        )

        array = np.zeros((NBITS,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        x[row_number] = array
        bit_infos.append(bit_info)

    return x, bit_infos


def calculate_metrics(
    model: DecisionTreeClassifier,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    prediction = model.predict(x)
    probability = model.predict_proba(x)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
    }

    metrics["roc_auc"] = (
        float(roc_auc_score(y, probability))
        if len(np.unique(y)) == 2
        else float("nan")
    )

    return metrics


def save_fragments(
    feature_table: pd.DataFrame,
    data: pd.DataFrame,
    fingerprints: np.ndarray,
    bit_infos: list[dict[int, list[tuple[int, int]]]],
    output_dir: Path,
    max_images: int,
    workflow_name: str,
    class_rule: str,
    target_label: str,
    representative_rng: np.random.Generator,
    used_representative_indices: set[int],
) -> None:
    '''Save all important Morgan-bit visualisations in one self-contained HTML file.'''
    import base64
    import html
    from io import BytesIO

    html_cards: list[str] = []
    selected_features = feature_table.head(max_images).copy()

    # Use the same bond length for every Morgan environment so that fragment
    # drawings are visually comparable instead of being independently scaled
    # to fill the available canvas.
    fragment_draw_options = rdMolDraw2D.MolDrawOptions()
    fragment_draw_options.fixedBondLength = 28.0
    fragment_size = (360, 300)
    maximum_importance = (
        float(selected_features["importance"].max())
        if not selected_features.empty
        else 1.0
    )

    for _, row in selected_features.iterrows():
        bit = int(row["bit"])
        candidates = [index for index, info in enumerate(bit_infos) if bit in info]
        if not candidates:
            continue

        unused_candidates = [
            index for index in candidates
            if index not in used_representative_indices
        ]

        if unused_candidates:
            molecule_index = int(representative_rng.choice(unused_candidates))
        else:
            # Reuse is allowed only when every molecule containing this bit has
            # already been used as a representative elsewhere in this run.
            molecule_index = int(representative_rng.choice(candidates))

        used_representative_indices.add(molecule_index)
        molecule = data.iloc[molecule_index]["mol"]
        rank = int(row["rank"])
        importance = float(row["importance"])
        molecules_containing_bit = int(row["molecules_containing_bit"])
        percentage_of_dataset = float(row["percentage_of_dataset"])
        mean_target_value = float(row["mean_target_value"])
        sd_target_value = float(row["sd_target_value"])

        try:
            svg_text = str(
                DrawMorganBit(
                    molecule,
                    bit,
                    bit_infos[molecule_index],
                    useSVG=True,
                    molSize=fragment_size,
                    drawOptions=fragment_draw_options,
                )
            )
            fragment_visual = f'<div class="fragment-image">{svg_text}</div>'
        except Exception as svg_error:
            try:
                image = DrawMorganBit(
                    molecule,
                    bit,
                    bit_infos[molecule_index],
                    useSVG=False,
                    molSize=fragment_size,
                    drawOptions=fragment_draw_options,
                )
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                encoded_png = base64.b64encode(buffer.getvalue()).decode("ascii")
                fragment_visual = (
                    '<div class="fragment-image">'
                    f'<img src="data:image/png;base64,{encoded_png}" '
                    f'alt="Highlighted environment for Morgan bit {bit}">'
                    '</div>'
                )
            except Exception as png_error:
                error_text = html.escape(f"SVG: {svg_error}; PNG: {png_error}")
                fragment_visual = (
                    '<div class="fragment-error">Fragment image unavailable.<br>'
                    f'{error_text}</div>'
                )

        try:
            atom_index, environment_radius = bit_infos[molecule_index][bit][0]
            environment_bonds = list(
                Chem.FindAtomEnvironmentOfRadiusN(
                    molecule,
                    environment_radius,
                    atom_index,
                )
            )
            environment_atoms = {atom_index}
            for bond_index in environment_bonds:
                bond = molecule.GetBondWithIdx(int(bond_index))
                environment_atoms.add(bond.GetBeginAtomIdx())
                environment_atoms.add(bond.GetEndAtomIdx())

            representative_drawer = rdMolDraw2D.MolDraw2DSVG(*fragment_size)
            representative_options = representative_drawer.drawOptions()
            representative_options.fixedBondLength = 28.0
            rdMolDraw2D.PrepareAndDrawMolecule(
                representative_drawer,
                molecule,
                highlightAtoms=sorted(environment_atoms),
                highlightBonds=environment_bonds,
            )
            representative_drawer.FinishDrawing()
            representative_svg = representative_drawer.GetDrawingText()
            representative_visual = (
                '<div class="representative-image">'
                f'{representative_svg}'
                '</div>'
            )
        except Exception as representative_error:
            representative_visual = (
                '<div class="fragment-error">'
                'Representative molecule image unavailable.<br>'
                f'{html.escape(str(representative_error))}'
                '</div>'
            )

        visual = (
            '<div class="molecule-visuals">'
            '<div><h3>Representative molecule</h3>'
            f'{representative_visual}</div>'
            '<div><h3>Highlighted Morgan environment</h3>'
            f'{fragment_visual}</div>'
            '</div>'
        )

        relative_width = (
            100.0 * importance / maximum_importance
            if maximum_importance > 0
            else 0.0
        )

        html_cards.append(
            f'''<article class="bit-card">
                <div class="card-heading">
                    <div>
                        <span class="rank">Rank {rank}</span>
                        <h2>Morgan bit {bit}</h2>
                    </div>
                    <strong>{importance:.6f}</strong>
                </div>
                <div class="importance-track" title="Relative feature importance">
                    <div style="width: {relative_width:.3f}%"></div>
                </div>
                {visual}
                <dl>
                    <div><dt>Feature importance</dt><dd>{importance:.6f}</dd></div>
                    <div><dt>Molecules containing bit</dt><dd>{molecules_containing_bit:,}</dd></div>
                    <div><dt>% of dataset</dt><dd>{percentage_of_dataset:.2f}%</dd></div>
                    <div><dt>Mean {target_label}</dt><dd>{mean_target_value:.6g}</dd></div>
                    <div><dt>SD {target_label}</dt><dd>{sd_target_value:.6g}</dd></div>
                </dl>
            </article>'''
        )

    gallery_html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{workflow_name}: important Morgan bits</title>
<style>
    :root {{ --background:#f4f6f8; --card:#fff; --text:#17202a; --muted:#5d6d7e; --border:#d5d8dc; --accent:#2874a6; --track:#e5e7e9; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Arial,Helvetica,sans-serif; background:var(--background); color:var(--text); }}
    main {{ max-width:1500px; margin:auto; padding:28px; }}
    header {{ margin-bottom:24px; }}
    h1 {{ margin:0 0 8px; }}
    header p {{ color:var(--muted); margin:5px 0; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:20px; }}
    .bit-card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px; break-inside:avoid; }}
    .card-heading {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }}
    .card-heading h2 {{ margin:4px 0 10px; font-size:1.2rem; }}
    .rank {{ color:var(--muted); font-size:.9rem; }}
    .importance-track {{ height:10px; border-radius:999px; overflow:hidden; background:var(--track); margin-bottom:14px; }}
    .importance-track div {{ height:100%; background:var(--accent); }}
    .molecule-visuals {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    .molecule-visuals h3 {{ margin:0 0 7px; color:var(--muted); font-size:.9rem; text-align:center; }}
    .fragment-image,.representative-image {{ min-height:250px; display:flex; align-items:center; justify-content:center; overflow:hidden; background:white; border:1px solid var(--border); border-radius:10px; padding:8px; }}
    .fragment-image svg,.fragment-image img,.representative-image svg,.representative-image img {{ width:100%; max-height:310px; object-fit:contain; }}
    .fragment-error {{ min-height:250px; display:grid; place-items:center; color:#a93226; border:1px dashed #a93226; border-radius:10px; padding:18px; }}
    dl {{ display:grid; grid-template-columns:1fr 1fr; gap:10px 16px; margin:16px 0; }}
    dl div {{ border-bottom:1px solid var(--border); padding-bottom:7px; }}
    dt {{ color:var(--muted); font-size:.85rem; }}
    dd {{ margin:3px 0 0; font-weight:600; }}
    details {{ border-top:1px solid var(--border); padding-top:12px; }}
    summary {{ cursor:pointer; font-weight:600; }}
    code {{ display:block; overflow-wrap:anywhere; white-space:pre-wrap; background:var(--background); border-radius:6px; padding:8px; }}
    @media (max-width:750px) {{ main {{ padding:14px; }} .grid {{ grid-template-columns:1fr; }} .molecule-visuals {{ grid-template-columns:1fr; }} dl {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
    <header>
        <h1>{workflow_name}: important Morgan fingerprint bits</h1>
        <p>Target class rule: {class_rule}.</p>
        <p>Fixed model: radius={RADIUS}, nBits={NBITS}, max depth={MAX_DEPTH}, criterion={CRITERION}.</p>
        <p>All images and metadata are embedded in this single HTML file. Bits are ordered by decision-tree feature importance.</p>
        <p>Highlighted atoms and bonds show the environment represented by each bit in one randomly selected dataset molecule. Representatives are sampled without replacement across all reports in this run whenever possible.</p>
    </header>
    <section class="grid">
        {''.join(html_cards) if html_cards else '<p>No non-zero feature importances were found.</p>'}
    </section>
</main>
</body>
</html>
'''

    (output_dir / "important_morgan_bits.html").write_text(
        gallery_html,
        encoding="utf-8",
    )

def run_workflow(
    *,
    workflow_key: str,
    workflow_name: str,
    direction: str,
    percentage: float,
    target_column: str,
    target_label: str,
    data: pd.DataFrame,
    x: np.ndarray,
    bit_infos: list[dict[int, list[tuple[int, int]]]],
    input_files: list[Path],
    args: argparse.Namespace,
    representative_rng: np.random.Generator,
    used_representative_indices: set[int],
) -> None:
    """Train, evaluate, and save one independent threshold workflow."""
    workflow_output_dir = args.output_dir / workflow_key
    workflow_output_dir.mkdir(parents=True, exist_ok=True)

    y, actual_percentage, boundary_value = make_labels(
        data[target_column],
        percentage,
        direction,
    )

    workflow_data = data.copy()
    workflow_data["class_label"] = y

    class_counts = Counter(y)
    if direction == "high":
        class_rule = (
            f"top {actual_percentage:.4g}% by {target_label} gives target "
            f"class 1 (boundary value {boundary_value:.6g})"
        )
    else:
        class_rule = (
            f"bottom {actual_percentage:.4g}% by {target_label} gives target "
            f"class 1 (boundary value {boundary_value:.6g})"
        )

    print(f"\n{'=' * 72}")
    print(f"{workflow_name} workflow")
    print(f"{'=' * 72}")
    print(f"Class rule: {class_rule}")
    print(
        f"Requested target percentage: {percentage:g}%; "
        f"actual: {actual_percentage:.4g}%"
    )
    print(f"Non-target class (0): {class_counts[0]:,}")
    print(f"Target class (1): {class_counts[1]:,}")

    indices = np.arange(len(workflow_data))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    split_names = np.full(len(workflow_data), "", dtype=object)
    split_names[train_indices] = "training"
    split_names[test_indices] = "untouched_test"
    workflow_data["data_split"] = split_names

    print(
        f"Training molecules: {len(train_indices):,}; "
        f"untouched test molecules: {len(test_indices):,}"
    )

    model = DecisionTreeClassifier(
        random_state=args.random_state,
        class_weight="balanced",
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        criterion=CRITERION,
    )
    model.fit(x[train_indices], y[train_indices])

    training_metrics = calculate_metrics(
        model,
        x[train_indices],
        y[train_indices],
    )
    test_metrics = calculate_metrics(
        model,
        x[test_indices],
        y[test_indices],
    )

    generalisation_gap = (
        training_metrics["balanced_accuracy"]
        - test_metrics["balanced_accuracy"]
    )

    print("\nTraining metrics:")
    for name, value in training_metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\nUntouched-test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    print(
        "\nTraining-test balanced-accuracy gap: "
        f"{generalisation_gap:.4f}"
    )

    metrics_table = pd.DataFrame(
        [
            {"data_split": "training", **training_metrics},
            {"data_split": "untouched_test", **test_metrics},
        ]
    )
    metrics_table.to_csv(
        workflow_output_dir / "training_and_test_metrics.csv",
        index=False,
    )

    test_predictions = model.predict(x[test_indices])
    report = classification_report(
        y[test_indices],
        test_predictions,
        digits=4,
        zero_division=0,
    )
    (workflow_output_dir / "test_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y[test_indices],
        test_predictions,
        ax=axis,
        cmap="Blues",
    )
    axis.set_title(f"{workflow_name}: untouched test set")
    figure.tight_layout()
    figure.savefig(
        workflow_output_dir / "test_confusion_matrix.png",
        dpi=250,
    )
    plt.close(figure)

    importances = model.feature_importances_
    important_bits = np.flatnonzero(importances > 0)
    order = important_bits[np.argsort(importances[important_bits])[::-1]]

    target_values = workflow_data[target_column].to_numpy(dtype=float)
    target_mask = y == 1
    non_target_mask = y == 0

    feature_rows = []
    for rank, bit in enumerate(order, start=1):
        present = x[:, bit] > 0
        target_present = int(np.count_nonzero(present & target_mask))
        non_target_present = int(np.count_nonzero(present & non_target_mask))
        target_total = int(np.count_nonzero(target_mask))
        non_target_total = int(np.count_nonzero(non_target_mask))

        target_percentage = (
            100.0 * target_present / target_total if target_total else 0.0
        )
        non_target_percentage = (
            100.0 * non_target_present / non_target_total
            if non_target_total
            else 0.0
        )

        feature_rows.append(
            {
                "rank": rank,
                "bit": int(bit),
                "importance": float(importances[bit]),
                "molecules_containing_bit": int(np.count_nonzero(present)),
                "percentage_of_dataset": (
                    100.0 * np.count_nonzero(present) / len(workflow_data)
                ),
                "mean_target_value": float(target_values[present].mean()),
                "sd_target_value": (
                    float(target_values[present].std(ddof=1))
                    if np.count_nonzero(present) > 1
                    else 0.0
                ),
                "target_class_molecules_containing_bit": target_present,
                "target_class_percentage_containing_bit": target_percentage,
                "non_target_class_molecules_containing_bit": non_target_present,
                "non_target_class_percentage_containing_bit": non_target_percentage,
            }
        )

    feature_table = pd.DataFrame(feature_rows)
    feature_table.to_csv(
        workflow_output_dir / "feature_importances.csv",
        index=False,
    )

    save_fragments(
        feature_table=feature_table,
        data=workflow_data,
        fingerprints=x,
        bit_infos=bit_infos,
        output_dir=workflow_output_dir,
        max_images=args.max_fragment_images,
        workflow_name=workflow_name,
        class_rule=class_rule,
        target_label=target_label,
        representative_rng=representative_rng,
        used_representative_indices=used_representative_indices,
    )

    workflow_data.drop(columns=["mol"]).to_csv(
        workflow_output_dir / "molecules_labels_and_splits.csv",
        index=False,
    )

    tree_rules = export_text(
        model,
        feature_names=[f"bit_{index}" for index in range(NBITS)],
    )
    (workflow_output_dir / "decision_tree_rules.txt").write_text(
        tree_rules,
        encoding="utf-8",
    )

    settings = {
        "workflow": workflow_key,
        "workflow_name": workflow_name,
        "direction": direction,
        "target_column": target_column,
        "target_label": target_label,
        "input_path": str(args.input_path.resolve()),
        "input_files": [str(path.resolve()) for path in input_files],
        "fixed_hyperparameters": {
            "radius": RADIUS,
            "nbits": NBITS,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "criterion": CRITERION,
            "class_weight": "balanced",
        },
        "requested_target_percentage": percentage,
        "actual_target_percentage": actual_percentage,
        "boundary_value": boundary_value,
        "class_rule": class_rule,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "remove_gold": args.remove_gold,
        "deduplicate_smiles": args.deduplicate_smiles,
        "number_of_valid_molecules": len(workflow_data),
        "training_molecules": int(len(train_indices)),
        "untouched_test_molecules": int(len(test_indices)),
        "class_counts": {
            "0": int(class_counts[0]),
            "1": int(class_counts[1]),
        },
        "actual_tree_depth": int(model.get_depth()),
        "tree_node_count": int(model.tree_.node_count),
        "tree_leaf_count": int(model.get_n_leaves()),
        "training_metrics": training_metrics,
        "test_metrics": test_metrics,
        "balanced_accuracy_generalisation_gap": float(generalisation_gap),
    }
    (workflow_output_dir / "run_settings_and_metrics.json").write_text(
        json.dumps(settings, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print(f"\nActual fitted depth: {model.get_depth()}")
    print(f"Tree nodes: {model.tree_.node_count}")
    print(f"Tree leaves: {model.get_n_leaves()}")
    print(f"Saved workflow results to: {workflow_output_dir.resolve()}")


def main() -> int:
    args = parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading runs from {args.input_path}...")
    raw, input_files = load_all_tables(args.input_path)
    data = prepare_data(raw, args)

    print(f"Valid molecules: {len(data):,}")

    print("\nFixed hyperparameter configuration:")
    print(f"  radius: {RADIUS}")
    print(f"  nbits: {NBITS}")
    print(f"  max_depth: {MAX_DEPTH}")
    print(f"  min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"  min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"  criterion: {CRITERION}")

    print(
        f"\nGenerating Morgan fingerprints "
        f"(radius={RADIUS}, nBits={NBITS})..."
    )
    x, bit_infos = make_fingerprints(data["mol"].tolist())

    # Representative molecules are sampled randomly without replacement across
    # all four reports generated during this execution.
    representative_rng = np.random.default_rng()
    used_representative_indices: set[int] = set()

    run_workflow(
        workflow_key="high_performing",
        workflow_name="High-performing",
        direction="high",
        percentage=args.high_fitness_percent,
        target_column="fitness",
        target_label="fitness",
        data=data,
        x=x,
        bit_infos=bit_infos,
        input_files=input_files,
        args=args,
        representative_rng=representative_rng,
        used_representative_indices=used_representative_indices,
    )

    run_workflow(
        workflow_key="low_performing",
        workflow_name="Low-performing",
        direction="low",
        percentage=args.low_fitness_percent,
        target_column="fitness",
        target_label="fitness",
        data=data,
        x=x,
        bit_infos=bit_infos,
        input_files=input_files,
        args=args,
        representative_rng=representative_rng,
        used_representative_indices=used_representative_indices,
    )

    run_workflow(
        workflow_key="high_log_p",
        workflow_name="High log P",
        direction="high",
        percentage=args.high_logp_percent,
        target_column="log_p",
        target_label=args.logp_column,
        data=data,
        x=x,
        bit_infos=bit_infos,
        input_files=input_files,
        args=args,
        representative_rng=representative_rng,
        used_representative_indices=used_representative_indices,
    )

    run_workflow(
        workflow_key="low_log_p",
        workflow_name="Low log P",
        direction="low",
        percentage=args.low_logp_percent,
        target_column="log_p",
        target_label=args.logp_column,
        data=data,
        x=x,
        bit_infos=bit_infos,
        input_files=input_files,
        args=args,
        representative_rng=representative_rng,
        used_representative_indices=used_representative_indices,
    )

    print(f"\nAll results saved under: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
