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

Class definition:
    fitness >= 0.5  -> high-performing (1)
    fitness <  0.5  -> low-performing  (0)

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
from rdkit.Chem.Draw import DrawMorganBit
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
    parser.add_argument("--label-threshold", type=float, default=0.5)
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
    required = {args.smiles_column, args.fitness_column}
    missing = required.difference(raw.columns)

    if missing:
        available = ", ".join(map(str, raw.columns))
        raise KeyError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns are: {available}"
        )

    data = raw[[args.smiles_column, args.fitness_column]].copy()
    data.columns = ["original_smiles", "fitness"]
    rows_read = len(data)

    data["fitness"] = pd.to_numeric(data["fitness"], errors="coerce")
    finite_fitness = np.isfinite(
        data["fitness"].to_numpy(dtype=float, na_value=np.nan)
    )
    invalid_fitness_count = int((~finite_fitness).sum())
    data = data.loc[finite_fitness].copy()

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
    fitness: pd.Series,
    threshold: float,
) -> tuple[np.ndarray, float]:
    if not math.isfinite(threshold):
        raise ValueError("The label threshold must be finite.")

    y = (fitness.to_numpy(dtype=float) >= threshold).astype(np.int8)
    counts = Counter(y)

    if len(counts) != 2:
        raise ValueError(
            f"Threshold {threshold} produced only one class: {dict(counts)}."
        )

    if min(counts.values()) < 2:
        raise ValueError(
            "The minority class has fewer than 2 molecules, so a stratified "
            "train/test split is not possible."
        )

    return y, threshold


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
) -> None:
    '''Save all important Morgan-bit visualisations in one self-contained HTML file.'''
    import base64
    import html
    from io import BytesIO

    html_cards: list[str] = []
    selected_features = feature_table.head(max_images).copy()
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

        molecule_index = max(candidates, key=lambda index: fingerprints[index, bit])
        molecule = data.iloc[molecule_index]["mol"]
        rank = int(row["rank"])
        importance = float(row["importance"])
        representative_smiles = str(data.iloc[molecule_index]["original_smiles"])
        canonical_smiles = str(data.iloc[molecule_index]["canonical_smiles"])
        fitness = float(data.iloc[molecule_index]["fitness"])
        count_in_molecule = int(fingerprints[molecule_index, bit])
        molecules_containing_bit = int(row["molecules_containing_bit"])
        mean_count_when_present = float(row["mean_count_when_present"])

        try:
            svg_text = str(
                DrawMorganBit(
                    molecule,
                    bit,
                    bit_infos[molecule_index],
                    useSVG=True,
                )
            )
            visual = f'<div class="fragment-image">{svg_text}</div>'
        except Exception as svg_error:
            try:
                image = DrawMorganBit(
                    molecule,
                    bit,
                    bit_infos[molecule_index],
                    useSVG=False,
                )
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                encoded_png = base64.b64encode(buffer.getvalue()).decode("ascii")
                visual = (
                    '<div class="fragment-image">'
                    f'<img src="data:image/png;base64,{encoded_png}" '
                    f'alt="Highlighted environment for Morgan bit {bit}">'
                    '</div>'
                )
            except Exception as png_error:
                error_text = html.escape(f"SVG: {svg_error}; PNG: {png_error}")
                visual = (
                    '<div class="fragment-error">Fragment image unavailable.<br>'
                    f'{error_text}</div>'
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
                    <div><dt>Representative fitness</dt><dd>{fitness:.6g}</dd></div>
                    <div><dt>Molecules containing bit</dt><dd>{molecules_containing_bit:,}</dd></div>
                    <div><dt>Count in representative</dt><dd>{count_in_molecule}</dd></div>
                    <div><dt>Mean count when present</dt><dd>{mean_count_when_present:.3f}</dd></div>
                    <div><dt>Representative row</dt><dd>{molecule_index}</dd></div>
                </dl>
                <details>
                    <summary>Show representative SMILES</summary>
                    <p><strong>Original:</strong><br><code>{html.escape(representative_smiles)}</code></p>
                    <p><strong>Canonical:</strong><br><code>{html.escape(canonical_smiles)}</code></p>
                </details>
            </article>'''
        )

    gallery_html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Important Morgan bits</title>
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
    .fragment-image {{ min-height:250px; display:flex; align-items:center; justify-content:center; overflow:hidden; background:white; border:1px solid var(--border); border-radius:10px; padding:8px; }}
    .fragment-image svg,.fragment-image img {{ width:100%; max-height:310px; object-fit:contain; }}
    .fragment-error {{ min-height:250px; display:grid; place-items:center; color:#a93226; border:1px dashed #a93226; border-radius:10px; padding:18px; }}
    dl {{ display:grid; grid-template-columns:1fr 1fr; gap:10px 16px; margin:16px 0; }}
    dl div {{ border-bottom:1px solid var(--border); padding-bottom:7px; }}
    dt {{ color:var(--muted); font-size:.85rem; }}
    dd {{ margin:3px 0 0; font-weight:600; }}
    details {{ border-top:1px solid var(--border); padding-top:12px; }}
    summary {{ cursor:pointer; font-weight:600; }}
    code {{ display:block; overflow-wrap:anywhere; white-space:pre-wrap; background:var(--background); border-radius:6px; padding:8px; }}
    @media (max-width:600px) {{ main {{ padding:14px; }} .grid {{ grid-template-columns:1fr; }} dl {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
    <header>
        <h1>Important Morgan fingerprint bits</h1>
        <p>Fixed model: radius={RADIUS}, nBits={NBITS}, max depth={MAX_DEPTH}, criterion={CRITERION}.</p>
        <p>All images and metadata are embedded in this single HTML file. Bits are ordered by decision-tree feature importance.</p>
        <p>Highlighted atoms and bonds show the environment represented by each bit in one example molecule.</p>
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

def main() -> int:
    args = parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading runs from {args.input_path}...")
    raw, input_files = load_all_tables(args.input_path)
    data = prepare_data(raw, args)

    y, threshold = make_labels(data["fitness"], args.label_threshold)
    data["class_label"] = y

    class_counts = Counter(y)
    print(f"Valid molecules: {len(data):,}")
    print(f"Class rule: fitness >= {threshold:g} is class 1")
    print(f"Class 0: {class_counts[0]:,}")
    print(f"Class 1: {class_counts[1]:,}")

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

    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    split_names = np.full(len(data), "", dtype=object)
    split_names[train_indices] = "training"
    split_names[test_indices] = "untouched_test"
    data["data_split"] = split_names

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
        args.output_dir / "training_and_test_metrics.csv",
        index=False,
    )

    test_predictions = model.predict(x[test_indices])
    report = classification_report(
        y[test_indices],
        test_predictions,
        digits=4,
        zero_division=0,
    )
    (args.output_dir / "test_classification_report.txt").write_text(
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
    axis.set_title("Fixed decision tree: untouched test set")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "test_confusion_matrix.png",
        dpi=250,
    )
    plt.close(figure)

    importances = model.feature_importances_
    important_bits = np.flatnonzero(importances > 0)
    order = important_bits[np.argsort(importances[important_bits])[::-1]]

    feature_table = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "bit": order,
            "importance": importances[order],
            "molecules_containing_bit": [
                int(np.count_nonzero(x[:, bit])) for bit in order
            ],
            "mean_count_when_present": [
                float(x[x[:, bit] > 0, bit].mean()) for bit in order
            ],
        }
    )
    feature_table.to_csv(
        args.output_dir / "feature_importances.csv",
        index=False,
    )

    save_fragments(
        feature_table=feature_table,
        data=data,
        fingerprints=x,
        bit_infos=bit_infos,
        output_dir=args.output_dir,
        max_images=args.max_fragment_images,
    )

    data.drop(columns=["mol"]).to_csv(
        args.output_dir / "molecules_labels_and_splits.csv",
        index=False,
    )

    tree_rules = export_text(
        model,
        feature_names=[f"bit_{index}" for index in range(NBITS)],
    )
    (args.output_dir / "decision_tree_rules.txt").write_text(
        tree_rules,
        encoding="utf-8",
    )

    settings = {
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
        "fitness_threshold": threshold,
        "class_rule": f"fitness >= {threshold:g} gives class 1",
        "test_size": args.test_size,
        "random_state": args.random_state,
        "remove_gold": args.remove_gold,
        "deduplicate_smiles": args.deduplicate_smiles,
        "number_of_valid_molecules": len(data),
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
    (args.output_dir / "run_settings_and_metrics.json").write_text(
        json.dumps(settings, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print(f"\nActual fitted depth: {model.get_depth()}")
    print(f"Tree nodes: {model.tree_.node_count}")
    print(f"Tree leaves: {model.get_n_leaves()}")
    print(f"\nSaved results to: {args.output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
