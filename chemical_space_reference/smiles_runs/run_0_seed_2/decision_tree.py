#!/usr/bin/env python3
"""
Decision-tree classifier for full_history.csv using hashed Morgan fingerprints.

Class definition:
    fitness >= 0.5  -> high-performing (1)
    fitness <  0.5  -> low-performing  (0)

Example:
    python decision_tree_fixed.py full_history.csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and interpret a decision tree using Morgan fingerprints."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("full_history.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("morgan_tree_results"),
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--fitness-column", default="fitness")
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=2048)
    parser.add_argument(
        "--tree-depths",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8, 10, 12, 16, 20],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
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
        # semicolons. Detect comma, semicolon, or tab from the header.
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

def clean_smiles_value(value: Any) -> str | None:
    """Return a clean SMILES string, or None for missing/non-text values."""
    if value is None or pd.isna(value):
        return None

    # Reject numeric values explicitly rather than sending them to RDKit.
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
    finite_fitness = np.isfinite(data["fitness"].to_numpy(dtype=float, na_value=np.nan))
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
        # This check guarantees RDKit always receives a Python string.
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
    print(f"Rows removed because RDKit could not parse SMILES: {invalid_rdkit_count:,}")
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

    # Stratified train/validation/test splitting needs enough members per class.
    if min(counts.values()) < 5:
        raise ValueError(
            "The minority class has fewer than 5 molecules, so a reliable "
            "stratified split is not possible."
        )

    return y, threshold


def make_fingerprints(
    molecules: list[Chem.Mol],
    radius: int,
    nbits: int,
) -> tuple[np.ndarray, list[dict[int, list[tuple[int, int]]]]]:
    if radius < 0:
        raise ValueError("Morgan radius must be non-negative.")
    if nbits <= 0:
        raise ValueError("nBits must be positive.")

    x = np.zeros((len(molecules), nbits), dtype=np.int32)
    bit_infos: list[dict[int, list[tuple[int, int]]]] = []

    for row_number, mol in enumerate(molecules):
        bit_info: dict[int, list[tuple[int, int]]] = {}
        fingerprint = rdMolDescriptors.GetHashedMorganFingerprint(
            mol,
            radius,
            nBits=nbits,
            useFeatures=False,
            bitInfo=bit_info,
        )
        array = np.zeros((nbits,), dtype=np.int32)
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

    result = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
    }

    result["roc_auc"] = (
        float(roc_auc_score(y, probability))
        if len(np.unique(y)) == 2
        else float("nan")
    )
    return result


def save_depth_plot(results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        results["max_depth"],
        results["training_balanced_accuracy"],
        marker="o",
        label="Training",
    )
    axis.plot(
        results["max_depth"],
        results["validation_balanced_accuracy"],
        marker="o",
        label="Validation",
    )
    axis.set_xlabel("Maximum tree depth")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Decision-tree depth selection")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=250)
    plt.close(figure)


def save_fragments(
    feature_table: pd.DataFrame,
    data: pd.DataFrame,
    fingerprints: np.ndarray,
    bit_infos: list[dict[int, list[tuple[int, int]]]],
    output_dir: Path,
    max_images: int,
) -> None:
    fragment_dir = output_dir / "important_fragments"
    fragment_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for _, row in feature_table.head(max_images).iterrows():
        bit = int(row["bit"])
        candidates = [
            index for index, info in enumerate(bit_infos) if bit in info
        ]
        if not candidates:
            continue

        molecule_index = max(
            candidates,
            key=lambda index: fingerprints[index, bit],
        )
        molecule = data.iloc[molecule_index]["mol"]
        rank = int(row["rank"])
        base_name = f"rank_{rank:02d}_bit_{bit}"

        image_path: Path | None = None
        image_error: str | None = None

        try:
            image = DrawMorganBit(
                molecule,
                bit,
                bit_infos[molecule_index],
                useSVG=False,
            )
            image_path = fragment_dir / f"{base_name}.png"
            image.save(str(image_path))
        except Exception as png_error:
            # SVG output does not require RDKit Cairo support.
            try:
                svg = DrawMorganBit(
                    molecule,
                    bit,
                    bit_infos[molecule_index],
                    useSVG=True,
                )
                image_path = fragment_dir / f"{base_name}.svg"
                image_path.write_text(str(svg), encoding="utf-8")
            except Exception as svg_error:
                image_error = f"PNG: {png_error}; SVG: {svg_error}"

        records.append(
            {
                "rank": rank,
                "bit": bit,
                "importance": float(row["importance"]),
                "representative_row": int(molecule_index),
                "representative_smiles": data.iloc[molecule_index][
                    "original_smiles"
                ],
                "canonical_smiles": data.iloc[molecule_index][
                    "canonical_smiles"
                ],
                "fitness": float(data.iloc[molecule_index]["fitness"]),
                "fragment_image": str(image_path) if image_path else "",
                "image_error": image_error or "",
            }
        )

    pd.DataFrame(records).to_csv(
        output_dir / "important_fragment_examples.csv",
        index=False,
    )


def main() -> int:
    args = parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")
    if args.test_size + args.validation_size >= 1:
        raise ValueError("test-size + validation-size must be less than 1.")
    if any(depth <= 0 for depth in args.tree_depths):
        raise ValueError("All tree depths must be positive integers.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.input_file}...")
    raw = load_table(args.input_file)
    data = prepare_data(raw, args)
    y, threshold = make_labels(data["fitness"], args.label_threshold)
    data["class_label"] = y

    class_counts = Counter(y)
    print(f"Valid molecules: {len(data):,}")
    print(f"Class rule: fitness >= {threshold:g} is class 1")
    print(f"Class 0: {class_counts[0]:,}")
    print(f"Class 1: {class_counts[1]:,}")

    print(
        f"Generating Morgan fingerprints "
        f"(radius={args.radius}, nBits={args.nbits})..."
    )
    x, bit_infos = make_fingerprints(
        data["mol"].tolist(),
        args.radius,
        args.nbits,
    )

    indices = np.arange(len(data))
    train_validation, test = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    relative_validation_size = args.validation_size / (1 - args.test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=args.random_state,
        stratify=y[train_validation],
    )

    split_names = np.full(len(data), "", dtype=object)
    split_names[train] = "training"
    split_names[validation] = "validation"
    split_names[test] = "test"
    data["data_split"] = split_names

    print(
        f"Training: {len(train):,}; validation: {len(validation):,}; "
        f"test: {len(test):,}"
    )

    depth_records: list[dict[str, float | int]] = []
    for depth in args.tree_depths:
        model = DecisionTreeClassifier(
            random_state=args.random_state,
            max_depth=depth,
            class_weight="balanced",
        )
        model.fit(x[train], y[train])

        training_metrics = calculate_metrics(model, x[train], y[train])
        validation_metrics = calculate_metrics(
            model,
            x[validation],
            y[validation],
        )

        depth_records.append(
            {
                "max_depth": depth,
                "training_balanced_accuracy": training_metrics[
                    "balanced_accuracy"
                ],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_f1": validation_metrics["f1"],
                "validation_mcc": validation_metrics["mcc"],
                "validation_roc_auc": validation_metrics["roc_auc"],
            }
        )

    depth_results = pd.DataFrame(depth_records).sort_values("max_depth")
    depth_results.to_csv(
        args.output_dir / "tree_depth_validation_scores.csv",
        index=False,
    )
    save_depth_plot(
        depth_results,
        args.output_dir / "tree_depth_validation_scores.png",
    )

    best_row = depth_results.sort_values(
        ["validation_balanced_accuracy", "max_depth"],
        ascending=[False, True],
    ).iloc[0]
    best_depth = int(best_row["max_depth"])
    print(f"Selected max_depth: {best_depth}")

    final_training = np.concatenate([train, validation])
    final_model = DecisionTreeClassifier(
        random_state=args.random_state,
        max_depth=best_depth,
        class_weight="balanced",
    )
    final_model.fit(x[final_training], y[final_training])

    test_metrics = calculate_metrics(final_model, x[test], y[test])
    print("Final untouched-test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    test_predictions = final_model.predict(x[test])
    report = classification_report(
        y[test],
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
        y[test],
        test_predictions,
        ax=axis,
        cmap="Blues",
    )
    axis.set_title("Decision tree: untouched test set")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "test_confusion_matrix.png",
        dpi=250,
    )
    plt.close(figure)

    importances = final_model.feature_importances_
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
        feature_table,
        data,
        x,
        bit_infos,
        args.output_dir,
        args.max_fragment_images,
    )

    data.drop(columns=["mol"]).to_csv(
        args.output_dir / "molecules_labels_and_splits.csv",
        index=False,
    )

    tree_rules = export_text(
        final_model,
        feature_names=[f"bit_{index}" for index in range(args.nbits)],
    )
    (args.output_dir / "decision_tree_rules.txt").write_text(
        tree_rules,
        encoding="utf-8",
    )

    settings = {
        "input_file": str(args.input_file.resolve()),
        "radius": args.radius,
        "nbits": args.nbits,
        "fitness_threshold": threshold,
        "class_rule": f"fitness >= {threshold:g} gives class 1",
        "tree_depths_tested": args.tree_depths,
        "selected_tree_depth": best_depth,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "random_state": args.random_state,
        "remove_gold": args.remove_gold,
        "deduplicate_smiles": args.deduplicate_smiles,
        "number_of_valid_molecules": len(data),
        "class_counts": {
            "0": int(class_counts[0]),
            "1": int(class_counts[1]),
        },
        "test_metrics": test_metrics,
    }
    (args.output_dir / "run_settings_and_metrics.json").write_text(
        json.dumps(settings, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print(f"Saved results to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
