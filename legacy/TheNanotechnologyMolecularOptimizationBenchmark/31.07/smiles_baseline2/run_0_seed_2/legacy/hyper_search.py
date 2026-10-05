#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import DrawMorganBit
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

RADII = [1, 2, 3]
NBITS_VALUES = [1024, 2048, 4096]
MAX_DEPTHS = [3, 5, 8, 10, 12, 16, 20]
MIN_SAMPLES_SPLITS = [2, 5, 10]
MIN_SAMPLES_LEAVES = [1, 2, 5]
CRITERIA = ["gini", "entropy"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1,134-model Morgan decision-tree grid search")
    p.add_argument("input_file", nargs="?", type=Path, default=Path("full_history.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("morgan_tree_results"))
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--fitness-column", default="fitness")
    p.add_argument("--label-threshold", type=float, default=0.5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--validation-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=1)
    p.add_argument("--max-fragment-images", type=int, default=12)
    p.add_argument("--remove-gold", action="store_true")
    p.add_argument("--deduplicate-smiles", action="store_true")
    return p.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Input file must be CSV or Parquet.")
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    counts = {",": header.count(","), ";": header.count(";"), "\t": header.count("\t")}
    delimiter = max(counts, key=counts.get)
    if counts[delimiter] == 0:
        raise ValueError("Could not detect CSV delimiter.")
    print(f"Detected delimiter: {repr(delimiter)}")
    df = pd.read_csv(path, sep=delimiter, low_memory=False, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_smiles(value: Any) -> str | None:
    if value is None or pd.isna(value) or isinstance(value, (int, float, np.integer, np.floating)):
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none", "null", "na", "n/a"} else text


def prepare_data(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    missing = {args.smiles_column, args.fitness_column}.difference(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    data = raw[[args.smiles_column, args.fitness_column]].copy()
    data.columns = ["original_smiles", "fitness"]
    rows_read = len(data)
    data["fitness"] = pd.to_numeric(data["fitness"], errors="coerce")
    data = data[np.isfinite(data["fitness"].to_numpy(dtype=float, na_value=np.nan))].copy()
    data["processed_smiles"] = data["original_smiles"].map(clean_smiles)
    data = data.dropna(subset=["processed_smiles"]).copy()
    if args.remove_gold:
        data["processed_smiles"] = data["processed_smiles"].str.replace("[Au]", "", regex=False).str.strip()
        data = data[data["processed_smiles"].ne("")].copy()

    rows = []
    bad = []
    for _, row in data.iterrows():
        mol = Chem.MolFromSmiles(row["processed_smiles"])
        if mol is None:
            if len(bad) < 10:
                bad.append(row["processed_smiles"])
            continue
        rows.append({
            "original_smiles": row["original_smiles"],
            "fitness": float(row["fitness"]),
            "processed_smiles": row["processed_smiles"],
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "mol": mol,
        })
    data = pd.DataFrame(rows)
    print(f"Rows read: {rows_read:,}")
    print(f"Valid RDKit molecules: {len(data):,}")
    if bad:
        print("Examples of unparseable SMILES:")
        for item in bad:
            print(f"  {item}")
    if args.deduplicate_smiles:
        before = len(data)
        data = data.sort_values("fitness", ascending=False).drop_duplicates("canonical_smiles").reset_index(drop=True)
        print(f"Duplicate molecules removed: {before - len(data):,}")
    else:
        data = data.reset_index(drop=True)
    if len(data) < 10:
        raise ValueError("Too few valid molecules for training.")
    return data


def make_labels(fitness: pd.Series, threshold: float) -> np.ndarray:
    if not math.isfinite(threshold):
        raise ValueError("The label threshold must be finite.")
    y = (fitness.to_numpy(dtype=float) >= threshold).astype(np.int8)
    counts = Counter(y)
    if len(counts) != 2 or min(counts.values()) < 5:
        raise ValueError(f"Threshold produced unsuitable class counts: {dict(counts)}")
    return y


def make_fingerprints(mols: list[Chem.Mol], radius: int, nbits: int):
    x = np.zeros((len(mols), nbits), dtype=np.int32)
    infos = []
    for i, mol in enumerate(mols):
        info = {}
        fp = rdMolDescriptors.GetHashedMorganFingerprint(
            mol, radius, nBits=nbits, useFeatures=False, bitInfo=info
        )
        arr = np.zeros(nbits, dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        x[i] = arr
        infos.append(info)
    return x, infos


def metrics(model, x, y):
    pred = model.predict(x)
    prob = model.predict_proba(x)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
    }


def save_top_plot(results: pd.DataFrame, path: Path):
    top = results.head(25).copy()
    labels = [
        f"r={r.radius}, n={r.nbits}, d={r.max_depth}, s={r.min_samples_split}, l={r.min_samples_leaf}, {r.criterion}"
        for r in top.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(12, 9))
    pos = np.arange(len(top))
    ax.barh(pos, top["validation_balanced_accuracy"])
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Validation balanced accuracy")
    ax.set_title("Top 25 hyperparameter combinations")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def save_fragments(feature_table, data, x, bit_infos, output_dir, max_images):
    fragment_dir = output_dir / "important_fragments"
    fragment_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in feature_table.head(max_images).iterrows():
        bit = int(row["bit"])
        candidates = [i for i, info in enumerate(bit_infos) if bit in info]
        if not candidates:
            continue
        idx = max(candidates, key=lambda i: x[i, bit])
        rank = int(row["rank"])
        base = f"rank_{rank:02d}_bit_{bit}"
        image_path = None
        error = ""
        try:
            image = DrawMorganBit(data.iloc[idx]["mol"], bit, bit_infos[idx], useSVG=False)
            image_path = fragment_dir / f"{base}.png"
            image.save(str(image_path))
        except Exception as png_error:
            try:
                svg = DrawMorganBit(data.iloc[idx]["mol"], bit, bit_infos[idx], useSVG=True)
                image_path = fragment_dir / f"{base}.svg"
                image_path.write_text(str(svg), encoding="utf-8")
            except Exception as svg_error:
                error = f"PNG: {png_error}; SVG: {svg_error}"
        records.append({
            "rank": rank,
            "bit": bit,
            "importance": float(row["importance"]),
            "representative_row": int(idx),
            "representative_smiles": data.iloc[idx]["original_smiles"],
            "canonical_smiles": data.iloc[idx]["canonical_smiles"],
            "fitness": float(data.iloc[idx]["fitness"]),
            "fragment_image": str(image_path) if image_path else "",
            "image_error": error,
        })
    pd.DataFrame(records).to_csv(output_dir / "important_fragment_examples.csv", index=False)


def main() -> int:
    args = parse_args()
    if not 0 < args.test_size < 1 or not 0 < args.validation_size < 1:
        raise ValueError("Test and validation sizes must be between 0 and 1.")
    if args.test_size + args.validation_size >= 1:
        raise ValueError("test-size + validation-size must be less than 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(load_table(args.input_file), args)
    y = make_labels(data["fitness"], args.label_threshold)
    data["class_label"] = y
    class_counts = Counter(y)
    print(f"Class 0: {class_counts[0]:,}; class 1: {class_counts[1]:,}")

    indices = np.arange(len(data))
    train_validation, test = train_test_split(
        indices, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    relative_validation_size = args.validation_size / (1 - args.test_size)
    train, validation = train_test_split(
        train_validation, test_size=relative_validation_size,
        random_state=args.random_state, stratify=y[train_validation]
    )
    split_names = np.full(len(data), "", dtype=object)
    split_names[train] = "training"
    split_names[validation] = "validation"
    split_names[test] = "test"
    data["data_split"] = split_names
    print(f"Training: {len(train):,}; validation: {len(validation):,}; test: {len(test):,}")

    total = len(RADII) * len(NBITS_VALUES) * len(MAX_DEPTHS) * len(MIN_SAMPLES_SPLITS) * len(MIN_SAMPLES_LEAVES) * len(CRITERIA)
    print(f"Testing {total} models...")
    cache = {}
    records = []
    completed = 0

    for radius, nbits in product(RADII, NBITS_VALUES):
        print(f"Generating fingerprints: radius={radius}, nBits={nbits}")
        x, infos = make_fingerprints(data["mol"].tolist(), radius, nbits)
        cache[(radius, nbits)] = (x, infos)
        for depth, split, leaf, criterion in product(
            MAX_DEPTHS, MIN_SAMPLES_SPLITS, MIN_SAMPLES_LEAVES, CRITERIA
        ):
            completed += 1
            model = DecisionTreeClassifier(
                random_state=args.random_state,
                max_depth=depth,
                min_samples_split=split,
                min_samples_leaf=leaf,
                criterion=criterion,
                class_weight="balanced",
            )
            model.fit(x[train], y[train])
            tr = metrics(model, x[train], y[train])
            va = metrics(model, x[validation], y[validation])
            records.append({
                "radius": radius, "nbits": nbits, "max_depth": depth,
                "min_samples_split": split, "min_samples_leaf": leaf,
                "criterion": criterion,
                "training_balanced_accuracy": tr["balanced_accuracy"],
                "validation_balanced_accuracy": va["balanced_accuracy"],
                "validation_accuracy": va["accuracy"],
                "validation_precision": va["precision"],
                "validation_recall": va["recall"],
                "validation_f1": va["f1"],
                "validation_mcc": va["mcc"],
                "validation_roc_auc": va["roc_auc"],
            })
            if completed % 25 == 0 or completed == total:
                print(f"Completed {completed}/{total}")

    results = pd.DataFrame(records).sort_values(
        ["validation_balanced_accuracy", "validation_mcc", "validation_f1", "max_depth"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results.to_csv(args.output_dir / "hyperparameter_validation_scores.csv", index=False)
    save_top_plot(results, args.output_dir / "hyperparameter_validation_top25.png")

    best = results.iloc[0]
    params = {
        "radius": int(best.radius), "nbits": int(best.nbits),
        "max_depth": int(best.max_depth),
        "min_samples_split": int(best.min_samples_split),
        "min_samples_leaf": int(best.min_samples_leaf),
        "criterion": str(best.criterion),
    }
    print("Selected hyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    x, bit_infos = cache[(params["radius"], params["nbits"])]
    final_training = np.concatenate([train, validation])
    final_model = DecisionTreeClassifier(
        random_state=args.random_state,
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        criterion=params["criterion"],
        class_weight="balanced",
    )
    final_model.fit(x[final_training], y[final_training])
    test_metrics = metrics(final_model, x[test], y[test])
    print("Final untouched-test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    predictions = final_model.predict(x[test])
    (args.output_dir / "test_classification_report.txt").write_text(
        classification_report(y[test], predictions, digits=4, zero_division=0),
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y[test], predictions, ax=ax, cmap="Blues")
    ax.set_title("Decision tree: untouched test set")
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_confusion_matrix.png", dpi=250)
    plt.close(fig)

    importances = final_model.feature_importances_
    bits = np.flatnonzero(importances > 0)
    order = bits[np.argsort(importances[bits])[::-1]]
    feature_table = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "bit": order,
        "importance": importances[order],
        "molecules_containing_bit": [int(np.count_nonzero(x[:, bit])) for bit in order],
        "mean_count_when_present": [float(x[x[:, bit] > 0, bit].mean()) for bit in order],
    })
    feature_table.to_csv(args.output_dir / "feature_importances.csv", index=False)
    save_fragments(feature_table, data, x, bit_infos, args.output_dir, args.max_fragment_images)

    data.drop(columns=["mol"]).to_csv(args.output_dir / "molecules_labels_and_splits.csv", index=False)
    rules = export_text(final_model, feature_names=[f"bit_{i}" for i in range(params["nbits"])])
    (args.output_dir / "decision_tree_rules.txt").write_text(rules, encoding="utf-8")

    settings = {
        "input_file": str(args.input_file.resolve()),
        "fitness_threshold": args.label_threshold,
        "hyperparameter_grid": {
            "radius": RADII, "nbits": NBITS_VALUES, "max_depth": MAX_DEPTHS,
            "min_samples_split": MIN_SAMPLES_SPLITS,
            "min_samples_leaf": MIN_SAMPLES_LEAVES,
            "criterion": CRITERIA, "number_of_models": total,
        },
        "selection_metric": "validation_balanced_accuracy",
        "selected_hyperparameters": params,
        "selected_validation_balanced_accuracy": float(best.validation_balanced_accuracy),
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "random_state": args.random_state,
        "remove_gold": args.remove_gold,
        "deduplicate_smiles": args.deduplicate_smiles,
        "number_of_valid_molecules": len(data),
        "class_counts": {"0": int(class_counts[0]), "1": int(class_counts[1])},
        "test_metrics": test_metrics,
    }
    (args.output_dir / "run_settings_and_metrics.json").write_text(
        json.dumps(settings, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(f"Saved results to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
