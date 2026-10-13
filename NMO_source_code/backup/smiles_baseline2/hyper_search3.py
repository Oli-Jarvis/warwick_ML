#!/usr/bin/env python3
"""
Minimal repeated-CV Morgan fingerprint + decision-tree hyperparameter search.

The script:
1. Reserves an untouched test set.
2. Generates or loads cached Morgan fingerprints.
3. evaluates randomly sampled decision-tree configurations using repeated
   stratified cross-validation on the development set.
4. Saves every configuration ordered only by mean validation balanced accuracy.

The untouched test set is not used for fitting, ranking, or model selection.

Example
-------
python hyper_search_all_runs.py smiles_baseline2 \
    --fitness-column fitness \
    --threshold 0.5 \
    --positive-above \
    --output-dir morgan_tree_cv_results

Dependencies
------------
pandas, numpy, scikit-learn, rdkit
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    ParameterSampler,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier


RADIUS_VALUES = [1, 2, 3]
NBIT_VALUES = [1024, 2048, 4096, 8192]

TREE_PARAMETER_SPACE: dict[str, list[Any]] = {
    "max_depth": [3, 5, 7, 9, 12, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_features": [None, "sqrt", "log2", 0.25, 0.5, 0.75],
    "ccp_alpha": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal repeated-CV search for Morgan decision trees."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Either one full_history.csv file or a directory such as "
            "smiles_baseline2 containing run_*/full_history.csv files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("morgan_tree_cv_results_all_runs"),
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--fitness-column", default="fitness")
    parser.add_argument("--threshold", type=float, default=None)

    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--positive-above", action="store_true")
    direction.add_argument("--positive-below", action="store_true")

    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configs-per-fingerprint", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def normalise_param(value: Any) -> str:
    """Return a stable parameter representation for checkpoint keys."""
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def configuration_key(radius: int, nbits: int, params: dict[str, Any]) -> str:
    parts = [str(radius), str(nbits)]
    for name in sorted(TREE_PARAMETER_SPACE):
        parts.append(f"{name}={normalise_param(params[name])}")
    return "|".join(parts)


def find_input_files(input_path: Path) -> list[Path]:
    """Return one input CSV or every run_*/full_history.csv below a directory."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.glob("run_*/full_history.csv"))
    if not files:
        files = sorted(input_path.rglob("full_history.csv"))
    if not files:
        raise FileNotFoundError(
            f"No full_history.csv files found below: {input_path}"
        )
    return files


def load_and_label_data(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    input_files = find_input_files(args.input_path)
    print(f"Found {len(input_files):,} input file(s):")

    frames: list[pd.DataFrame] = []
    for csv_file in input_files:
        print(f"  {csv_file}")
        frame = pd.read_csv(csv_file, sep=None, engine="python")

        required = [args.smiles_column, args.fitness_column]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s) {missing} in {csv_file}. "
                f"Available columns: {list(frame.columns)}"
            )

        frame = frame[[args.smiles_column, args.fitness_column]].copy()
        frame["source_file"] = str(csv_file)
        frames.append(frame)

    working = pd.concat(frames, ignore_index=True)
    print(f"Combined rows before cleaning: {len(working):,}")

    required = [args.smiles_column, args.fitness_column]
    working[args.fitness_column] = pd.to_numeric(
        working[args.fitness_column], errors="coerce"
    )
    working = working.dropna(subset=required)
    working[args.smiles_column] = working[args.smiles_column].astype(str)

    valid_molecules: list[Chem.Mol] = []
    valid_rows: list[int] = []
    for idx, smiles in enumerate(working[args.smiles_column]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_molecules.append(mol)
            valid_rows.append(idx)

    dropped = len(working) - len(valid_rows)
    if dropped:
        print(f"Dropped {dropped:,} rows containing invalid SMILES.")

    working = working.iloc[valid_rows].reset_index(drop=True)
    working["_mol"] = valid_molecules

    if args.threshold is None:
        args.threshold = float(input("Enter the fitness threshold: ").strip())

    fitness = working[args.fitness_column].to_numpy(dtype=float)
    if args.positive_below:
        labels = (fitness <= args.threshold).astype(np.int8)
        rule = f"{args.fitness_column} <= {args.threshold}"
    else:
        labels = (fitness >= args.threshold).astype(np.int8)
        rule = f"{args.fitness_column} >= {args.threshold}"

    counts = np.bincount(labels, minlength=2)
    if np.any(counts < max(args.folds, 2)):
        raise ValueError(
            "Too few molecules in one class for stratified CV. "
            f"Class counts: {counts.tolist()}"
        )

    print(f"Usable molecules across all runs: {len(working):,}")
    print(f"Class 1 rule: {rule}")
    print(f"Class 0: {counts[0]:,}; class 1: {counts[1]:,}")
    return working, labels


def create_or_load_fingerprints(
    molecules: list[Chem.Mol],
    radius: int,
    nbits: int,
    cache_dir: Path,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"morgan_radius_{radius}_nbits_{nbits}.npz"

    if cache_file.exists():
        loaded = np.load(cache_file)
        matrix = loaded["X"]
        if matrix.shape == (len(molecules), nbits):
            print(f"Loaded cached fingerprints: {cache_file.name}")
            return matrix
        print(f"Ignoring incompatible cache file: {cache_file.name}")

    print(f"Generating fingerprints: radius={radius}, nBits={nbits}")
    matrix = np.zeros((len(molecules), nbits), dtype=np.uint8)

    for i, mol in enumerate(molecules):
        sparse_fp = GetHashedMorganFingerprint(
            mol,
            radius,
            nBits=nbits,
            useFeatures=False,
        )
        DataStructs.ConvertToNumpyArray(sparse_fp, matrix[i])

    matrix = (matrix > 0).astype(np.uint8)
    np.savez_compressed(cache_file, X=matrix)
    return matrix


def evaluate_configuration(
    X: np.ndarray,
    y: np.ndarray,
    dev_indices: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
    tree_seed: int,
) -> dict[str, float]:
    """Evaluate one configuration without changing the original CV behaviour."""
    X_dev = X[dev_indices]
    y_dev = y[dev_indices]

    training_scores: list[float] = []
    validation_scores: list[float] = []
    node_counts: list[int] = []

    for train_local, validation_local in cv_splits:
        model = DecisionTreeClassifier(random_state=tree_seed, **params)
        model.fit(X_dev[train_local], y_dev[train_local])

        training_scores.append(
            balanced_accuracy_score(
                y_dev[train_local], model.predict(X_dev[train_local])
            )
        )
        validation_scores.append(
            balanced_accuracy_score(
                y_dev[validation_local], model.predict(X_dev[validation_local])
            )
        )
        node_counts.append(model.tree_.node_count)

    training = np.asarray(training_scores, dtype=float)
    validation = np.asarray(validation_scores, dtype=float)
    nodes = np.asarray(node_counts, dtype=float)

    return {
        "mean_training_balanced_accuracy": float(np.mean(training)),
        "sd_training_balanced_accuracy": float(np.std(training, ddof=1)),
        "mean_validation_balanced_accuracy": float(np.mean(validation)),
        "sd_validation_balanced_accuracy": float(np.std(validation, ddof=1)),
        "mean_node_count": float(np.mean(nodes)),
        "sd_node_count": float(np.std(nodes, ddof=1)),
    }


def main() -> None:
    args = parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.configs_per_fingerprint < 1:
        raise ValueError("--configs-per-fingerprint must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "fingerprint_cache"
    checkpoint_path = args.output_dir / "hyperparameter_cv_checkpoint.csv"

    data, labels = load_and_label_data(args)
    molecules = data["_mol"].tolist()

    all_indices = np.arange(len(data))
    dev_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.split_seed,
        stratify=labels,
    )

    split_labels = np.full(len(data), "development", dtype=object)
    split_labels[test_indices] = "untouched_test"
    pd.DataFrame(
        {
            "row_index": np.arange(len(data)),
            args.smiles_column: data[args.smiles_column],
            args.fitness_column: data[args.fitness_column],
            "source_file": data["source_file"],
            "class_label": labels,
            "split": split_labels,
        }
    ).to_csv(
        args.output_dir / "molecules_labels_and_outer_split.csv",
        index=False,
    )

    cv = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.cv_seed,
    )
    cv_splits = list(
        cv.split(np.zeros(len(dev_indices)), labels[dev_indices])
    )

    print(
        f"Development set: {len(dev_indices):,}; "
        f"untouched test set: {len(test_indices):,}"
    )
    print(
        f"Cross-validation: {args.folds} folds × {args.repeats} repeats "
        f"= {len(cv_splits)} fits per configuration."
    )

    results: list[dict[str, Any]] = []
    completed_keys: set[str] = set()

    if checkpoint_path.exists() and not args.no_resume:
        checkpoint = pd.read_csv(checkpoint_path)
        results = checkpoint.to_dict("records")
        if "configuration_key" in checkpoint.columns:
            completed_keys = set(checkpoint["configuration_key"].astype(str))
        print(f"Resuming from {len(completed_keys):,} completed configurations.")

    sampled_configs = list(
        ParameterSampler(
            TREE_PARAMETER_SPACE,
            n_iter=args.configs_per_fingerprint,
            random_state=args.cv_seed,
        )
    )

    total_requested = len(RADIUS_VALUES) * len(NBIT_VALUES) * len(sampled_configs)
    print(f"Requested configurations: {total_requested:,}")
    print(f"Expected tree fits: {total_requested * len(cv_splits):,}")

    newly_completed = 0
    search_start = time.time()

    for radius, nbits in itertools.product(RADIUS_VALUES, NBIT_VALUES):
        X = create_or_load_fingerprints(
            molecules=molecules,
            radius=radius,
            nbits=nbits,
            cache_dir=cache_dir,
        )

        for sampled_params in sampled_configs:
            params = dict(sampled_params)
            key = configuration_key(radius, nbits, params)
            if key in completed_keys:
                continue

            row = {
                "configuration_key": key,
                "radius": radius,
                "nbits": nbits,
                **params,
                **evaluate_configuration(
                    X=X,
                    y=labels,
                    dev_indices=dev_indices,
                    cv_splits=cv_splits,
                    params=params,
                    tree_seed=args.cv_seed,
                ),
            }
            results.append(row)
            completed_keys.add(key)
            newly_completed += 1

            done = len(completed_keys)
            if done == 1 or done % 25 == 0 or done == total_requested:
                elapsed = time.time() - search_start
                rate = newly_completed / elapsed if elapsed > 0 else 0.0
                remaining = max(total_requested - done, 0)
                eta_minutes = remaining / rate / 60 if rate > 0 else float("nan")
                print(
                    f"Completed {done:,}/{total_requested:,} configurations; "
                    f"estimated remaining time: {eta_minutes:.1f} min"
                )

            if newly_completed % args.checkpoint_every == 0:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    raw_results = pd.DataFrame(results)
    raw_results.to_csv(checkpoint_path, index=False)

    if len(raw_results) != total_requested:
        print(
            f"Warning: expected {total_requested:,} unique configurations but found "
            f"{len(raw_results):,}. Check whether the checkpoint came from a different run."
        )

    # Rank only by mean validation balanced accuracy, highest first.
    ranked = raw_results.sort_values(
        "mean_validation_balanced_accuracy",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "validation_accuracy_rank", np.arange(1, len(ranked) + 1))

    output_columns = [
        "validation_accuracy_rank",
        "radius",
        "nbits",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "criterion",
        "max_features",
        "ccp_alpha",
        "mean_training_balanced_accuracy",
        "sd_training_balanced_accuracy",
        "mean_validation_balanced_accuracy",
        "sd_validation_balanced_accuracy",
        "mean_node_count",
        "sd_node_count",
        "configuration_key",
    ]
    ranked = ranked[output_columns]

    ranked_path = args.output_dir / "all_configurations_by_validation_accuracy.csv"
    ranked.to_csv(ranked_path, index=False)

    print("\nHighest-validation configuration:")
    for column in output_columns:
        if column != "configuration_key":
            print(f"  {column}: {ranked.iloc[0][column]}")

    print("\nSaved:")
    print(f"  {ranked_path.resolve()}")
    print(f"  {checkpoint_path.resolve()}")
    print(
        f"  {(args.output_dir / 'molecules_labels_and_outer_split.csv').resolve()}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Re-run the same command to resume from the checkpoint.",
            file=sys.stderr,
        )
        raise
