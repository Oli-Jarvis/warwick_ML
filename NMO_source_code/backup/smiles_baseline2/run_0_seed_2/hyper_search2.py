#!/usr/bin/env python3
"""
Repeated-CV Morgan fingerprint + decision-tree hyperparameter search.

Implements:
1. Cached Morgan fingerprints for several radius/nBits combinations.
2. Repeated stratified cross-validation for tree hyperparameters.
3. Detailed per-configuration metrics and model-complexity statistics.
4. Multi-objective ranking with an overfitting-gap filter.
5. Pareto-frontier analysis for validation performance, overfitting and complexity.

The script reserves an untouched test set but DOES NOT fit or evaluate a final
model on it. That should be done only after you have inspected these outputs.

Example
-------
python decision_tree_steps_1_to_5.py full_history.csv \
    --fitness-column fitness \
    --threshold 2.0 \
    --positive-above \
    --output-dir morgan_tree_cv_results

Dependencies
------------
pandas, numpy, matplotlib, scikit-learn, rdkit
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    ParameterSampler,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier


RADIUS_VALUES = [1, 2, 3]
NBIT_VALUES = [1024, 2048, 4096, 8192]

# A deliberately broad search space. A random subset is sampled independently
# for every radius/nBits pair, avoiding an impractically large exhaustive grid.
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
        description="Repeated-CV hyperparameter search for Morgan decision trees."
    )
    parser.add_argument("csv_file", type=Path, help="Input CSV containing SMILES and fitness.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("morgan_tree_cv_results"),
        help="Directory for all outputs.",
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--fitness-column", default="fitness")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fitness threshold used to create the binary labels. Prompted if omitted.",
    )

    direction = parser.add_mutually_exclusive_group()
    direction.add_argument(
        "--positive-above",
        action="store_true",
        help="Class 1 means fitness >= threshold. This is the default.",
    )
    direction.add_argument(
        "--positive-below",
        action="store_true",
        help="Class 1 means fitness <= threshold.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction reserved as an untouched final test set.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified folds per repeat.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of complete CV repetitions.",
    )
    parser.add_argument(
        "--configs-per-fingerprint",
        type=int,
        default=100,
        help=(
            "Randomly sampled tree configurations for each radius/nBits pair. "
            "Default 100 gives 1,200 configurations and 18,000 fits with 5x3 CV."
        ),
    )
    parser.add_argument(
        "--overfit-gap-limit",
        type=float,
        default=0.05,
        help="Maximum mean train-validation balanced-accuracy gap for ranking eligibility.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Rewrite the checkpoint CSV after this many newly completed configurations.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and start the search again.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def normalise_param(value: Any) -> str:
    """Stable string representation for resume keys."""
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def configuration_key(
    radius: int,
    nbits: int,
    params: dict[str, Any],
) -> str:
    parts = [str(radius), str(nbits)]
    for name in sorted(TREE_PARAMETER_SPACE):
        parts.append(f"{name}={normalise_param(params[name])}")
    return "|".join(parts)


def load_and_label_data(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    if not args.csv_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.csv_file}")

    # Automatically detect whether the file is comma-, semicolon-, or tab-separated.
    # Your full_history.csv is semicolon-separated despite having a .csv extension.
    df = pd.read_csv(args.csv_file, sep=None, engine="python")

    required = [args.smiles_column, args.fitness_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Available columns: {list(df.columns)}"
        )

    working = df[[args.smiles_column, args.fitness_column]].copy()
    working[args.fitness_column] = pd.to_numeric(
        working[args.fitness_column], errors="coerce"
    )
    working = working.dropna(subset=[args.smiles_column, args.fitness_column])
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

    threshold = args.threshold
    if threshold is None:
        threshold = float(input("Enter the fitness threshold: ").strip())
        args.threshold = threshold

    positive_above = not args.positive_below
    fitness = working[args.fitness_column].to_numpy(dtype=float)
    if positive_above:
        labels = (fitness >= threshold).astype(np.int8)
        rule = f"{args.fitness_column} >= {threshold}"
    else:
        labels = (fitness <= threshold).astype(np.int8)
        rule = f"{args.fitness_column} <= {threshold}"

    counts = np.bincount(labels, minlength=2)
    if np.any(counts < max(args.folds, 2)):
        raise ValueError(
            "There are too few molecules in one class for the requested stratified CV. "
            f"Class counts are {counts.tolist()}."
        )

    print(f"Usable molecules: {len(working):,}")
    print(f"Class rule: class 1 when {rule}")
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

    # Decision trees only need to know whether a hashed environment is present.
    matrix = (matrix > 0).astype(np.uint8)
    np.savez_compressed(cache_file, X=matrix)
    return matrix


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def evaluate_configuration(
    X: np.ndarray,
    y: np.ndarray,
    dev_indices: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
    tree_seed: int,
) -> dict[str, float]:
    X_dev = X[dev_indices]
    y_dev = y[dev_indices]

    train_ba: list[float] = []
    val_ba: list[float] = []
    val_mcc: list[float] = []
    val_auc: list[float] = []
    node_counts: list[int] = []
    leaf_counts: list[int] = []
    actual_depths: list[int] = []

    for train_local, val_local in cv_splits:
        model = DecisionTreeClassifier(random_state=tree_seed, **params)
        model.fit(X_dev[train_local], y_dev[train_local])

        train_prediction = model.predict(X_dev[train_local])
        val_prediction = model.predict(X_dev[val_local])
        val_probability = model.predict_proba(X_dev[val_local])[:, 1]

        train_ba.append(
            balanced_accuracy_score(y_dev[train_local], train_prediction)
        )
        val_ba.append(
            balanced_accuracy_score(y_dev[val_local], val_prediction)
        )
        val_mcc.append(matthews_corrcoef(y_dev[val_local], val_prediction))
        val_auc.append(safe_roc_auc(y_dev[val_local], val_probability))
        node_counts.append(model.tree_.node_count)
        leaf_counts.append(model.get_n_leaves())
        actual_depths.append(model.get_depth())

    train_arr = np.asarray(train_ba, dtype=float)
    val_arr = np.asarray(val_ba, dtype=float)

    return {
        "mean_training_balanced_accuracy": float(np.mean(train_arr)),
        "sd_training_balanced_accuracy": float(np.std(train_arr, ddof=1)),
        "mean_validation_balanced_accuracy": float(np.mean(val_arr)),
        "sd_validation_balanced_accuracy": float(np.std(val_arr, ddof=1)),
        "mean_validation_mcc": float(np.mean(val_mcc)),
        "sd_validation_mcc": float(np.std(val_mcc, ddof=1)),
        "mean_validation_roc_auc": float(np.nanmean(val_auc)),
        "sd_validation_roc_auc": float(np.nanstd(val_auc, ddof=1)),
        "mean_generalisation_gap": float(np.mean(train_arr - val_arr)),
        "sd_generalisation_gap": float(np.std(train_arr - val_arr, ddof=1)),
        "mean_node_count": float(np.mean(node_counts)),
        "sd_node_count": float(np.std(node_counts, ddof=1)),
        "mean_leaf_count": float(np.mean(leaf_counts)),
        "sd_leaf_count": float(np.std(leaf_counts, ddof=1)),
        "mean_actual_depth": float(np.mean(actual_depths)),
        "sd_actual_depth": float(np.std(actual_depths, ddof=1)),
    }


def is_pareto_efficient(values: np.ndarray) -> np.ndarray:
    """
    Return mask for three objectives already expressed as minimisation:
    column 0 = -validation BA, column 1 = gap, column 2 = node count.
    """
    n = values.shape[0]
    efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not efficient[i]:
            continue
        dominated_by_any = np.any(
            np.all(values <= values[i], axis=1)
            & np.any(values < values[i], axis=1)
        )
        if dominated_by_any:
            efficient[i] = False

    return efficient


def rank_results(results: pd.DataFrame, gap_limit: float) -> pd.DataFrame:
    ranked = results.copy()
    ranked["eligible_under_gap_limit"] = (
        ranked["mean_generalisation_gap"] <= gap_limit
    )

    if ranked["eligible_under_gap_limit"].any():
        eligibility_group = ranked["eligible_under_gap_limit"].astype(int)
    else:
        print(
            "Warning: no model met the overfitting-gap limit. "
            "Ranking all models without the gap filter."
        )
        eligibility_group = pd.Series(np.ones(len(ranked), dtype=int), index=ranked.index)

    ranked["_eligibility_sort"] = eligibility_group

    ranked = ranked.sort_values(
        by=[
            "_eligibility_sort",
            "mean_validation_balanced_accuracy",
            "mean_validation_mcc",
            "sd_validation_balanced_accuracy",
            "mean_node_count",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    ranked["multi_objective_rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected_by_ranking"] = ranked["multi_objective_rank"].eq(1)
    ranked = ranked.drop(columns="_eligibility_sort")

    objectives = np.column_stack(
        [
            -ranked["mean_validation_balanced_accuracy"].to_numpy(),
            ranked["mean_generalisation_gap"].to_numpy(),
            ranked["mean_node_count"].to_numpy(),
        ]
    )
    ranked["pareto_optimal"] = is_pareto_efficient(objectives)
    return ranked


def save_top_models_plot(ranked: pd.DataFrame, output_path: Path, top_n: int = 25) -> None:
    subset = ranked.head(min(top_n, len(ranked))).iloc[::-1]
    labels = [
        f"R{r.radius}/B{r.nbits}/D{r.max_depth}"
        for r in subset.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.30 * len(subset))))
    ax.barh(labels, subset["mean_validation_balanced_accuracy"])
    ax.set_xlabel("Mean validation balanced accuracy")
    ax.set_ylabel("Configuration")
    ax.set_title(f"Top {len(subset)} configurations after multi-objective ranking")
    ax.set_xlim(
        max(0.0, subset["mean_validation_balanced_accuracy"].min() - 0.02),
        min(1.0, subset["mean_validation_balanced_accuracy"].max() + 0.01),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_pareto_gap_plot(ranked: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    non_pareto = ranked.loc[~ranked["pareto_optimal"]]
    pareto = ranked.loc[ranked["pareto_optimal"]]

    ax.scatter(
        non_pareto["mean_generalisation_gap"],
        non_pareto["mean_validation_balanced_accuracy"],
        alpha=0.35,
        label="Other configurations",
    )
    ax.scatter(
        pareto["mean_generalisation_gap"],
        pareto["mean_validation_balanced_accuracy"],
        marker="x",
        s=70,
        label="Pareto-optimal",
    )
    ax.set_xlabel("Mean training–validation balanced-accuracy gap")
    ax.set_ylabel("Mean validation balanced accuracy")
    ax.set_title("Performance–overfitting trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_pareto_complexity_plot(ranked: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    non_pareto = ranked.loc[~ranked["pareto_optimal"]]
    pareto = ranked.loc[ranked["pareto_optimal"]]

    ax.scatter(
        non_pareto["mean_node_count"],
        non_pareto["mean_validation_balanced_accuracy"],
        alpha=0.35,
        label="Other configurations",
    )
    ax.scatter(
        pareto["mean_node_count"],
        pareto["mean_validation_balanced_accuracy"],
        marker="x",
        s=70,
        label="Pareto-optimal",
    )
    ax.set_xlabel("Mean number of tree nodes")
    ax.set_ylabel("Mean validation balanced accuracy")
    ax.set_title("Performance–complexity trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


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
    split_table = pd.DataFrame(
        {
            "row_index": np.arange(len(data)),
            args.smiles_column: data[args.smiles_column],
            args.fitness_column: data[args.fitness_column],
            "class_label": labels,
            "split": split_labels,
        }
    )
    split_table.to_csv(args.output_dir / "molecules_labels_and_outer_split.csv", index=False)

    cv = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.cv_seed,
    )
    dev_labels = labels[dev_indices]
    cv_splits = list(cv.split(np.zeros(len(dev_indices)), dev_labels))

    print(
        f"Development set: {len(dev_indices):,}; untouched test set: {len(test_indices):,}"
    )
    print(
        f"Cross-validation: {args.folds} folds × {args.repeats} repeats "
        f"= {len(cv_splits)} fits per configuration."
    )

    existing_results: list[dict[str, Any]] = []
    completed_keys: set[str] = set()

    if checkpoint_path.exists() and not args.no_resume:
        checkpoint_df = pd.read_csv(checkpoint_path)
        existing_results = checkpoint_df.to_dict("records")
        if "configuration_key" in checkpoint_df.columns:
            completed_keys = set(checkpoint_df["configuration_key"].astype(str))
        print(f"Resuming from {len(existing_results):,} completed configurations.")

    sampled_configs = list(
        ParameterSampler(
            TREE_PARAMETER_SPACE,
            n_iter=args.configs_per_fingerprint,
            random_state=args.cv_seed,
        )
    )

    total_requested = (
        len(RADIUS_VALUES) * len(NBIT_VALUES) * len(sampled_configs)
    )
    print(f"Requested configurations: {total_requested:,}")
    print(f"Expected tree fits: {total_requested * len(cv_splits):,}")

    results = list(existing_results)
    newly_completed = 0
    search_start = time.time()

    for radius, nbits in itertools.product(RADIUS_VALUES, NBIT_VALUES):
        X = create_or_load_fingerprints(
            molecules=molecules,
            radius=radius,
            nbits=nbits,
            cache_dir=cache_dir,
        )

        for params in sampled_configs:
            params = dict(params)
            key = configuration_key(radius, nbits, params)
            if key in completed_keys:
                continue

            metrics = evaluate_configuration(
                X=X,
                y=labels,
                dev_indices=dev_indices,
                cv_splits=cv_splits,
                params=params,
                tree_seed=args.cv_seed,
            )

            row: dict[str, Any] = {
                "configuration_key": key,
                "radius": radius,
                "nbits": nbits,
                **params,
                **metrics,
            }
            results.append(row)
            completed_keys.add(key)
            newly_completed += 1

            done = len(completed_keys)
            if done == 1 or done % 25 == 0 or done == total_requested:
                elapsed = time.time() - search_start
                rate = newly_completed / elapsed if elapsed > 0 else 0.0
                remaining = max(total_requested - done, 0)
                eta = remaining / rate if rate > 0 else float("nan")
                print(
                    f"Completed {done:,}/{total_requested:,} configurations "
                    f"({done * len(cv_splits):,} fits); "
                    f"estimated remaining time: {eta / 60:.1f} min"
                )

            if newly_completed % args.checkpoint_every == 0:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    raw_results = pd.DataFrame(results)
    raw_results.to_csv(checkpoint_path, index=False)

    if len(raw_results) != total_requested:
        print(
            f"Warning: expected {total_requested:,} unique configurations but have "
            f"{len(raw_results):,}. Check the checkpoint for duplicates or changed settings."
        )

    ranked = rank_results(raw_results, args.overfit_gap_limit)

    # Put ranking columns first.
    preferred_order = [
        "multi_objective_rank",
        "selected_by_ranking",
        "pareto_optimal",
        "eligible_under_gap_limit",
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
        "mean_validation_mcc",
        "sd_validation_mcc",
        "mean_validation_roc_auc",
        "sd_validation_roc_auc",
        "mean_generalisation_gap",
        "sd_generalisation_gap",
        "mean_actual_depth",
        "sd_actual_depth",
        "mean_node_count",
        "sd_node_count",
        "mean_leaf_count",
        "sd_leaf_count",
        "configuration_key",
    ]
    remaining_columns = [c for c in ranked.columns if c not in preferred_order]
    ranked = ranked[preferred_order + remaining_columns]

    ranked_path = args.output_dir / "hyperparameter_repeated_cv_scores.csv"
    ranked.to_csv(ranked_path, index=False)

    pareto = ranked.loc[ranked["pareto_optimal"]].copy()
    pareto = pareto.sort_values(
        ["mean_validation_balanced_accuracy", "mean_generalisation_gap"],
        ascending=[False, True],
    )
    pareto.to_csv(args.output_dir / "pareto_optimal_models.csv", index=False)

    top25 = ranked.head(25)
    top25.to_csv(args.output_dir / "top_25_ranked_models.csv", index=False)

    best = ranked.iloc[0]
    best_settings = {
        "selection_method": (
            "First require mean_generalisation_gap <= overfit_gap_limit; then sort by "
            "mean validation balanced accuracy (descending), mean validation MCC "
            "(descending), validation BA standard deviation (ascending), and mean "
            "node count (ascending)."
        ),
        "overfit_gap_limit": args.overfit_gap_limit,
        "radius": int(best["radius"]),
        "nbits": int(best["nbits"]),
        "max_depth": json_safe(best["max_depth"]),
        "min_samples_split": int(best["min_samples_split"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "criterion": str(best["criterion"]),
        "max_features": json_safe(best["max_features"]),
        "ccp_alpha": float(best["ccp_alpha"]),
        "mean_validation_balanced_accuracy": float(
            best["mean_validation_balanced_accuracy"]
        ),
        "sd_validation_balanced_accuracy": float(
            best["sd_validation_balanced_accuracy"]
        ),
        "mean_validation_mcc": float(best["mean_validation_mcc"]),
        "mean_validation_roc_auc": float(best["mean_validation_roc_auc"]),
        "mean_generalisation_gap": float(best["mean_generalisation_gap"]),
        "mean_node_count": float(best["mean_node_count"]),
        "pareto_optimal": bool(best["pareto_optimal"]),
    }
    with open(args.output_dir / "selected_model_settings.json", "w") as handle:
        json.dump(best_settings, handle, indent=2)

    run_settings = {
        "input_csv": str(args.csv_file.resolve()),
        "smiles_column": args.smiles_column,
        "fitness_column": args.fitness_column,
        "threshold": args.threshold,
        "class_1_rule": (
            f"{args.fitness_column} <= {args.threshold}"
            if args.positive_below
            else f"{args.fitness_column} >= {args.threshold}"
        ),
        "radius_values": RADIUS_VALUES,
        "nbits_values": NBIT_VALUES,
        "tree_parameter_space": TREE_PARAMETER_SPACE,
        "sampled_tree_configurations_per_fingerprint": args.configs_per_fingerprint,
        "total_evaluated_configurations": int(len(ranked)),
        "cv_folds": args.folds,
        "cv_repeats": args.repeats,
        "fits_per_configuration": len(cv_splits),
        "test_size": args.test_size,
        "development_molecules": int(len(dev_indices)),
        "untouched_test_molecules": int(len(test_indices)),
        "split_seed": args.split_seed,
        "cv_seed": args.cv_seed,
        "overfit_gap_limit": args.overfit_gap_limit,
        "note": (
            "The untouched test set was reserved but not used to fit, rank or evaluate "
            "a final model in this steps-1-to-5 script."
        ),
    }
    with open(args.output_dir / "run_settings.json", "w") as handle:
        json.dump(run_settings, handle, indent=2)

    save_top_models_plot(
        ranked,
        args.output_dir / "top_25_ranked_models.png",
    )
    save_pareto_gap_plot(
        ranked,
        args.output_dir / "pareto_validation_vs_overfitting.png",
    )
    save_pareto_complexity_plot(
        ranked,
        args.output_dir / "pareto_validation_vs_complexity.png",
    )

    print("\nSelected configuration from steps 1–5:")
    for key, value in best_settings.items():
        if key != "selection_method":
            print(f"  {key}: {value}")

    print("\nSaved:")
    for path in [
        ranked_path,
        args.output_dir / "pareto_optimal_models.csv",
        args.output_dir / "top_25_ranked_models.csv",
        args.output_dir / "selected_model_settings.json",
        args.output_dir / "run_settings.json",
        args.output_dir / "molecules_labels_and_outer_split.csv",
        args.output_dir / "top_25_ranked_models.png",
        args.output_dir / "pareto_validation_vs_overfitting.png",
        args.output_dir / "pareto_validation_vs_complexity.png",
        checkpoint_path,
    ]:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. The most recent checkpoint can be used to resume.", file=sys.stderr)
        raise
