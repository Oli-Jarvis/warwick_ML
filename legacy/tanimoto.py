#!/usr/bin/env python3
"""Compare FragNet prediction errors with Morgan-fingerprint Tanimoto similarity.

Run from the parent of fragnet_combined_selected_stereo:

    python tanimoto.py

Or point to the FragNet prediction tables explicitly:

    python fragnet_tanimoto_similarity.py \
        --predictions-dir fragnet_log_P_upconversion_regression/xtb_fragnet_evaluation

Uses the prediction CSVs saved alongside the current experiment/ft.pt checkpoint.
Each test molecule is compared against the current run's training molecules.
Each training molecule is compared against the other training molecules, excluding
itself and any duplicate with the same canonical SMILES.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_RESULTS_DIR = Path("fragnet_combined_selected_stereo")
SPLIT_COLOURS = {"train": "#2563eb", "test": "#e99658"}
TRUE_COLUMNS = ("xtb_value", "true", "y_true", "target", "actual", "label", "log_P_upconversion", "y")
PREDICTION_COLUMNS = ("fragnet_prediction", "predicted", "prediction", "y_pred", "pred", "prediction_log_P_upconversion")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate nearest-training-molecule Tanimoto similarities and plot "
            "FragNet absolute prediction error against top-1 and mean top-k similarity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="FragNet results directory used to resolve prediction-table locations.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory containing train_predictions.csv and test_predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to PREDICTIONS_DIR/tanimoto_similarity.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=["test"],
        help="Splits to analyse; the training split remains the similarity reference.",
    )
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint length.")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of nearest neighbours to average."
    )
    parser.add_argument(
        "--include-chirality", action=argparse.BooleanOptionalAction,
        default=True, help="Include chirality in fingerprints."
    )
    parser.add_argument(
        "--similarity-bins",
        type=int,
        default=10,
        help="Number of equal-width similarity bins for summary statistics.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=500, help="Progress-reporting interval."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Resolution of PNG plots.")
    args = parser.parse_args(argv)

    for name in ("radius", "n_bits", "top_k", "similarity_bins", "progress_every", "dpi"):
        minimum = 0 if name == "radius" else 1
        if getattr(args, name) < minimum:
            parser.error(f"--{name.replace('_', '-')} must be at least {minimum}.")
    args.splits = list(dict.fromkeys(args.splits))
    return args


def resolve_predictions_dir(args: argparse.Namespace) -> Path:
    if args.predictions_dir is not None:
        return args.predictions_dir.expanduser().resolve()

    results_dir = args.results_dir.expanduser().resolve()
    candidates = (
        results_dir / "experiment",
        results_dir / "xtb_fragnet_evaluation",
        results_dir / "train_test_evaluation",
    )
    for candidate in candidates:
        if (candidate / "train_predictions.csv").is_file() and all(
            (candidate / f"{split}_predictions.csv").is_file() for split in args.splits
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find existing FragNet prediction tables. Expected "
        "train_predictions.csv and the requested split prediction CSVs under "
        f"{', '.join(map(str, candidates))}. Run the FragNet evaluator first, "
        "or pass --predictions-dir /path/to/the/prediction/directory."
    )


def choose_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(
        f"Prediction table has no {label} column. Expected one of "
        f"{', '.join(candidates)}; available columns: {', '.join(frame.columns)}."
    )


def read_prediction_table(path: Path, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} prediction table: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"The {split} prediction table is empty: {path}")
    if "smiles" not in frame.columns:
        raise ValueError(f"The {split} prediction table has no 'smiles' column: {path}")

    true_column = choose_column(frame, TRUE_COLUMNS, "original xTB target")
    prediction_column = choose_column(frame, PREDICTION_COLUMNS, "FragNet prediction")

    xtb_values = pd.to_numeric(frame[true_column], errors="coerce").to_numpy(dtype=float)
    predictions = pd.to_numeric(frame[prediction_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(xtb_values).all() or not np.isfinite(predictions).all():
        raise ValueError(f"The {split} prediction table contains missing or non-finite values.")

    result = frame.copy()
    result["split"] = split
    result["xtb_value"] = xtb_values
    result["fragnet_prediction"] = predictions
    result["residual"] = predictions - xtb_values
    result["absolute_error"] = np.abs(result["residual"].to_numpy(dtype=float))
    result["squared_error"] = np.square(result["residual"].to_numpy(dtype=float))
    return result


def load_rdkit(args: argparse.Namespace) -> tuple[Any, Any, Callable[[Any], Any]]:
    try:
        from rdkit import Chem, DataStructs
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is not installed in the active Python environment. Activate the "
            "FragNet/chemistry environment that contains RDKit before running this script."
        ) from exc

    try:
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=args.radius,
            fpSize=args.n_bits,
            includeChirality=args.include_chirality,
        )
        fingerprint = generator.GetFingerprint
    except (ImportError, AttributeError):
        from rdkit.Chem import AllChem

        def fingerprint(molecule: Any) -> Any:
            return AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                radius=args.radius,
                nBits=args.n_bits,
                useChirality=args.include_chirality,
            )

    return Chem, DataStructs, fingerprint


def fingerprint_table(
    frame: pd.DataFrame,
    *,
    split: str,
    chemistry: Any,
    fingerprint: Callable[[Any], Any],
) -> tuple[pd.DataFrame, list[Any]]:
    fingerprints: list[Any] = []
    selected_smiles: list[str] = []
    canonical_smiles: list[str] = []
    failures: list[tuple[int, str]] = []
    source_smiles = frame["xtb_source_smiles"] if "xtb_source_smiles" in frame else None

    for index, graph_smiles in enumerate(frame["smiles"].tolist()):
        candidates = [graph_smiles]
        if source_smiles is not None:
            candidates.append(source_smiles.iloc[index])

        molecule = None
        chosen = ""
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            chosen = candidate.strip()
            try:
                molecule = chemistry.MolFromSmiles(chosen)
            except Exception:
                molecule = None
            if molecule is not None:
                break

        if molecule is None:
            failures.append((index, str(graph_smiles)))
            continue

        try:
            canonical = chemistry.MolToSmiles(molecule, canonical=True)
            molecular_fingerprint = fingerprint(molecule)
        except Exception as exc:
            failures.append((index, f"{chosen} ({exc})"))
            continue

        selected_smiles.append(chosen)
        canonical_smiles.append(canonical)
        fingerprints.append(molecular_fingerprint)

    if failures:
        examples = "; ".join(f"row {index}: {value}" for index, value in failures[:5])
        raise ValueError(
            f"Could not calculate Morgan fingerprints for {len(failures):,} {split} "
            f"molecules. Examples: {examples}."
        )

    prepared = frame.copy()
    prepared["fingerprint_smiles"] = selected_smiles
    prepared["canonical_smiles"] = canonical_smiles
    return prepared, fingerprints


def top_neighbour_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    available = int(np.isfinite(scores).sum())
    if available == 0:
        raise ValueError("No eligible training neighbours remain after self-exclusion.")
    count = min(top_k, available)
    indices = np.argpartition(-scores, count - 1)[:count]
    order = np.lexsort((indices, -scores[indices]))
    return indices[order]


def calculate_similarities(
    *,
    split: str,
    frame: pd.DataFrame,
    fingerprints: list[Any],
    train_frame: pd.DataFrame,
    train_fingerprints: list[Any],
    data_structs: Any,
    top_k: int,
    progress_every: int,
) -> pd.DataFrame:
    train_indices_by_smiles: dict[str, list[int]] = defaultdict(list)
    for index, canonical in enumerate(train_frame["canonical_smiles"].tolist()):
        train_indices_by_smiles[canonical].append(index)

    if split == "train" and len(train_fingerprints) < 2:
        raise ValueError("Training-set self-exclusion requires at least two training molecules.")

    top1_similarities: list[float] = []
    mean_similarities: list[float] = []
    neighbour_counts: list[int] = []
    nearest_indices: list[int] = []
    nearest_smiles: list[str] = []
    nearest_targets: list[float] = []
    nearest_predictions: list[float] = []
    neighbour_smiles: list[str] = []
    neighbour_scores: list[str] = []
    exact_matches: list[bool] = []

    train_smiles = train_frame["fingerprint_smiles"].tolist()
    train_xtb = train_frame["xtb_value"].to_numpy(dtype=float)
    train_predicted = train_frame["fragnet_prediction"].to_numpy(dtype=float)
    total = len(frame)

    for index, (molecular_fingerprint, canonical) in enumerate(
        zip(fingerprints, frame["canonical_smiles"].tolist()), start=1
    ):
        scores = np.asarray(
            data_structs.BulkTanimotoSimilarity(molecular_fingerprint, train_fingerprints),
            dtype=float,
        )
        if scores.shape != (len(train_fingerprints),):
            raise ValueError(
                f"Tanimoto calculation returned {len(scores):,} scores for "
                f"{len(train_fingerprints):,} training molecules."
            )

        matching_indices = train_indices_by_smiles.get(canonical, [])
        exact_matches.append(bool(matching_indices) if split != "train" else False)
        if split == "train":
            scores[np.asarray(matching_indices, dtype=int)] = -np.inf

        best_indices = top_neighbour_indices(scores, top_k)
        best_scores = scores[best_indices]
        nearest = int(best_indices[0])

        top1_similarities.append(float(best_scores[0]))
        mean_similarities.append(float(np.mean(best_scores)))
        neighbour_counts.append(int(len(best_indices)))
        nearest_indices.append(nearest)
        nearest_smiles.append(train_smiles[nearest])
        nearest_targets.append(float(train_xtb[nearest]))
        nearest_predictions.append(float(train_predicted[nearest]))
        neighbour_smiles.append(json.dumps([train_smiles[int(item)] for item in best_indices]))
        neighbour_scores.append(json.dumps([float(value) for value in best_scores]))

        if index % progress_every == 0 or index == total:
            print(f"  {split}: {index:,}/{total:,} molecules processed", flush=True)

    result = frame.copy()
    result["top1_similarity"] = top1_similarities
    result[f"top{top_k}_mean_similarity"] = mean_similarities
    result[f"top{top_k}_neighbour_count"] = neighbour_counts
    result["top1_training_index"] = nearest_indices
    result["top1_training_smiles"] = nearest_smiles
    result["top1_training_xtb_value"] = nearest_targets
    result["top1_training_fragnet_prediction"] = nearest_predictions
    result[f"top{top_k}_training_smiles"] = neighbour_smiles
    result[f"top{top_k}_similarities"] = neighbour_scores
    result["exact_training_molecule_match"] = exact_matches
    return result


def safe_correlation(
    x: np.ndarray, y: np.ndarray, method: str
) -> tuple[float | None, float | None]:
    if len(x) < 2 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return None, None
    calculation = stats.pearsonr if method == "pearson" else stats.spearmanr
    statistic, p_value = calculation(x, y)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        return None, None
    return float(statistic), float(p_value)


def calculate_error_metrics(
    frame: pd.DataFrame, *, split: str, similarity_column: str
) -> dict[str, Any]:
    similarities = frame[similarity_column].to_numpy(dtype=float)
    absolute_errors = frame["absolute_error"].to_numpy(dtype=float)
    squared_errors = frame["squared_error"].to_numpy(dtype=float)
    pearson, pearson_p = safe_correlation(similarities, absolute_errors, "pearson")
    spearman, spearman_p = safe_correlation(similarities, absolute_errors, "spearman")
    return {
        "split": split,
        "similarity_measure": similarity_column,
        "n": int(len(frame)),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "similarity_mean": float(np.mean(similarities)),
        "similarity_median": float(np.median(similarities)),
        "similarity_min": float(np.min(similarities)),
        "similarity_max": float(np.max(similarities)),
        "error_similarity_pearson_r": pearson,
        "error_similarity_pearson_p": pearson_p,
        "error_similarity_spearman_rho": spearman,
        "error_similarity_spearman_p": spearman_p,
        "exact_training_molecule_matches": int(
            frame["exact_training_molecule_match"].sum()
        ),
    }


def similarity_bin_statistics(
    frame: pd.DataFrame, *, split: str, similarity_column: str, number_of_bins: int
) -> list[dict[str, Any]]:
    similarities = frame[similarity_column].to_numpy(dtype=float)
    errors = frame["absolute_error"].to_numpy(dtype=float)
    residuals = frame["residual"].to_numpy(dtype=float)
    boundaries = np.linspace(0.0, 1.0, number_of_bins + 1)
    rows: list[dict[str, Any]] = []

    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (similarities >= lower) & (
            similarities <= upper if index == number_of_bins - 1 else similarities < upper
        )
        count = int(np.count_nonzero(mask))
        rows.append(
            {
                "split": split,
                "similarity_measure": similarity_column,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "n": count,
                "mean_similarity": float(np.mean(similarities[mask])) if count else None,
                "mae": float(np.mean(errors[mask])) if count else None,
                "rmse": float(np.sqrt(np.mean(np.square(residuals[mask])))) if count else None,
                "mean_signed_error": float(np.mean(residuals[mask])) if count else None,
            }
        )
    return rows


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def add_error_scatter(
    axis: Any,
    *,
    frame: pd.DataFrame,
    split: str,
    similarity_column: str,
    metrics: dict[str, Any],
    top_k: int,
) -> None:
    similarities = frame[similarity_column].to_numpy(dtype=float)
    errors = frame["absolute_error"].to_numpy(dtype=float)
    axis.scatter(
        similarities,
        errors,
        s=18,
        alpha=0.47,
        color=SPLIT_COLOURS.get(split, "#2563eb"),
        edgecolors="none",
        rasterized=True,
    )
    axis.set_xlim(-0.025, 1.025)
    axis.set_ylim(-0.025 * max(float(np.max(errors)), 1.0), 1.055 * max(float(np.max(errors)), 1.0))
    label = "Top-1 training similarity" if similarity_column == "top1_similarity" else (
        f"Mean top-{top_k} training similarity"
    )
    axis.set_xlabel(f"{label} (Morgan Tanimoto)")
    axis.set_ylabel("Absolute FragNet prediction error")
    axis.set_title(f"{split.capitalize()} set: prediction error vs similarity", fontweight="semibold")
    axis.grid(False)
    axis.text(
        0.025,
        0.975,
        "\n".join(
            (
                f"n = {metrics['n']:,}",
                f"MAE = {format_metric(metrics['mae'])}",
                f"Mean similarity = {format_metric(metrics['similarity_mean'])}",
                f"Pearson r = {format_metric(metrics['error_similarity_pearson_r'])}",
                f"Spearman ρ = {format_metric(metrics['error_similarity_spearman_rho'])}",
            )
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "alpha": 0.90,
            "edgecolor": "#cbd5e1",
        },
    )


def save_scatter(
    *,
    output_dir: Path,
    frame: pd.DataFrame,
    split: str,
    similarity_column: str,
    metrics: dict[str, Any],
    top_k: int,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 7.0), constrained_layout=True)
    add_error_scatter(
        axis,
        frame=frame,
        split=split,
        similarity_column=similarity_column,
        metrics=metrics,
        top_k=top_k,
    )
    stem = f"{split}_error_vs_{similarity_column}"
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, facecolor="white")
    figure.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    plt.close(figure)


def save_combined_plot(
    *,
    output_dir: Path,
    results: dict[str, pd.DataFrame],
    metrics_by_split: dict[str, dict[str, dict[str, Any]]],
    similarity_columns: tuple[str, str],
    top_k: int,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(results),
        len(similarity_columns),
        figsize=(7.0 * len(similarity_columns), 6.1 * len(results)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, (split, frame) in enumerate(results.items()):
        for column_index, similarity_column in enumerate(similarity_columns):
            add_error_scatter(
                axes[row_index, column_index],
                frame=frame,
                split=split,
                similarity_column=similarity_column,
                metrics=metrics_by_split[split][similarity_column],
                top_k=top_k,
            )
    figure.suptitle(
        "FragNet prediction error and similarity to training molecules",
        fontsize=14,
        fontweight="semibold",
    )
    stem = "all_splits_error_vs_tanimoto_similarity"
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, facecolor="white")
    figure.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions_dir = resolve_predictions_dir(args)
    checkpoint = predictions_dir / "ft.pt"
    if not checkpoint.is_file():
        print(f"Note: no ft.pt found in {predictions_dir}; using the saved CSV predictions.")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else predictions_dir / "tanimoto_similarity"
    )

    chemistry, data_structs, fingerprint = load_rdkit(args)
    train_frame = read_prediction_table(predictions_dir / "train_predictions.csv", "train")
    metrics_path = predictions_dir / "regression_metrics.json"
    if metrics_path.is_file():
        saved_metrics = json.loads(metrics_path.read_text())
        for split in dict.fromkeys(["train", *args.splits]):
            table = train_frame if split == "train" else read_prediction_table(
                predictions_dir / f"{split}_predictions.csv", split
            )
            actual_rmse = float(np.sqrt(np.mean(table["squared_error"])))
            expected = saved_metrics.get(split, {})
            if "rmse" in expected and not math.isclose(actual_rmse, float(expected["rmse"]), rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError(
                    f"{split} CSV RMSE {actual_rmse:.4f} differs from saved RMSE "
                    f"{expected['rmse']:.4f}. Check prediction and target columns."
                )
    audit_path = args.results_dir / "split_audit.json"
    if audit_path.is_file() and predictions_dir == (args.results_dir / "experiment").resolve():
        counts = json.loads(audit_path.read_text())["accepted_counts"]
        if len(train_frame) != counts["train"]:
            raise ValueError("Train predictions row count differs from split_audit.json; check the run paths.")
        for split in args.splits:
            if split != "train":
                observed = len(pd.read_csv(predictions_dir / f"{split}_predictions.csv", usecols=["smiles"]))
                if observed != counts[split]:
                    raise ValueError(f"{split} predictions row count differs from split_audit.json; check the run paths.")
    print(f"Prediction directory: {predictions_dir}")
    print(f"Training reference molecules: {len(train_frame):,}")
    print(
        f"Morgan fingerprint: radius={args.radius}, bits={args.n_bits}, "
        f"chirality={args.include_chirality}"
    )
    print(f"Similarity measures: top-1 and mean top-{args.top_k}")
    print("Generating training fingerprints...", flush=True)
    train_frame, train_fingerprints = fingerprint_table(
        train_frame,
        split="train",
        chemistry=chemistry,
        fingerprint=fingerprint,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    metrics_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    flat_metrics: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    similarity_columns = ("top1_similarity", f"top{args.top_k}_mean_similarity")

    for split in args.splits:
        if split == "train":
            frame, fingerprints = train_frame, train_fingerprints
            print("\nAnalysing training molecules; exact self-matches are excluded.")
        else:
            print(f"\nGenerating {split} fingerprints...", flush=True)
            frame = read_prediction_table(predictions_dir / f"{split}_predictions.csv", split)
            frame, fingerprints = fingerprint_table(
                frame,
                split=split,
                chemistry=chemistry,
                fingerprint=fingerprint,
            )
            print(f"Analysing {len(frame):,} {split} molecules against the training set.")

        analysed = calculate_similarities(
            split=split,
            frame=frame,
            fingerprints=fingerprints,
            train_frame=train_frame,
            train_fingerprints=train_fingerprints,
            data_structs=data_structs,
            top_k=args.top_k,
            progress_every=args.progress_every,
        )
        exact_matches = int(analysed["exact_training_molecule_match"].sum())
        if split != "train" and exact_matches:
            print(
                f"  WARNING: {exact_matches:,} {split} molecules have an exact "
                "canonical-SMILES match in the training set. Check for split leakage."
            )

        analysed.to_csv(output_dir / f"{split}_tanimoto_predictions.csv", index=False)
        results[split] = analysed
        metrics_by_split[split] = {}

        for similarity_column in similarity_columns:
            metrics = calculate_error_metrics(
                analysed,
                split=split,
                similarity_column=similarity_column,
            )
            metrics_by_split[split][similarity_column] = metrics
            flat_metrics.append(metrics)
            bin_rows.extend(
                similarity_bin_statistics(
                    analysed,
                    split=split,
                    similarity_column=similarity_column,
                    number_of_bins=args.similarity_bins,
                )
            )
            save_scatter(
                output_dir=output_dir,
                frame=analysed,
                split=split,
                similarity_column=similarity_column,
                metrics=metrics,
                top_k=args.top_k,
                dpi=args.dpi,
            )
            print(
                f"  {similarity_column}: mean={metrics['similarity_mean']:.3f}, "
                f"error Pearson r={format_metric(metrics['error_similarity_pearson_r'])}, "
                f"error Spearman ρ={format_metric(metrics['error_similarity_spearman_rho'])}"
            )

    summary = {
        "fingerprint": {
            "type": "Morgan",
            "radius": args.radius,
            "n_bits": args.n_bits,
            "include_chirality": args.include_chirality,
        },
        "training_reference_count": len(train_frame),
        "top_k": args.top_k,
        "training_self_matches_excluded": True,
        "splits": metrics_by_split,
    }
    (output_dir / "similarity_error_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(flat_metrics).to_csv(output_dir / "similarity_error_metrics_summary.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(output_dir / "similarity_bins_summary.csv", index=False)
    save_combined_plot(
        output_dir=output_dir,
        results=results,
        metrics_by_split=metrics_by_split,
        similarity_columns=similarity_columns,
        top_k=args.top_k,
        dpi=args.dpi,
    )
    print(f"\nTanimoto prediction tables, plots, and statistics saved to:\n  {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

