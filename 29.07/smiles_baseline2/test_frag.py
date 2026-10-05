#!/usr/bin/env python3
"""Evaluate a trained FragNet regressor against its original xTB targets.

Examples
--------
Evaluate the training and test splits from the directory containing
fragnet_log_P_upconversion_regression:

    python evaluate_fragnet_xtb.py

Evaluate only one of the two available splits:

    python evaluate_fragnet_xtb.py --splits train

Point to a different results directory or FragNet checkout:

    python evaluate_fragnet_xtb.py \
        --results-dir /path/to/fragnet_log_P_upconversion_regression \
        --fragnet-root /path/to/FragNet

The script loads the existing checkpoint; it never retrains the model or
recalculates xTB properties.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


DEFAULT_RESULTS_DIR = Path("fragnet_log_P_upconversion_regression")
SPLIT_ALIASES = {"train": "train", "test": "test"}
GRAPH_SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}
SPLIT_COLOURS = {"train": "#2563eb", "validation": "#0891b2", "test": "#d97706"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FragNet predictions, xTB-versus-FragNet scatter plots, "
            "and regression metrics from an existing trained checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing fragnet_regression.yaml, experiment/, graph_data/, and splits/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to RESULTS_DIR/xtb_fragnet_evaluation.",
    )
    parser.add_argument(
        "--fragnet-root",
        type=Path,
        default=None,
        help="Root of the FragNet checkout if its Python package is not already importable.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_ALIASES),
        default=["train", "test"],
        help="Dataset splits to evaluate; the default includes the training and test sets only.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of molecular graphs evaluated per batch.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device; auto uses CUDA when it is available.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of exported PNG scatter plots.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.dpi < 1:
        parser.error("--dpi must be at least 1")

    # Preserve the requested order without evaluating aliases twice.
    args.splits = list(dict.fromkeys(SPLIT_ALIASES[name] for name in args.splits))
    return args


def ensure_fragnet_importable(results_dir: Path, explicit_root: Path | None) -> None:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root.expanduser().resolve())
    else:
        for parent in (Path.cwd(), results_dir.parent, results_dir.parent.parent):
            candidates.extend((parent, parent / "FragNet"))

    for candidate in dict.fromkeys(candidates):
        if (candidate / "fragnet" / "dataset" / "data.py").is_file():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return

    if explicit_root is not None:
        raise FileNotFoundError(
            f"--fragnet-root does not contain fragnet/dataset/data.py: {explicit_root}"
        )


def read_manifest(results_dir: Path) -> dict[str, Any]:
    manifest_path = results_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read the training manifest: {manifest_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Training manifest must contain a JSON object: {manifest_path}")
    return data


def target_information(manifest: dict[str, Any]) -> tuple[str, str]:
    target_column = str(
        manifest.get("target_column")
        or manifest.get("cleaning", {}).get("target_column")
        or "log_P_upconversion"
    )
    target_transform = str(
        manifest.get("target_transform")
        or manifest.get("cleaning", {}).get("target_transform")
        or "none"
    ).lower()
    if target_transform == "log10" and not target_column.lower().startswith("log"):
        display_label = f"log10({target_column})"
    else:
        display_label = target_column

    column_suffix = re.sub(r"[^A-Za-z0-9_]+", "_", display_label).strip("_")
    return display_label, column_suffix or "target"


def select_device(torch: Any, requested: str) -> Any:
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    selected = "cuda" if requested == "auto" and cuda_available else requested
    if selected == "auto":
        selected = "cpu"
    return torch.device(selected)


def build_model(cfg: Any, torch: Any, checkpoint: Path, device: Any) -> Any:
    from fragnet.model.gat.gat2 import FragNetFineTune

    model = FragNetFineTune(
        n_classes=cfg.finetune.model.n_classes,
        atom_features=cfg.atom_features,
        frag_features=cfg.frag_features,
        edge_features=cfg.edge_features,
        num_layer=cfg.finetune.model.num_layer,
        drop_ratio=cfg.finetune.model.drop_ratio,
        num_heads=cfg.finetune.model.num_heads,
        emb_dim=cfg.finetune.model.emb_dim,
        h1=cfg.finetune.model.h1,
        h2=cfg.finetune.model.h2,
        h3=cfg.finetune.model.h3,
        h4=cfg.finetune.model.h4,
        act=cfg.finetune.model.act,
        fthead=cfg.finetune.model.fthead,
    )

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
        if state and all(isinstance(key, str) and key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def resolve_graph_path(results_dir: Path, cfg: Any, split: str) -> Path:
    graph_split = GRAPH_SPLIT_NAMES[split]
    local_graph_path = results_dir / "graph_data" / f"{graph_split}.pkl"
    if local_graph_path.is_file():
        return local_graph_path

    configured_path = Path(str(getattr(cfg.finetune, graph_split).path)).expanduser()
    candidates = [configured_path]
    if not configured_path.is_absolute():
        candidates.append(results_dir / configured_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find the {split} graph dataset. Checked "
        f"{local_graph_path} and configured path {configured_path}."
    )


def graph_smiles(dataset: Any) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(dataset):
        value = getattr(item, "smiles", None)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if value is None or not str(value).strip():
            raise ValueError(f"Graph dataset entry {index} does not contain a usable SMILES string.")
        values.append(str(value))
    return values


def canonicalize_smiles(smiles: str) -> str | None:
    """Normalize equivalent aromatic, Kekulé, and explicit-hydrogen SMILES."""
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None


def match_split_rows_by_key(
    original: pd.DataFrame,
    source_keys: list[str | None],
    graph_keys: list[str | None],
) -> pd.DataFrame | None:
    """Match graph rows to source rows while retaining duplicate occurrences."""
    if any(key is None for key in graph_keys):
        return None

    keyed_source = original.copy()
    keyed_source["__match_key"] = source_keys
    keyed_source = keyed_source.loc[keyed_source["__match_key"].notna()].copy()
    keyed_source["__occurrence"] = keyed_source.groupby("__match_key").cumcount()

    ordered = pd.DataFrame(
        {
            "__match_key": graph_keys,
            "__graph_order": range(len(graph_keys)),
        }
    )
    ordered["__occurrence"] = ordered.groupby("__match_key").cumcount()
    merged = ordered.merge(
        keyed_source,
        how="left",
        on=["__match_key", "__occurrence"],
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any():
        return None

    merged = merged.sort_values("__graph_order", kind="stable")
    return merged.drop(
        columns=["__match_key", "__occurrence", "__graph_order", "_merge"]
    ).reset_index(drop=True)


def match_split_rows_by_unique_target(
    original: pd.DataFrame,
    source_targets: np.ndarray,
    graph_targets: np.ndarray,
) -> pd.DataFrame | None:
    """Use original xTB targets only when every match is uniquely identifiable."""
    available = np.ones(len(source_targets), dtype=bool)
    source_indices: list[int] = []

    for graph_target in graph_targets:
        candidates = np.flatnonzero(
            available
            & np.isclose(source_targets, graph_target, rtol=1e-5, atol=1e-5)
        )
        if len(candidates) != 1:
            return None
        source_index = int(candidates[0])
        source_indices.append(source_index)
        available[source_index] = False

    return original.iloc[source_indices].reset_index(drop=True)


def original_split_metadata(
    results_dir: Path,
    split: str,
    smiles: list[str],
    graph_targets: np.ndarray,
) -> pd.DataFrame:
    graph_split = GRAPH_SPLIT_NAMES[split]
    split_path = results_dir / "splits" / f"{graph_split}.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Original {split} xTB split table not found: {split_path}")

    original = pd.read_csv(split_path)
    if "smiles" not in original.columns:
        raise ValueError(f"Original split table has no 'smiles' column: {split_path}")
    if "y" not in original.columns:
        raise ValueError(f"Original split table has no 'y' xTB target column: {split_path}")

    original = original.copy()
    original["smiles"] = original["smiles"].astype(str)
    source_targets = pd.to_numeric(original["y"], errors="coerce").to_numpy(dtype=float)

    matched = match_split_rows_by_key(
        original,
        original["smiles"].tolist(),
        smiles,
    )
    if matched is not None:
        matched.attrs["match_method"] = "exact_smiles"
        return matched

    canonical_source = [canonicalize_smiles(value) for value in original["smiles"]]
    canonical_graph = [canonicalize_smiles(value) for value in smiles]
    matched = match_split_rows_by_key(original, canonical_source, canonical_graph)
    if matched is not None:
        matched.attrs["match_method"] = "canonical_smiles"
        print(
            f"  Matched {len(smiles):,} {split} molecules using canonicalized "
            "SMILES rather than their original text representations."
        )
        return matched

    if (
        len(original) == len(graph_targets)
        and np.isfinite(source_targets).all()
        and np.allclose(source_targets, graph_targets, rtol=1e-5, atol=1e-5)
    ):
        matched = original.reset_index(drop=True)
        matched.attrs["match_method"] = "verified_target_order"
        print(
            f"  Matched {len(smiles):,} {split} molecules by identical row "
            "order after verifying every original xTB target."
        )
        return matched

    if np.isfinite(source_targets).all():
        matched = match_split_rows_by_unique_target(
            original,
            source_targets,
            graph_targets,
        )
        if matched is not None:
            matched.attrs["match_method"] = "unique_xtb_targets"
            print(
                f"  Matched {len(smiles):,} {split} molecules through their "
                "uniquely identifiable original xTB targets."
            )
            return matched

    # Graph objects already contain the xTB target used to train/evaluate the
    # model. Those values remain correctly paired with predictions even when
    # the split CSV was rewritten or SMILES representations cannot be matched.
    # Evaluation must remain possible; only optional source metadata is lost.
    matched = pd.DataFrame({"smiles": smiles, "y": graph_targets})
    matched.attrs["match_method"] = "graph_xtb_targets_only"
    print(
        f"  WARNING: Could not reliably associate the {split} split CSV with "
        "the graph SMILES. Using the original xTB targets stored inside the "
        "graph dataset; additional split-table metadata will not be included."
    )
    return matched


def prediction_table(
    *,
    results_dir: Path,
    split: str,
    smiles: list[str],
    true: Any,
    predicted: Any,
    target_suffix: str,
) -> pd.DataFrame:
    xtb_graph = np.asarray(true, dtype=float).reshape(-1)
    fragnet = np.asarray(predicted, dtype=float).reshape(-1)
    if len(xtb_graph) != len(fragnet) or len(xtb_graph) != len(smiles):
        raise ValueError(
            f"Inconsistent {split} result lengths: xTB={len(xtb_graph)}, "
            f"FragNet={len(fragnet)}, SMILES={len(smiles)}."
        )
    if not len(xtb_graph):
        raise ValueError(f"The {split} dataset contains no evaluated molecules.")
    if not np.isfinite(xtb_graph).all() or not np.isfinite(fragnet).all():
        raise ValueError(f"The {split} results contain NaN or infinite values.")

    source = original_split_metadata(results_dir, split, smiles, xtb_graph)
    original_xtb = pd.to_numeric(source["y"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(original_xtb).all():
        raise ValueError(f"The original {split} xTB split contains non-numeric or missing targets.")
    if not np.allclose(original_xtb, xtb_graph, rtol=1e-5, atol=1e-5):
        differences = np.abs(original_xtb - xtb_graph)
        worst_index = int(np.argmax(differences))
        raise ValueError(
            f"The {split} graph targets do not match the original xTB split table. "
            f"Largest difference is {differences[worst_index]:.8g} at row "
            f"{worst_index}, SMILES={smiles[worst_index]!r}."
        )

    residual = fragnet - original_xtb
    result = pd.DataFrame(
        {
            "split": split,
            "smiles": smiles,
            f"xtb_{target_suffix}": original_xtb,
            f"fragnet_{target_suffix}": fragnet,
            "xtb_value": original_xtb,
            "fragnet_prediction": fragnet,
            "true": original_xtb,
            "predicted": fragnet,
            "residual": residual,
            "absolute_error": np.abs(residual),
            "squared_error": residual**2,
        }
    )

    result["metadata_match_method"] = source.attrs.get("match_method", "unknown")
    if not np.array_equal(source["smiles"].astype(str).to_numpy(), np.asarray(smiles)):
        result["xtb_source_smiles"] = source["smiles"].astype(str).to_numpy()

    for column in source.columns:
        if column == "smiles":
            continue
        output_column = column if column not in result.columns else f"source_{column}"
        result[output_column] = source[column].to_numpy()
    return result


def calculate_metrics(table: pd.DataFrame) -> dict[str, int | float | None]:
    xtb = table["xtb_value"].to_numpy(dtype=float)
    predicted = table["fragnet_prediction"].to_numpy(dtype=float)
    residual = predicted - xtb
    enough_variation = len(xtb) >= 2 and np.std(xtb) > 0 and np.std(predicted) > 0

    pearson = float(np.corrcoef(xtb, predicted)[0, 1]) if enough_variation else None
    spearman_raw = (
        pd.Series(xtb).corr(pd.Series(predicted), method="spearman")
        if enough_variation
        else float("nan")
    )
    spearman = float(spearman_raw) if np.isfinite(spearman_raw) else None
    if enough_variation:
        slope, intercept = np.polyfit(xtb, predicted, deg=1)
        regression_slope: float | None = float(slope)
        regression_intercept: float | None = float(intercept)
    else:
        regression_slope = None
        regression_intercept = None

    metrics: dict[str, int | float | None] = {
        "n": int(len(xtb)),
        "r2": float(r2_score(xtb, predicted)) if len(xtb) >= 2 else None,
        "rmse": float(math.sqrt(mean_squared_error(xtb, predicted))),
        "mse": float(mean_squared_error(xtb, predicted)),
        "mae": float(mean_absolute_error(xtb, predicted)),
        "median_absolute_error": float(median_absolute_error(xtb, predicted)),
        "max_absolute_error": float(max_error(xtb, predicted)),
        "mean_error_bias": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=1)) if len(residual) >= 2 else 0.0,
        "explained_variance": (
            float(explained_variance_score(xtb, predicted)) if len(xtb) >= 2 else None
        ),
        "pearson_r": pearson,
        "pearson_r_squared": float(pearson**2) if pearson is not None else None,
        "spearman_rho": spearman,
        "regression_slope": regression_slope,
        "regression_intercept": regression_intercept,
        "xtb_min": float(np.min(xtb)),
        "xtb_max": float(np.max(xtb)),
        "xtb_mean": float(np.mean(xtb)),
        "fragnet_min": float(np.min(predicted)),
        "fragnet_max": float(np.max(predicted)),
        "fragnet_mean": float(np.mean(predicted)),
    }
    return {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in metrics.items()
    }


def metric_text(value: int | float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def add_scatter(
    axis: Any,
    *,
    table: pd.DataFrame,
    metrics: dict[str, int | float | None],
    split: str,
    target_label: str,
) -> None:
    xtb = table["xtb_value"].to_numpy(dtype=float)
    predicted = table["fragnet_prediction"].to_numpy(dtype=float)
    colour = SPLIT_COLOURS.get(split, "#2563eb")
    axis.scatter(
        xtb,
        predicted,
        s=18,
        alpha=0.47,
        color=colour,
        edgecolors="none",
        rasterized=True,
    )

    minimum = float(min(np.min(xtb), np.min(predicted)))
    maximum = float(max(np.max(xtb), np.max(predicted)))
    span = max(maximum - minimum, 1e-9)
    lower = minimum - 0.045 * span
    upper = maximum + 0.045 * span
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"xTB {target_label}")
    axis.set_ylabel(f"FragNet-predicted {target_label}")
    axis.set_title(f"{split.capitalize()} set: FragNet vs xTB", fontweight="semibold")
    axis.grid(False)
    axis.text(
        0.025,
        0.975,
        "\n".join(
            (
                f"n = {metric_text(metrics['n'])}",
                f"R² = {metric_text(metrics['r2'])}",
                f"RMSE = {metric_text(metrics['rmse'])}",
                f"MAE = {metric_text(metrics['mae'])}",
                f"Pearson r = {metric_text(metrics['pearson_r'])}",
                f"Spearman ρ = {metric_text(metrics['spearman_rho'])}",
            )
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.89, "edgecolor": "#cbd5e1"},
    )
def save_split_outputs(
    *,
    output_dir: Path,
    table: pd.DataFrame,
    metrics: dict[str, int | float | None],
    split: str,
    target_label: str,
    dpi: int,
) -> None:
    table.to_csv(output_dir / f"{split}_predictions.csv", index=False)
    (output_dir / f"{split}_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(7.4, 7.0), constrained_layout=True)
    add_scatter(axis, table=table, metrics=metrics, split=split, target_label=target_label)
    figure.savefig(output_dir / f"{split}_fragnet_vs_xtb.png", dpi=dpi, facecolor="white")
    figure.savefig(output_dir / f"{split}_fragnet_vs_xtb.pdf", facecolor="white")
    plt.close(figure)


def save_combined_outputs(
    *,
    output_dir: Path,
    results: dict[str, tuple[pd.DataFrame, dict[str, int | float | None]]],
    target_label: str,
    dpi: int,
) -> None:
    metrics_by_split = {split: metrics for split, (_, metrics) in results.items()}
    (output_dir / "regression_metrics.json").write_text(
        json.dumps(metrics_by_split, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"split": split, **metrics} for split, metrics in metrics_by_split.items()]
    ).to_csv(output_dir / "regression_metrics_summary.csv", index=False)

    if len(results) < 2:
        return

    figure, axes = plt.subplots(
        1,
        len(results),
        figsize=(7.0 * len(results), 6.5),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (split, (table, metrics)) in zip(axes[0], results.items()):
        add_scatter(axis, table=table, metrics=metrics, split=split, target_label=target_label)

    figure.suptitle("FragNet predictions compared with original xTB values", fontsize=14, fontweight="semibold")
    figure.savefig(output_dir / "all_splits_fragnet_vs_xtb.png", dpi=dpi, facecolor="white")
    figure.savefig(output_dir / "all_splits_fragnet_vs_xtb.pdf", facecolor="white")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"FragNet results directory does not exist: {results_dir}")

    config_path = results_dir / "fragnet_regression.yaml"
    checkpoint_path = results_dir / "experiment" / "ft.pt"
    for required_path, description in (
        (config_path, "FragNet configuration"),
        (checkpoint_path, "trained FragNet checkpoint"),
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing {description}: {required_path}")

    ensure_fragnet_importable(results_dir, args.fragnet_root)
    try:
        import torch
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader

        from fragnet.dataset.data import collate_fn
        from fragnet.dataset.dataset import load_pickle_dataset
        from fragnet.train.utils import TrainerFineTune
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency '{exc.name}'. Activate the same Python environment "
            "used to train FragNet and, if necessary, provide --fragnet-root /path/to/FragNet."
        ) from exc

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    manifest = read_manifest(results_dir)
    target_label, target_suffix = target_information(manifest)
    device = select_device(torch, args.device)
    model = build_model(cfg, torch, checkpoint_path, device)
    trainer = TrainerFineTune(target_type=cfg.finetune.target_type)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "xtb_fragnet_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory: {results_dir}")
    print(f"Trained checkpoint: {checkpoint_path}")
    print(f"Target: {target_label}")
    print(f"Evaluation device: {device}")
    print(f"Requested splits: {', '.join(args.splits)}")

    results: dict[str, tuple[pd.DataFrame, dict[str, int | float | None]]] = {}
    for split in args.splits:
        dataset_path = resolve_graph_path(results_dir, cfg, split)
        dataset = load_pickle_dataset(dataset_path)
        loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        print(f"\nEvaluating {split}: {len(loader.dataset):,} molecules")
        print(f"Graph dataset: {dataset_path}")
        with torch.no_grad():
            fragnet_score, true, predicted = trainer.test(
                model=model,
                loader=loader,
                device=device,
            )

        smiles = graph_smiles(loader.dataset)
        table = prediction_table(
            results_dir=results_dir,
            split=split,
            smiles=smiles,
            true=true,
            predicted=predicted,
            target_suffix=target_suffix,
        )
        metrics = calculate_metrics(table)
        score = float(fragnet_score)
        metrics["fragnet_reported_rmse"] = math.sqrt(score) if np.isfinite(score) and score >= 0 else None

        save_split_outputs(
            output_dir=output_dir,
            table=table,
            metrics=metrics,
            split=split,
            target_label=target_label,
            dpi=args.dpi,
        )
        results[split] = (table, metrics)
        print(
            f"  R²={metric_text(metrics['r2'], digits=4)}, "
            f"RMSE={metric_text(metrics['rmse'], digits=4)}, "
            f"MAE={metric_text(metrics['mae'], digits=4)}, "
            f"Pearson r={metric_text(metrics['pearson_r'], digits=4)}, "
        )

    save_combined_outputs(
        output_dir=output_dir,
        results=results,
        target_label=target_label,
        dpi=args.dpi,
    )
    print(f"\nAll prediction tables, plots, and regression statistics saved to:\n  {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
