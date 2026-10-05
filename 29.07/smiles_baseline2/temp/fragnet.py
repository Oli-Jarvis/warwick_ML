#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train official FragNet as a regressor on xTB P values."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help=(
            "Dataset parent containing run_*/full_history.csv. When omitted, "
            "the script detects whether it is being run from the dataset parent "
            "or from its FragNet subdirectory."
        ),
    )
    parser.add_argument(
        "--fragnet-root",
        type=Path,
        default=None,
        help=(
            "Root of the cloned PNNL FragNet repository. When omitted, the "
            "script checks the current directory and ./FragNet."
        ),
    )
    parser.add_argument(
        "--target-column",
        default="log_P_upconversion",
        help="Continuous regression target, e.g. log_P_upconversion or P_upconversion.",
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<dataset>/fragnet_<target-column>_regression."
        ),
    )
    parser.add_argument(
        "--split-method",
        choices=("scaffold", "random"),
        default="scaffold",
        help="Use scaffold splitting to assess generalisation to new scaffolds.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--deduplicate-smiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Average the target over duplicate canonical SMILES and retain one "
            "row per molecule. Enabled by default to prevent split leakage."
        ),
    )
    parser.add_argument(
        "--remove-gold",
        action="store_true",
        help="Remove literal [Au] tokens before parsing the SMILES.",
    )
    parser.add_argument(
        "--target-transform",
        choices=("none", "log10"),
        default="none",
        help=(
            "Optional target transform. Do not use log10 when the selected "
            "column is already log_P_upconversion."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of molecules passed to FragNet graph generation at a time.",
    )
    parser.add_argument(
        "--frag-type",
        choices=("brics", "murcko"),
        default="brics",
    )
    parser.add_argument(
        "--data-type",
        default="exp1s",
        help="Official FragNet feature configuration; exp1s is recommended.",
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--early-stopping-patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional FragNet pretraining checkpoint. If omitted, the script "
            "uses fragnet/exps/pt/unimol_exp1s4/pt.pt only when it exists."
        ),
    )
    parser.add_argument(
        "--no-pretraining",
        action="store_true",
        help="Train without loading FragNet pretrained weights.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Clean, split, generate graphs, and write YAML without training.",
    )
    parser.add_argument(
        "--skip-graph-generation",
        action="store_true",
        help="Reuse existing graph_data/train.pkl, val.pkl and test.pkl.",
    )
    parser.add_argument(
        "--overwrite-graphs",
        action="store_true",
        help="Replace existing graph pickle files.",
    )
    return parser.parse_args()


def is_fragnet_root(path: Path) -> bool:
    return (
        (path / "setup.py").exists()
        and (path / "fragnet" / "train" / "finetune" / "finetune_gat2.py").exists()
        and (path / "fragnet" / "dataset" / "dataset.py").exists()
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cwd = Path.cwd().resolve()

    if args.fragnet_root is not None:
        fragnet_root = args.fragnet_root.expanduser().resolve()
    elif is_fragnet_root(cwd):
        fragnet_root = cwd
    elif is_fragnet_root(cwd / "FragNet"):
        fragnet_root = cwd / "FragNet"
    else:
        raise FileNotFoundError(
            "Could not locate the FragNet repository. Supply --fragnet-root."
        )

    if not is_fragnet_root(fragnet_root):
        raise FileNotFoundError(
            f"This does not appear to be a FragNet repository: {fragnet_root}"
        )

    if args.input_path is not None:
        input_path = args.input_path.expanduser().resolve()
    elif is_fragnet_root(cwd):
        input_path = cwd.parent
    else:
        input_path = cwd

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        safe_target = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in args.target_column
        )
        output_dir = input_path / f"fragnet_{safe_target}_regression"

    return fragnet_root, input_path, output_dir


def validate_arguments(args: argparse.Namespace) -> None:
    fractions = [
        args.train_fraction,
        args.validation_fraction,
        args.test_fraction,
    ]
    if any(not math.isfinite(value) or value <= 0 for value in fractions):
        raise ValueError("Every split fraction must be finite and greater than zero.")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1.")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1.")


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline()
    counts = {
        ",": header.count(","),
        ";": header.count(";"),
        "\t": header.count("\t"),
    }
    delimiter = max(counts, key=counts.get)
    if counts[delimiter] == 0:
        raise ValueError(
            f"Could not detect comma, semicolon, or tab delimiter in {path}."
        )
    return delimiter


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        delimiter = detect_delimiter(path)
        print(f"  {path}  delimiter={delimiter!r}")
        frame = pd.read_csv(
            path,
            sep=delimiter,
            low_memory=False,
            encoding="utf-8-sig",
        )
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame
    raise ValueError(f"Unsupported input format: {path}")


def find_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.glob("run_*/full_history.csv"))
    if not files:
        files = sorted(input_path.rglob("full_history.csv"))
    if not files:
        raise FileNotFoundError(
            f"No run_*/full_history.csv files were found under {input_path}."
        )
    return files


def clean_smiles_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    return text


def load_and_clean(
    input_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    input_files = find_input_files(input_path)
    print(f"Found {len(input_files):,} input file(s):")

    pieces: list[pd.DataFrame] = []
    for path in input_files:
        frame = load_table(path)
        missing = {
            args.smiles_column,
            args.target_column,
        }.difference(frame.columns)
        if missing:
            raise KeyError(
                f"{path} is missing {sorted(missing)}. "
                f"Available columns: {list(frame.columns)}"
            )

        piece = frame[[args.smiles_column, args.target_column]].copy()
        piece.columns = ["original_smiles", "target"]
        piece["source_file"] = str(path)
        pieces.append(piece)

    raw = pd.concat(pieces, ignore_index=True)
    rows_read = len(raw)

    raw["target"] = pd.to_numeric(raw["target"], errors="coerce")
    target_array = raw["target"].to_numpy(dtype=float, na_value=np.nan)
    finite_target = np.isfinite(target_array)
    invalid_target_count = int((~finite_target).sum())
    data = raw.loc[finite_target].copy()

    data["processed_smiles"] = data["original_smiles"].map(clean_smiles_text)
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

    valid_rows: list[dict[str, Any]] = []
    invalid_smiles_examples: list[str] = []

    for row in data.itertuples(index=False):
        try:
            molecule = Chem.MolFromSmiles(row.processed_smiles)
        except Exception:
            molecule = None

        if molecule is None:
            if len(invalid_smiles_examples) < 10:
                invalid_smiles_examples.append(str(row.processed_smiles))
            continue

        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        valid_rows.append(
            {
                "smiles": canonical_smiles,
                "y": float(row.target),
                "source_file": row.source_file,
            }
        )

    cleaned = pd.DataFrame(valid_rows)
    rdkit_failures = len(data) - len(cleaned)

    removed_for_transform = 0
    if args.target_transform == "log10":
        positive = cleaned["y"] > 0
        removed_for_transform = int((~positive).sum())
        cleaned = cleaned.loc[positive].copy()
        cleaned["y"] = np.log10(cleaned["y"].to_numpy(dtype=float))

    duplicates_removed = 0
    if args.deduplicate_smiles:
        before = len(cleaned)
        cleaned = (
            cleaned.groupby("smiles", as_index=False)
            .agg(
                y=("y", "mean"),
                target_sd_across_duplicates=("y", "std"),
                duplicate_count=("y", "size"),
                source_file=("source_file", "first"),
            )
        )
        cleaned["target_sd_across_duplicates"] = (
            cleaned["target_sd_across_duplicates"].fillna(0.0)
        )
        duplicates_removed = before - len(cleaned)
    else:
        cleaned["target_sd_across_duplicates"] = 0.0
        cleaned["duplicate_count"] = 1

    cleaned = cleaned.sample(
        frac=1.0,
        random_state=args.seed,
    ).reset_index(drop=True)

    if len(cleaned) < 30:
        raise ValueError(
            f"Only {len(cleaned)} valid molecules remain; this is too few for "
            "a train/validation/test FragNet workflow."
        )

    report = {
        "rows_read": int(rows_read),
        "invalid_or_missing_target": invalid_target_count,
        "missing_or_non_text_smiles": missing_smiles_count,
        "rdkit_parse_failures": rdkit_failures,
        "removed_before_log10_transform": removed_for_transform,
        "duplicates_removed": duplicates_removed,
        "valid_molecules": int(len(cleaned)),
        "target_column": args.target_column,
        "target_transform": args.target_transform,
        "target_min": float(cleaned["y"].min()),
        "target_max": float(cleaned["y"].max()),
        "target_mean": float(cleaned["y"].mean()),
        "target_sd": float(cleaned["y"].std(ddof=1)),
        "invalid_smiles_examples": invalid_smiles_examples,
    }
    return cleaned, input_files, report


def scaffold_key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Unexpected invalid canonical SMILES: {smiles}")

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)

    # Acyclic molecules have an empty Murcko scaffold. Keeping every acyclic
    # molecule in one enormous group can make the split unusable, so these are
    # treated as separate scaffold groups.
    return scaffold_smiles if scaffold_smiles else f"ACYCLIC::{smiles}"


def random_split(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    shuffled = data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    total = len(shuffled)

    n_train = int(round(args.train_fraction * total))
    n_validation = int(round(args.validation_fraction * total))
    n_train = min(max(n_train, 1), total - 2)
    n_validation = min(max(n_validation, 1), total - n_train - 1)

    return {
        "train": shuffled.iloc[:n_train].copy().reset_index(drop=True),
        "val": shuffled.iloc[n_train:n_train + n_validation].copy().reset_index(drop=True),
        "test": shuffled.iloc[n_train + n_validation:].copy().reset_index(drop=True),
    }


def scaffold_split(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    groups: dict[str, list[int]] = {}
    for index, smiles in data["smiles"].items():
        groups.setdefault(scaffold_key(smiles), []).append(index)

    rng = random.Random(args.seed)
    grouped_indices = list(groups.values())
    rng.shuffle(grouped_indices)
    grouped_indices.sort(key=len, reverse=True)

    target_sizes = {
        "train": args.train_fraction * len(data),
        "val": args.validation_fraction * len(data),
        "test": args.test_fraction * len(data),
    }
    assigned: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for group in grouped_indices:
        # Select the split with the largest proportional deficit.
        destination = max(
            assigned,
            key=lambda name: (
                target_sizes[name] - len(assigned[name])
            ) / target_sizes[name],
        )
        assigned[destination].extend(group)

    result = {
        name: data.loc[indices]
        .sample(frac=1.0, random_state=args.seed + offset)
        .reset_index(drop=True)
        for offset, (name, indices) in enumerate(assigned.items())
    }

    if any(frame.empty for frame in result.values()):
        raise ValueError(
            "Scaffold splitting produced an empty split. Try --split-method random."
        )
    return result


def verify_split_integrity(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for first_index, first_name in enumerate(names):
        first_smiles = set(splits[first_name]["smiles"])
        for second_name in names[first_index + 1:]:
            overlap = first_smiles.intersection(splits[second_name]["smiles"])
            if overlap:
                raise RuntimeError(
                    f"SMILES leakage between {first_name} and {second_name}: "
                    f"{len(overlap)} duplicated molecules."
                )


def configure_fragnet_import(fragnet_root: Path) -> None:
    root_text = str(fragnet_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def create_fragnet_graph_pickle(
    frame: pd.DataFrame,
    *,
    output_pickle: Path,
    rejected_csv: Path,
    chunk_size: int,
    data_type: str,
    frag_type: str,
) -> dict[str, int]:
    """
    Create official FragNet Data objects in chunks.

    FragNet's FinetuneData class and CreateData pipeline are used unchanged.
    Chunking limits peak memory use and preserves successful molecules if a
    later chunk fails.
    """
    from fragnet.dataset.dataset import FinetuneData
    from fragnet.dataset.utils import extract_data

    dataset_builder = FinetuneData(
        target_name="y",
        data_type=data_type,
        frag_type=frag_type,
    )

    graph_objects: list[Any] = []
    rejected_rows: list[dict[str, Any]] = []

    number_of_chunks = math.ceil(len(frame) / chunk_size)

    for chunk_number, start in enumerate(
        range(0, len(frame), chunk_size),
        start=1,
    ):
        stop = min(start + chunk_size, len(frame))
        chunk = frame.iloc[start:stop][["smiles", "y"]].copy()
        chunk.reset_index(drop=True, inplace=True)

        print(
            f"  chunk {chunk_number}/{number_of_chunks}: "
            f"molecules {start + 1:,}-{stop:,}"
        )

        # First try the entire chunk for speed.
        try:
            generated = dataset_builder.get_ft_dataset(chunk)
            successful = extract_data(generated)
            graph_objects.extend(successful)

            if len(successful) != len(chunk):
                successful_smiles = {str(item.smiles) for item in successful}
                for row in chunk.itertuples(index=False):
                    if row.smiles not in successful_smiles:
                        rejected_rows.append(
                            {
                                "smiles": row.smiles,
                                "y": row.y,
                                "reason": "FragNet returned no graph object",
                            }
                        )
        except Exception as chunk_error:
            print(
                "    Chunk generation failed; retrying its molecules "
                "individually to isolate failures."
            )

            for row in chunk.itertuples(index=False):
                single = pd.DataFrame(
                    [{"smiles": row.smiles, "y": row.y}]
                )
                try:
                    generated = dataset_builder.get_ft_dataset(single)
                    successful = extract_data(generated)
                    if successful:
                        graph_objects.extend(successful)
                    else:
                        rejected_rows.append(
                            {
                                "smiles": row.smiles,
                                "y": row.y,
                                "reason": "FragNet returned no graph object",
                            }
                        )
                except Exception as molecule_error:
                    rejected_rows.append(
                        {
                            "smiles": row.smiles,
                            "y": row.y,
                            "reason": (
                                f"{type(molecule_error).__name__}: "
                                f"{molecule_error}"
                            ),
                        }
                    )

            print(
                f"    Original chunk error: "
                f"{type(chunk_error).__name__}: {chunk_error}"
            )

        # Save progress after every chunk so an interrupted graph-generation
        # job does not lose all completed work.
        temporary_path = output_pickle.with_suffix(".partial.pkl")
        with temporary_path.open("wb") as handle:
            pickle.dump(graph_objects, handle)

    if not graph_objects:
        raise RuntimeError(
            f"FragNet could not generate any graphs for {output_pickle.stem}."
        )

    with output_pickle.open("wb") as handle:
        pickle.dump(graph_objects, handle)

    partial_path = output_pickle.with_suffix(".partial.pkl")
    if partial_path.exists():
        partial_path.unlink()

    pd.DataFrame(
        rejected_rows,
        columns=["smiles", "y", "reason"],
    ).to_csv(rejected_csv, index=False)

    return {
        "requested": int(len(frame)),
        "generated": int(len(graph_objects)),
        "rejected": int(len(rejected_rows)),
    }


def verify_graph_dataset(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        dataset = pickle.load(handle)

    dataset = [item for item in dataset if item is not None]
    if not dataset:
        raise ValueError(f"Graph dataset is empty: {path}")

    first = dataset[0]
    required_attributes = ["x_atoms", "edge_index", "y", "smiles"]
    missing = [
        attribute for attribute in required_attributes
        if not hasattr(first, attribute)
    ]

    # FragNet object names have changed in some revisions, so only fail on
    # fields that are essential to training and clearly absent.
    essential_missing = [
        attribute for attribute in ["y", "smiles"]
        if not hasattr(first, attribute)
    ]
    if essential_missing:
        raise ValueError(
            f"Graph objects in {path} lack {essential_missing}."
        )

    targets = np.array(
        [float(item.y.reshape(-1)[0].item()) for item in dataset],
        dtype=float,
    )
    return {
        "count": len(dataset),
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "target_mean": float(targets.mean()),
        "nonessential_attributes_not_seen": missing,
    }


def resolve_pretrained_checkpoint(
    args: argparse.Namespace,
    fragnet_root: Path,
) -> Path | None:
    if args.no_pretraining:
        return None

    if args.pretrained_checkpoint is not None:
        checkpoint = args.pretrained_checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint does not exist: {checkpoint}"
            )
        return checkpoint

    candidate = (
        fragnet_root
        / "fragnet"
        / "exps"
        / "pt"
        / "unimol_exp1s4"
        / "pt.pt"
    )
    return candidate if candidate.exists() else None


def write_fragnet_config(
    *,
    config_path: Path,
    experiment_dir: Path,
    graph_paths: dict[str, Path],
    checkpoint: Path | None,
    args: argparse.Namespace,
) -> None:
    config = {
        "seed": args.seed,
        "data_seed": None,
        "exp_dir": str(experiment_dir.resolve()),
        "model_version": "gat2",
        "device": "gpu",
        "atom_features": 167,
        "frag_features": 167,
        "edge_features": 17,
        "fedge_in": 6,
        "fbond_edge_in": 6,
        "pretrain": {
            "model_version": "gat2",
            "num_layer": 4,
            "drop_ratio": 0.2,
            "num_heads": 4,
            "emb_dim": 128,
            "chkpoint_name": (
                str(checkpoint.resolve()) if checkpoint is not None else None
            ),
            "loss": "mse",
            "batch_size": 128,
            "es_patience": 500,
            "lr": 1e-4,
            "n_epochs": 20000,
            "n_classes": 1,
        },
        "finetune": {
            "n_multi_task_heads": 0,
            "batch_size": args.batch_size,
            "lr": args.learning_rate,
            "model": {
                "n_classes": 1,
                "num_layer": 4,
                "drop_ratio": 0.1,
                "num_heads": 4,
                "emb_dim": 128,
                "h1": 128,
                "h2": 1024,
                "h3": 1024,
                "h4": 512,
                "act": "relu",
                "fthead": "FTHead3",
            },
            "n_epochs": args.epochs,
            "target_type": "regr",
            "loss": "mse",
            "use_schedular": False,
            "es_patience": args.early_stopping_patience,
            "chkpoint_name": str((experiment_dir / "ft.pt").resolve()),
            "train": {"path": str(graph_paths["train"].resolve())},
            "val": {"path": str(graph_paths["val"].resolve())},
            "test": {"path": str(graph_paths["test"].resolve())},
        },
    }

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def run_training(
    *,
    fragnet_root: Path,
    config_path: Path,
) -> None:
    training_script = (
        fragnet_root
        / "fragnet"
        / "train"
        / "finetune"
        / "finetune_gat2.py"
    )
    environment = os.environ.copy()
    previous_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(fragnet_root), previous_pythonpath)
        if item
    )

    command = [
        sys.executable,
        str(training_script),
        "--config",
        str(config_path),
    ]
    print("\nStarting official FragNet fine-tuning:")
    print("  " + " ".join(command))
    subprocess.run(
        command,
        cwd=fragnet_root / "fragnet",
        env=environment,
        check=True,
    )

def find_prediction_pickle(
    experiment_dir: Path,
    prefix: str,
    seed: int,
) -> Path:
    expected = experiment_dir / f"{prefix}_{seed}.pkl"
    if expected.exists():
        return expected

    candidates = sorted(
        experiment_dir.glob(f"{prefix}_*.pkl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {prefix}_*.pkl was produced in {experiment_dir}."
        )
    return candidates[-1]


def calculate_regression_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    rmse = float(mean_squared_error(true, predicted) ** 0.5)
    mae = float(mean_absolute_error(true, predicted))
    r2 = float(r2_score(true, predicted))

    pearson = (
        float(np.corrcoef(true, predicted)[0, 1])
        if len(true) > 1
        and np.std(true) > 0
        and np.std(predicted) > 0
        else float("nan")
    )
    spearman = float(
        pd.Series(true).corr(pd.Series(predicted), method="spearman")
    )

    return {
        "n": int(len(true)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }


def save_prediction_outputs(
    result_pickle: Path,
    *,
    split_name: str,
    output_dir: Path,
    target_label: str,
) -> dict[str, float | int]:
    with result_pickle.open("rb") as handle:
        result = pickle.load(handle)

    true = np.asarray(result["true"], dtype=float).reshape(-1)
    predicted = np.asarray(result["pred"], dtype=float).reshape(-1)
    smiles = list(result.get("smiles", [""] * len(true)))

    if not (len(true) == len(predicted) == len(smiles)):
        raise ValueError(
            f"Inconsistent result lengths in {result_pickle}."
        )

    residual = predicted - true
    table = pd.DataFrame(
        {
            "smiles": smiles,
            "true": true,
            "predicted": predicted,
            "residual": residual,
            "absolute_error": np.abs(residual),
        }
    )
    table.to_csv(
        output_dir / f"{split_name}_predictions.csv",
        index=False,
    )

    metrics = calculate_regression_metrics(true, predicted)

    # Predicted-versus-actual plot.
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(true, predicted, alpha=0.55, s=18)
    lower = float(min(true.min(), predicted.min()))
    upper = float(max(true.max(), predicted.max()))
    axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2)
    axis.set_xlabel(f"Actual {target_label}")
    axis.set_ylabel(f"Predicted {target_label}")
    axis.set_title(
        f"FragNet {split_name}: predicted vs actual\n"
        f"RMSE={metrics['rmse']:.4g}, MAE={metrics['mae']:.4g}, "
        f"R²={metrics['r2']:.4g}"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{split_name}_predicted_vs_actual.png",
        dpi=300,
    )
    plt.close(figure)

    # Residual plot.
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.scatter(predicted, residual, alpha=0.55, s=18)
    axis.axhline(0.0, linestyle="--", linewidth=1.2)
    axis.set_xlabel(f"Predicted {target_label}")
    axis.set_ylabel("Residual (predicted − actual)")
    axis.set_title(f"FragNet {split_name}: residuals")
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{split_name}_residuals.png",
        dpi=300,
    )
    plt.close(figure)

    return metrics

def main() -> int:
    args = parse_args()
    validate_arguments(args)

    fragnet_root, input_path, output_dir = resolve_paths(args)
    configure_fragnet_import(fragnet_root)

    print("Resolved paths:")
    print(f"  FragNet repository: {fragnet_root}")
    print(f"  Dataset input:      {input_path}")
    print(f"  Output directory:   {output_dir}")
    print(f"  Target column:      {args.target_column}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    graph_dir = output_dir / "graph_data"
    experiment_dir = output_dir / "experiment"
    for directory in (split_dir, graph_dir, experiment_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data, input_files, cleaning_report = load_and_clean(input_path, args)

    print("\nCleaning summary:")
    for key, value in cleaning_report.items():
        if key != "invalid_smiles_examples":
            print(f"  {key}: {value}")

    if cleaning_report["invalid_smiles_examples"]:
        print("  Invalid SMILES examples:")
        for example in cleaning_report["invalid_smiles_examples"]:
            print(f"    {example}")

    if args.split_method == "scaffold":
        splits = scaffold_split(data, args)
    else:
        splits = random_split(data, args)

    verify_split_integrity(splits)

    split_paths: dict[str, Path] = {}
    print("\nSplit sizes:")
    for name, frame in splits.items():
        path = split_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        split_paths[name] = path
        print(
            f"  {name}: {len(frame):,} "
            f"({100.0 * len(frame) / len(data):.2f}%)"
        )

    graph_paths = {
        name: graph_dir / f"{name}.pkl"
        for name in splits
    }
    graph_generation_report: dict[str, Any] = {}

    for name, frame in splits.items():
        graph_path = graph_paths[name]
        rejected_path = graph_dir / f"{name}_rejected.csv"

        if args.skip_graph_generation:
            if not graph_path.exists():
                raise FileNotFoundError(
                    f"--skip-graph-generation was used, but {graph_path} "
                    "does not exist."
                )
        elif graph_path.exists() and not args.overwrite_graphs:
            print(
                f"\nReusing existing graph dataset: {graph_path}\n"
                "Use --overwrite-graphs to regenerate it."
            )
        else:
            print(f"\nGenerating official FragNet graphs for {name}:")
            graph_generation_report[name] = create_fragnet_graph_pickle(
                frame,
                output_pickle=graph_path,
                rejected_csv=rejected_path,
                chunk_size=args.chunk_size,
                data_type=args.data_type,
                frag_type=args.frag_type,
            )

        graph_generation_report.setdefault(name, {})
        graph_generation_report[name]["verification"] = verify_graph_dataset(
            graph_path
        )

    checkpoint = resolve_pretrained_checkpoint(args, fragnet_root)
    if checkpoint is None:
        print(
            "\nNo pretrained checkpoint will be loaded. The model will be "
            "trained from randomly initialised weights."
        )
    else:
        print(f"\nUsing pretrained checkpoint: {checkpoint}")

    config_path = output_dir / "fragnet_regression.yaml"
    write_fragnet_config(
        config_path=config_path,
        experiment_dir=experiment_dir,
        graph_paths=graph_paths,
        checkpoint=checkpoint,
        args=args,
    )
    print(f"Saved FragNet configuration: {config_path}")

    manifest = {
        "fragnet_root": str(fragnet_root),
        "input_path": str(input_path),
        "input_files": [str(path.resolve()) for path in input_files],
        "output_dir": str(output_dir),
        "target_column": args.target_column,
        "target_transform": args.target_transform,
        "split_method": args.split_method,
        "split_sizes_before_graph_generation": {
            name: int(len(frame)) for name, frame in splits.items()
        },
        "cleaning": cleaning_report,
        "graph_generation": graph_generation_report,
        "pretrained_checkpoint": (
            str(checkpoint) if checkpoint is not None else None
        ),
        "configuration": vars(args),
    }
    # Convert any Path values in configuration to strings.
    manifest["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in manifest["configuration"].items()
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    if args.prepare_only:
        print("\nPreparation completed; training was not started.")
        print("Train later with:")
        print(
            f"  python "
            f"{fragnet_root / 'fragnet/train/finetune/finetune_gat2.py'} "
            f"--config {config_path}"
        )
        return 0

    run_training(
        fragnet_root=fragnet_root,
        config_path=config_path,
    )

    metrics: dict[str, dict[str, float | int]] = {}
    for split_name, prefix in (
        ("validation", "val_res"),
        ("test", "test_res"),
    ):
        result_pickle = find_prediction_pickle(
            experiment_dir,
            prefix,
            args.seed,
        )
        metrics[split_name] = save_prediction_outputs(
            result_pickle,
            split_name=split_name,
            output_dir=output_dir,
            target_label=args.target_column,
        )

    (output_dir / "regression_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print("\nFinal regression metrics:")
    for split_name, split_metrics in metrics.items():
        print(f"  {split_name}:")
        for metric_name, value in split_metrics.items():
            print(f"    {metric_name}: {value}")

    print(f"\nAll results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(
            f"\nERROR: Official FragNet training exited with code "
            f"{error.returncode}.",
            file=sys.stderr,
        )
        raise
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        raise
