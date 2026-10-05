#!/usr/bin/env python3
"""Stage 1: broad FragNet tuning on combined SMILES+GGS molecules.

Run this from the project directory that contains ``smiles_baseline2/`` and
``ggs_2/``. The script recursively reads every ``run_*/full_history.csv`` from
both trees, canonicalises and deduplicates molecules across both sources,
creates one leakage-controlled split, and tunes learning rate, dropout, and
batch size on the validation set. The held-out test set is prepared but never
evaluated here; a later fine-search/final-training script should reuse it.

The hyperparameter search follows Panapitiya et al., "FragNet: A Graph Neural
Network for Molecular Property Prediction with Four Levels of
Interpretability" (JACS 2026; preprint arXiv:2410.12156), plus the dropout
parameter present in the authors' official ``hp/hpoptuna.py`` implementation.
The regression-head shape and activation are deliberately fixed for this broad
stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine all SMILES and GGS runs and perform broad FragNet "
            "optimisation of learning rate, dropout, and batch size."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help=(
            "Project directory containing smiles_baseline2 and ggs_2. Defaults "
            "to the current directory (or the project directory inferred from "
            "the FragNet repository when run inside it). A single CSV file is "
            "also accepted for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=["smiles_baseline2", "ggs_2"],
        help=(
            "Dataset directories below --input-path. Every nested "
            "run_*/full_history.csv is included. Defaults: "
            "smiles_baseline2 ggs_2."
        ),
    )
    parser.add_argument(
        "--fragnet-root",
        type=Path,
        default=None,
        help=(
            "Root of the cloned PNNL FragNet repository. When omitted, the "
            "script also checks ./FragNet and "
            "./smiles_baseline2/FragNet."
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
            "<project>/fragnet_combined_<target-column>_broad_optuna."
        ),
    )
    parser.add_argument(
        "--split-method",
        choices=("paper-scaffold", "balanced-scaffold", "random"),
        default="paper-scaffold",
        help=(
            "paper-scaffold reproduces FragNet/Mole-BERT's deterministic "
            "Bemis-Murcko assignment; balanced-scaffold retains the original "
            "proportional-deficit splitter; random is provided only as a "
            "less stringent comparison."
        ),
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
        "--n-trials",
        type=int,
        default=20,
        help=(
            "Total number of broad Optuna trials. Defaults to 20. An "
            "interrupted study is resumed until this total is reached."
        ),
    )
    parser.add_argument(
        "--tuning-epochs",
        type=int,
        default=100,
        help="Maximum fine-tuning epochs in each Optuna trial.",
    )
    parser.add_argument(
        "--tuning-patience",
        type=int,
        default=20,
        help=(
            "Early-stopping patience used inside each broad-search trial. "
            "This is separate from --early-stopping-patience in the emitted "
            "configuration for later full training."
        ),
    )
    parser.add_argument(
        "--prune-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Optuna pruning of unpromising trials.",
    )
    parser.add_argument(
        "--skip-hyperparameter-search",
        action="store_true",
        help=(
            "Do not launch new trials; load the best completed trial from the "
            "existing Optuna study in the output directory."
        ),
    )
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
        help=(
            "Clean, split, generate graphs, and write the search YAML without "
            "running the broad Optuna search."
        ),
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
    else:
        candidates = [
            cwd,
            cwd.parent,
            cwd / "FragNet",
            cwd / "smiles_baseline2" / "FragNet",
            cwd / "ggs_2" / "FragNet",
        ]
        candidates.extend(sorted(cwd.glob("*/FragNet")))
        fragnet_root = next(
            (
                candidate.resolve()
                for candidate in candidates
                if is_fragnet_root(candidate)
            ),
            None,
        )
        if fragnet_root is None:
            raise FileNotFoundError(
                "Could not locate the FragNet repository. Supply "
                "--fragnet-root explicitly."
            )

    if not is_fragnet_root(fragnet_root):
        raise FileNotFoundError(
            f"This does not appear to be a FragNet repository: {fragnet_root}"
        )

    if args.input_path is not None:
        input_path = args.input_path.expanduser().resolve()
    elif is_fragnet_root(cwd):
        input_path = (
            cwd.parent.parent
            if cwd.parent.name in set(args.dataset_dirs)
            else cwd.parent
        )
    elif is_fragnet_root(cwd.parent):
        input_path = (
            cwd.parent.parent.parent
            if cwd.parent.parent.name in set(args.dataset_dirs)
            else cwd.parent.parent
        )
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
        output_parent = input_path.parent if input_path.is_file() else input_path
        output_dir = (
            output_parent / f"fragnet_combined_{safe_target}_broad_optuna"
        )

    return fragnet_root, input_path, output_dir


def resolve_dataset_paths(
    input_path: Path,
    args: argparse.Namespace,
) -> dict[str, Path]:
    """Resolve the named SMILES/GGS roots without silently omitting either."""
    if input_path.is_file():
        return {input_path.stem: input_path}

    resolved = {
        Path(name).name: (
            Path(name).expanduser().resolve()
            if Path(name).expanduser().is_absolute()
            else (input_path / name).resolve()
        )
        for name in args.dataset_dirs
    }
    missing = [str(path) for path in resolved.values() if not path.exists()]

    if not missing:
        return resolved

    # Retain support for the old single-dataset invocation where --input-path
    # itself contains the run directories. Never use this fallback when one of
    # the two requested combined roots exists, because that could silently train
    # on only SMILES or only GGS.
    any_named_root_exists = any(path.exists() for path in resolved.values())
    contains_runs = any(input_path.rglob("full_history.csv"))
    if not any_named_root_exists and contains_runs:
        return {input_path.name: input_path}

    raise FileNotFoundError(
        "The following requested dataset directories do not exist: "
        + ", ".join(missing)
    )


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
    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1.")
    if args.tuning_epochs < 1:
        raise ValueError("--tuning-epochs must be at least 1.")
    if args.tuning_patience < 1:
        raise ValueError("--tuning-patience must be at least 1.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and greater than zero.")


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


def find_input_files(dataset_paths: dict[str, Path]) -> list[tuple[str, Path]]:
    """Find every run history recursively, including nested GGS layouts."""
    labelled_files: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    for dataset_name, dataset_path in dataset_paths.items():
        if dataset_path.is_file():
            candidates = [dataset_path]
        else:
            candidates = [
                path
                for path in sorted(dataset_path.rglob("full_history.csv"))
                if any(part.startswith("run_") for part in path.parts)
            ]

        if not candidates:
            raise FileNotFoundError(
                "No nested run_*/full_history.csv files were found under "
                f"{dataset_path}."
            )

        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                labelled_files.append((dataset_name, resolved))
                seen.add(resolved)

    return labelled_files


def clean_smiles_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    return text


def load_and_clean(
    dataset_paths: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    labelled_files = find_input_files(dataset_paths)
    input_files = [path for _, path in labelled_files]
    print(f"Found {len(input_files):,} input file(s) across:")
    for dataset_name, dataset_path in dataset_paths.items():
        count = sum(name == dataset_name for name, _ in labelled_files)
        print(f"  {dataset_name}: {count:,} file(s) under {dataset_path}")

    pieces: list[pd.DataFrame] = []
    for dataset_name, path in labelled_files:
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
        piece["source_dataset"] = dataset_name
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
                "source_dataset": row.source_dataset,
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
                source_dataset=(
                    "source_dataset",
                    lambda values: "|".join(sorted(set(values))),
                ),
                source_file=(
                    "source_file",
                    lambda values: "|".join(sorted(set(values))),
                ),
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
        "input_files_by_dataset": {
            name: int(sum(label == name for label, _ in labelled_files))
            for name in dataset_paths
        },
        "invalid_smiles_examples": invalid_smiles_examples,
    }
    return cleaned, input_files, report


def scaffold_key(smiles: str, *, group_acyclic: bool = True) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Unexpected invalid canonical SMILES: {smiles}")

    scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=True,
    )

    # The paper/Mole-BERT implementation groups all empty scaffolds together.
    # The older balanced splitter can still treat each acyclic molecule as its
    # own group to avoid a single oversized acyclic set.
    if scaffold_smiles or group_acyclic:
        return scaffold_smiles
    return f"ACYCLIC::{smiles}"


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
        "val": shuffled.iloc[
            n_train:n_train + n_validation
        ].copy().reset_index(drop=True),
        "test": shuffled.iloc[n_train + n_validation:].copy().reset_index(drop=True),
    }


def paper_scaffold_split(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    """Reproduce FragNet's Mole-BERT/DeepChem 80:10:10 scaffold logic."""
    groups: dict[str, list[int]] = {}
    for index, smiles in data["smiles"].items():
        groups.setdefault(
            scaffold_key(smiles, group_acyclic=True), []
        ).append(index)

    grouped_indices = [
        indices
        for _, indices in sorted(
            groups.items(),
            key=lambda item: (len(item[1]), item[1][0]),
            reverse=True,
        )
    ]
    train_cutoff = args.train_fraction * len(data)
    validation_cutoff = (
        args.train_fraction + args.validation_fraction
    ) * len(data)
    assigned: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for group in grouped_indices:
        if len(assigned["train"]) + len(group) <= train_cutoff:
            assigned["train"].extend(group)
        elif (
            len(assigned["train"])
            + len(assigned["val"])
            + len(group)
            <= validation_cutoff
        ):
            assigned["val"].extend(group)
        else:
            assigned["test"].extend(group)

    result = {
        name: data.loc[indices].reset_index(drop=True)
        for name, indices in assigned.items()
    }
    if any(frame.empty for frame in result.values()):
        raise ValueError(
            "Paper-style scaffold splitting produced an empty split. This can "
            "happen when one scaffold group is unusually large; use "
            "--split-method balanced-scaffold in that case."
        )
    return result


def balanced_scaffold_split(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    groups: dict[str, list[int]] = {}
    for index, smiles in data["smiles"].items():
        groups.setdefault(
            scaffold_key(smiles, group_acyclic=False), []
        ).append(index)

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


def verify_scaffold_split_integrity(
    splits: dict[str, pd.DataFrame],
    *,
    group_acyclic: bool,
) -> None:
    scaffold_sets = {
        name: {
            scaffold_key(smiles, group_acyclic=group_acyclic)
            for smiles in frame["smiles"]
        }
        for name, frame in splits.items()
    }
    names = list(splits)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1:]:
            overlap = scaffold_sets[first_name].intersection(
                scaffold_sets[second_name]
            )
            if overlap:
                raise RuntimeError(
                    "Murcko scaffold leakage between "
                    f"{first_name} and {second_name}: {len(overlap)} "
                    "scaffold group(s)."
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
    model_params: dict[str, Any] | None = None,
) -> None:
    model_params = model_params or {}
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
            "batch_size": int(
                model_params.get("batch_size", args.batch_size)
            ),
            "lr": float(
                model_params.get("learning_rate", args.learning_rate)
            ),
            "model": {
                "n_classes": 1,
                "num_layer": 4,
                "drop_ratio": float(model_params.get("drop_ratio", 0.1)),
                "num_heads": 4,
                "emb_dim": 128,
                "h1": int(model_params.get("h1", 128)),
                "h2": int(model_params.get("h2", 1024)),
                "h3": int(model_params.get("h3", 1024)),
                "h4": int(model_params.get("h4", 512)),
                "act": str(model_params.get("act", "relu")),
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


BROAD_HYPERPARAMETER_SEARCH_SPACE: dict[str, Any] = {
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-3,
        "log": True,
    },
    "drop_ratio": {
        "type": "float",
        "low": 0.0,
        "high": 0.4,
        "step": 0.05,
    },
    "batch_size": [16, 32, 64, 128],
}

FIXED_HEAD_PARAMETERS: dict[str, Any] = {
    "h1": 128,
    "h2": 1024,
    "h3": 1024,
    "h4": 512,
    "act": "relu",
    "fthead": "FTHead3",
}


def load_optuna_module() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "Optuna is required for hyperparameter search. Install it in the "
            "FragNet environment with: pip install optuna"
        ) from error
    return optuna


def seed_tuning_trial(seed: int, torch: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scalar_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def run_broad_trial(
    trial: Any,
    *,
    config: dict[str, Any],
    train_dataset: Any,
    validation_dataset: Any,
    tuning_dir: Path,
    args: argparse.Namespace,
) -> float:
    """Train one broad-search trial and return its best validation MSE."""
    import gc

    import torch
    from torch.utils.data import DataLoader

    from fragnet.dataset.data import collate_fn
    from fragnet.model.gat.gat2 import FragNetFineTune
    from fragnet.train.utils import EarlyStopping
    from fragnet.train.utils import TrainerFineTune as Trainer

    seed_tuning_trial(args.seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True,
    )
    drop_ratio = trial.suggest_float(
        "drop_ratio",
        0.0,
        0.4,
        step=0.05,
    )
    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64, 128],
    )

    finetune = config["finetune"]
    model_config = finetune["model"]
    pretrain = config["pretrain"]
    trial_checkpoint = tuning_dir / f"trial_{trial.number}.pt"
    model = None

    try:
        model = FragNetFineTune(
            n_classes=model_config["n_classes"],
            atom_features=config["atom_features"],
            frag_features=config["frag_features"],
            edge_features=config["edge_features"],
            num_heads=model_config["num_heads"],
            num_layer=model_config["num_layer"],
            drop_ratio=drop_ratio,
            h1=FIXED_HEAD_PARAMETERS["h1"],
            h2=FIXED_HEAD_PARAMETERS["h2"],
            h3=FIXED_HEAD_PARAMETERS["h3"],
            h4=FIXED_HEAD_PARAMETERS["h4"],
            act=FIXED_HEAD_PARAMETERS["act"],
            fthead=FIXED_HEAD_PARAMETERS["fthead"],
        )

        pretrained_checkpoint = pretrain.get("chkpoint_name")
        if pretrained_checkpoint:
            from fragnet.model.gat.gat2_pretrain import FragNetPreTrain

            pretrained_model = FragNetPreTrain(
                atom_features=config["atom_features"],
                frag_features=config["frag_features"],
                edge_features=config["edge_features"],
                num_layer=pretrain["num_layer"],
                drop_ratio=pretrain["drop_ratio"],
                num_heads=pretrain["num_heads"],
                emb_dim=pretrain["emb_dim"],
            )
            pretrained_model.load_state_dict(
                torch.load(pretrained_checkpoint, map_location=device)
            )
            model.pretrain.load_state_dict(
                pretrained_model.pretrain.state_dict()
            )
            del pretrained_model

        train_loader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        validation_loader = DataLoader(
            validation_dataset,
            collate_fn=collate_fn,
            batch_size=128,
            shuffle=False,
            drop_last=False,
        )

        model.to(device)
        trainer = Trainer(target_type=finetune["target_type"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
        early_stopping = EarlyStopping(
            patience=args.tuning_patience,
            verbose=True,
            chkpoint_name=str(trial_checkpoint),
        )

        for epoch in range(args.tuning_epochs):
            trainer.train_regr(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=None,
                device=device,
                val_loader=validation_loader,
            )
            validation_loss, _, _ = trainer.test_regr(
                model=model,
                loader=validation_loader,
                device=device,
            )
            validation_mse = scalar_float(validation_loss)
            trial.report(validation_mse, epoch)

            early_stopping(validation_mse, model)
            if args.prune_trials and trial.should_prune():
                optuna = load_optuna_module()
                raise optuna.TrialPruned()
            if early_stopping.early_stop:
                break

        if not trial_checkpoint.exists():
            raise RuntimeError(
                f"Trial {trial.number} did not produce an early-stopping "
                "checkpoint."
            )

        model.load_state_dict(
            torch.load(trial_checkpoint, map_location=device)
        )
        best_validation_loss, _, _ = trainer.test_regr(
            model=model,
            loader=validation_loader,
            device=device,
        )
        return scalar_float(best_validation_loss)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            trial.set_user_attr("failure", "out of memory")
            optuna = load_optuna_module()
            raise optuna.TrialPruned() from error
        raise
    finally:
        trial_checkpoint.unlink(missing_ok=True)
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def run_hyperparameter_search(
    *,
    fragnet_root: Path,
    config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run/resume the broad Optuna search on validation MSE."""
    print("\nFragNet broad Optuna search space:")
    print("  learning_rate: 1e-5..1e-3 (logarithmic)")
    print("  drop_ratio: 0.0..0.4 in steps of 0.05")
    print("  batch_size: 16, 32, 64, 128")
    print(
        "  fixed regression head: h1=128, h2=1024, h3=1024, "
        "h4=512, activation=relu"
    )
    optuna = load_optuna_module()
    optuna_dir = output_dir / "broad_optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    database_path = optuna_dir / "fragnet_broad_optuna.db"
    study_name = "fragnet_broad_optuna"
    study_previously_existed = database_path.exists()
    if args.skip_hyperparameter_search and not study_previously_existed:
        raise FileNotFoundError(
            "--skip-hyperparameter-search was supplied, but no existing "
            f"study was found at {database_path}."
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = (
        optuna.pruners.MedianPruner()
        if args.prune_trials
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{database_path}",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    finished_trials = sum(
        trial.state.name not in {"RUNNING", "WAITING"}
        for trial in study.trials
    )

    remaining_trials = max(0, args.n_trials - finished_trials)
    if args.skip_hyperparameter_search:
        print(
            "\nSkipping new broad trials and loading the existing study: "
            f"{database_path}"
        )
    elif remaining_trials > 0:
        from fragnet.dataset.dataset import load_pickle_dataset

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        train_dataset = load_pickle_dataset(
            config["finetune"]["train"]["path"]
        )
        validation_dataset = load_pickle_dataset(
            config["finetune"]["val"]["path"]
        )

        print("\nStarting FragNet broad Optuna search:")
        print(
            f"  completed/resumable trials: {finished_trials}/{args.n_trials}; "
            f"launching {remaining_trials}"
        )
        study.optimize(
            lambda trial: run_broad_trial(
                trial,
                config=config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                tuning_dir=optuna_dir,
                args=args,
            ),
            n_trials=remaining_trials,
            gc_after_trial=True,
        )
    else:
        print(
            f"\nBroad study already has {finished_trials} finished trial(s); "
            "no new trials are required."
        )

    try:
        best_trial = study.best_trial
    except ValueError as error:
        raise RuntimeError(
            "The Optuna study has no completed trial from which to select "
            "hyperparameters."
        ) from error

    required = {"learning_rate", "drop_ratio", "batch_size"}
    missing = required.difference(best_trial.params)
    if missing:
        raise RuntimeError(
            "Best Optuna trial is missing expected FragNet parameters: "
            f"{sorted(missing)}"
        )

    best_params = {
        key: best_trial.params[key]
        for key in sorted(required)
    }
    trials_csv = optuna_dir / "broad_optuna_trials.csv"
    study.trials_dataframe(
        attrs=("number", "value", "params", "state")
    ).to_csv(trials_csv, index=False)

    summary = {
        "method": (
            "Seeded Optuna TPESampler; MedianPruner when enabled"
        ),
        "stage": "broad optimisation",
        "selection_data": "validation split only",
        "objective": "minimise validation MSE",
        "requested_total_trials": int(args.n_trials),
        "finished_trials": int(
            sum(
                trial.state.name not in {"RUNNING", "WAITING"}
                for trial in study.trials
            )
        ),
        "best_trial_number": int(best_trial.number),
        "best_validation_mse": float(best_trial.value),
        "best_validation_rmse": float(math.sqrt(best_trial.value)),
        "best_params": best_params,
        "search_space": BROAD_HYPERPARAMETER_SEARCH_SPACE,
        "fixed_during_search": {
            "backbone_layers": 4,
            "attention_heads": 4,
            "embedding_dimension": 128,
            "regression_head": FIXED_HEAD_PARAMETERS,
            "loss": "MSE",
        },
        "test_set_evaluated": False,
        "study_database": str(database_path),
        "trials_csv": str(trials_csv),
    }
    (optuna_dir / "best_broad_hyperparameters.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nBest broad validation hyperparameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    print(f"  validation MSE: {summary['best_validation_mse']:.6g}")
    print(f"  validation RMSE: {summary['best_validation_rmse']:.6g}")
    return best_params, summary


def main() -> int:
    args = parse_args()
    validate_arguments(args)

    fragnet_root, input_path, output_dir = resolve_paths(args)
    dataset_paths = resolve_dataset_paths(input_path, args)
    configure_fragnet_import(fragnet_root)

    print("Resolved paths:")
    print(f"  FragNet repository: {fragnet_root}")
    print(f"  Project input:      {input_path}")
    for dataset_name, dataset_path in dataset_paths.items():
        print(f"  Dataset {dataset_name}: {dataset_path}")
    print(f"  Output directory:   {output_dir}")
    print(f"  Target column:      {args.target_column}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    graph_dir = output_dir / "graph_data"
    experiment_dir = output_dir / "experiment"
    for directory in (split_dir, graph_dir, experiment_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data, input_files, cleaning_report = load_and_clean(dataset_paths, args)

    print("\nCleaning summary:")
    for key, value in cleaning_report.items():
        if key != "invalid_smiles_examples":
            print(f"  {key}: {value}")

    if cleaning_report["invalid_smiles_examples"]:
        print("  Invalid SMILES examples:")
        for example in cleaning_report["invalid_smiles_examples"]:
            print(f"    {example}")

    if args.split_method == "paper-scaffold":
        splits = paper_scaffold_split(data, args)
    elif args.split_method == "balanced-scaffold":
        splits = balanced_scaffold_split(data, args)
    else:
        splits = random_split(data, args)

    verify_split_integrity(splits)
    if args.split_method in {"paper-scaffold", "balanced-scaffold"}:
        verify_scaffold_split_integrity(
            splits,
            group_acyclic=args.split_method == "paper-scaffold",
        )

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

    search_config_path = output_dir / "fragnet_broad_search.yaml"
    write_fragnet_config(
        config_path=search_config_path,
        experiment_dir=experiment_dir,
        graph_paths=graph_paths,
        checkpoint=checkpoint,
        args=args,
    )
    print(f"Saved FragNet search configuration: {search_config_path}")

    manifest = {
        "paper": {
            "title": (
                "FragNet: A Graph Neural Network for Molecular Property "
                "Prediction with Four Levels of Interpretability"
            ),
            "doi": "10.1021/jacs.5c22620",
            "arxiv": "2410.12156",
        },
        "fragnet_root": str(fragnet_root),
        "input_path": str(input_path),
        "dataset_paths": {
            name: str(path) for name, path in dataset_paths.items()
        },
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
        "hyperparameter_search_space": BROAD_HYPERPARAMETER_SEARCH_SPACE,
        "fixed_head_parameters": FIXED_HEAD_PARAMETERS,
        "test_set_policy": (
            "Held out from the broad search and not evaluated by this script. "
            "The future fine-search/final-training stage must reuse it."
        ),
        "configuration": vars(args),
    }
    # Convert any Path values in configuration to strings.
    manifest["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in manifest["configuration"].items()
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    if args.prepare_only:
        print("\nPreparation completed; broad Optuna search was not started.")
        print("Resume from the same project directory with:")
        print(
            f"  python {Path(__file__).resolve()} --skip-graph-generation"
        )
        return 0

    best_params, search_summary = run_hyperparameter_search(
        fragnet_root=fragnet_root,
        config_path=search_config_path,
        output_dir=output_dir,
        args=args,
    )
    manifest["hyperparameter_search"] = search_summary

    config_path = output_dir / "fragnet_broad_optimised.yaml"
    write_fragnet_config(
        config_path=config_path,
        experiment_dir=experiment_dir,
        graph_paths=graph_paths,
        checkpoint=checkpoint,
        args=args,
        model_params=best_params,
    )
    manifest["broad_optimised_config_path"] = str(config_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(f"Saved broad-optimised configuration: {config_path}")
    print(
        "\nBroad search complete. No final model was trained and the test set "
        "was not evaluated. Reuse the saved splits, graph_data, and "
        "fragnet_broad_optimised.yaml in the later fine-search script."
    )
    print(f"\nAll broad-search results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        raise
