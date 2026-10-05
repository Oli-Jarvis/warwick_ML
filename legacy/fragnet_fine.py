#!/usr/bin/env python3
"""Stage 2: fine FragNet hyperparameter optimisation and optional final fit.

This script continues the broad search produced by ``fragnet_broad.py``.  It
does not rebuild or resplit the data.  Instead, it reuses:

* ``graph_data/train.pkl`` and ``graph_data/val.pkl``;
* ``fragnet_broad_optimised.yaml``;
* ``broad_optuna/best_broad_hyperparameters.json``.

The winning broad parameters (learning rate, dropout and batch size) are held
fixed.  The fine stage tunes the four widths and activation of FragNet's
``FTHead3`` regression head, using the search space in the official PNNL
``fragnet/hp/hpoptuna.py`` implementation.  The held-out test pickle is not
loaded during either the Optuna search or the multi-seed confirmation step.

By default the script performs the search, confirms the leading configurations
over three random seeds, and writes ``fragnet_fine_optimised.yaml``.  Add
``--train-final`` to train the selected model and evaluate the test set once.

Typical use on the HPC, from ``/storage/msszkb_grp/msshfg``::

    python fragnet_fine.py

To search and then fit/evaluate the final model in one job::

    python fragnet_fine.py --train-final

The Optuna study is persistent.  Increasing ``--n-trials`` resumes the same
study until that total number of finished trials has been reached.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ACTIVATIONS = [
    "relu",
    "silu",
    "gelu",
    "celu",
    "selu",
    "rrelu",
    "relu6",
    "prelu",
    "leakyrelu",
]

BASELINE_HEAD = {
    "h1": 128,
    "h2": 1024,
    "h3": 1024,
    "h4": 512,
    "act": "relu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse a completed FragNet broad search, tune FTHead3, and "
            "optionally train/evaluate the final model."
        )
    )
    parser.add_argument(
        "--broad-dir",
        type=Path,
        default=Path("fragnet_combined_log_P_upconversion_broad_optuna"),
        help="Output directory created by fragnet_broad.py.",
    )
    parser.add_argument(
        "--fragnet-root",
        type=Path,
        default=None,
        help=(
            "Root of the cloned PNNL FragNet repository. If omitted, use the "
            "path recorded by the broad-stage manifest or common local paths."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help=(
            "Total number of fine-search trials, including existing trials. "
            "Use a larger value later to resume the same study."
        ),
    )
    parser.add_argument("--tuning-epochs", type=int, default=100)
    parser.add_argument("--tuning-patience", type=int, default=20)
    parser.add_argument(
        "--prune-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prune weak trials after a conservative warm-up period.",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Load the existing fine Optuna study without launching new trials.",
    )
    parser.add_argument(
        "--confirm-top-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Retrain the leading completed trials over multiple seeds and "
            "select by mean validation MSE."
        ),
    )
    parser.add_argument("--confirm-top-k", type=int, default=3)
    parser.add_argument(
        "--confirmation-seeds",
        nargs="+",
        type=int,
        default=[123, 456, 789],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Optuna sampler seed and fixed training seed used during search.",
    )
    parser.add_argument(
        "--head-min",
        type=int,
        default=64,
        help="Smallest FTHead3 hidden width.",
    )
    parser.add_argument(
        "--head-max",
        type=int,
        default=2048,
        help="Largest FTHead3 hidden width.",
    )
    parser.add_argument(
        "--head-step",
        type=int,
        default=64,
        help="Step between allowed FTHead3 hidden widths.",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--train-final",
        action="store_true",
        help=(
            "After model selection, train the selected configuration and "
            "evaluate validation and test once."
        ),
    )
    parser.add_argument("--final-seed", type=int, default=123)
    parser.add_argument("--final-epochs", type=int, default=10000)
    parser.add_argument("--final-patience", type=int, default=100)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/configuration and print the search plan only.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "n_trials": args.n_trials,
        "tuning_epochs": args.tuning_epochs,
        "tuning_patience": args.tuning_patience,
        "confirm_top_k": args.confirm_top_k,
        "head_min": args.head_min,
        "head_max": args.head_max,
        "head_step": args.head_step,
        "validation_batch_size": args.validation_batch_size,
        "final_epochs": args.final_epochs,
        "final_patience": args.final_patience,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.head_min > args.head_max:
        raise ValueError("--head-min cannot exceed --head-max.")
    if (args.head_max - args.head_min) % args.head_step != 0:
        raise ValueError(
            "(--head-max - --head-min) must be divisible by --head-step."
        )
    if not args.confirmation_seeds:
        raise ValueError("At least one --confirmation-seeds value is required.")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return value


def is_fragnet_root(path: Path) -> bool:
    return (
        (path / "setup.py").exists()
        and (path / "fragnet" / "model" / "gat" / "gat2.py").exists()
        and (path / "fragnet" / "dataset" / "dataset.py").exists()
    )


def resolve_fragnet_root(
    requested: Path | None,
    broad_dir: Path,
) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested.expanduser())

    manifest_path = broad_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        recorded = manifest.get("fragnet_root")
        if recorded:
            candidates.append(Path(str(recorded)).expanduser())

    candidates.extend(
        [
            Path.cwd() / "FragNet",
            Path.cwd() / "smiles_baseline2" / "FragNet",
            broad_dir.parent / "FragNet",
            broad_dir.parent / "smiles_baseline2" / "FragNet",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if is_fragnet_root(candidate):
            return candidate

    attempted = "\n".join(f"  {path}" for path in seen)
    raise FileNotFoundError(
        "Could not locate the PNNL FragNet repository. Tried:\n" + attempted
    )


def configure_fragnet_import(fragnet_root: Path) -> None:
    root_text = str(fragnet_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def resolve_graph_paths(
    config: dict[str, Any],
    broad_dir: Path,
) -> dict[str, Path]:
    try:
        finetune = config["finetune"]
    except KeyError as error:
        raise KeyError("The broad YAML has no 'finetune' section.") from error

    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        configured_text = finetune.get(split, {}).get("path")
        configured = (
            Path(str(configured_text)).expanduser()
            if configured_text
            else Path("__missing__")
        )
        fallback = broad_dir / "graph_data" / f"{split}.pkl"
        if configured.exists():
            chosen = configured.resolve()
        elif fallback.exists():
            chosen = fallback.resolve()
            print(
                f"WARNING: configured {split} graph path was unavailable; "
                f"using {chosen}"
            )
        else:
            raise FileNotFoundError(
                f"Could not find the {split} graph pickle at either "
                f"{configured} or {fallback}."
            )
        paths[split] = chosen
        config["finetune"].setdefault(split, {})["path"] = str(chosen)
    return paths


def load_broad_inputs(
    broad_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    config_path = broad_dir / "fragnet_broad_optimised.yaml"
    summary_path = broad_dir / "broad_optuna" / "best_broad_hyperparameters.json"
    config = read_yaml(config_path)
    summary = read_json(summary_path)

    if config.get("model_version") != "gat2":
        raise ValueError(
            "This script expects model_version='gat2' in the broad YAML."
        )
    model = config.get("finetune", {}).get("model", {})
    if model.get("fthead") != "FTHead3":
        raise ValueError(
            "This fine search tunes FTHead3, but the broad YAML uses "
            f"{model.get('fthead')!r}."
        )

    broad_params = summary.get("best_params")
    if not isinstance(broad_params, dict):
        raise KeyError("Broad summary has no 'best_params' mapping.")
    required = {"learning_rate", "drop_ratio", "batch_size"}
    missing = required.difference(broad_params)
    if missing:
        raise KeyError(f"Broad summary is missing: {sorted(missing)}")
    locked = {
        "learning_rate": float(broad_params["learning_rate"]),
        "drop_ratio": float(broad_params["drop_ratio"]),
        "batch_size": int(broad_params["batch_size"]),
    }
    if not math.isfinite(locked["learning_rate"]) or locked["learning_rate"] <= 0:
        raise ValueError("The broad learning rate must be finite and positive.")
    if not 0 <= locked["drop_ratio"] < 1:
        raise ValueError("The broad drop ratio must lie in [0, 1).")
    if locked["batch_size"] < 1:
        raise ValueError("The broad batch size must be positive.")

    # The JSON is authoritative because it records the actual winning trial.
    config["finetune"]["lr"] = locked["learning_rate"]
    config["finetune"]["batch_size"] = locked["batch_size"]
    config["finetune"]["model"]["drop_ratio"] = locked["drop_ratio"]
    graph_paths = resolve_graph_paths(config, broad_dir)
    return config, summary, locked, graph_paths


def choose_device(requested: str, torch: Any) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_load(path: Path, device: Any, torch: Any) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def unwrap_state_dict(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested
    return value


def build_model(
    config: dict[str, Any],
    head_params: dict[str, Any],
) -> Any:
    from fragnet.model.gat.gat2 import FragNetFineTune

    model_config = config["finetune"]["model"]
    return FragNetFineTune(
        n_classes=int(model_config["n_classes"]),
        atom_features=int(config["atom_features"]),
        frag_features=int(config["frag_features"]),
        edge_features=int(config["edge_features"]),
        num_heads=int(model_config["num_heads"]),
        num_layer=int(model_config["num_layer"]),
        drop_ratio=float(model_config["drop_ratio"]),
        emb_dim=int(model_config.get("emb_dim", 128)),
        h1=int(head_params["h1"]),
        h2=int(head_params["h2"]),
        h3=int(head_params["h3"]),
        h4=int(head_params["h4"]),
        act=str(head_params["act"]),
        fthead="FTHead3",
    )


def load_pretrained_weights(
    model: Any,
    config: dict[str, Any],
    device: Any,
    torch: Any,
) -> None:
    checkpoint_text = config.get("pretrain", {}).get("chkpoint_name")
    if not checkpoint_text:
        return
    checkpoint = Path(str(checkpoint_text)).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint}")

    from fragnet.model.gat.gat2_pretrain import FragNetPreTrain

    pretrain = config["pretrain"]
    pretrained_model = FragNetPreTrain(
        atom_features=int(config["atom_features"]),
        frag_features=int(config["frag_features"]),
        edge_features=int(config["edge_features"]),
        num_layer=int(pretrain["num_layer"]),
        drop_ratio=float(pretrain["drop_ratio"]),
        num_heads=int(pretrain["num_heads"]),
        emb_dim=int(pretrain["emb_dim"]),
    )
    state = unwrap_state_dict(torch_load(checkpoint, device, torch))
    pretrained_model.load_state_dict(state)
    model.pretrain.load_state_dict(pretrained_model.pretrain.state_dict())
    del pretrained_model, state


def scalar_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = float(np.asarray(value).reshape(-1)[0])
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite loss encountered: {result}")
    return result


def run_one_training(
    *,
    config: dict[str, Any],
    head_params: dict[str, Any],
    train_dataset: Any,
    validation_dataset: Any,
    checkpoint_path: Path,
    seed: int,
    max_epochs: int,
    patience: int,
    validation_batch_size: int,
    requested_device: str,
    trial: Any | None = None,
    prune_trials: bool = False,
    keep_checkpoint: bool = False,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from fragnet.dataset.data import collate_fn
    from fragnet.train.utils import TrainerFineTune as Trainer

    seed_everything(seed, torch)
    device = choose_device(requested_device, torch)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.unlink(missing_ok=True)
    model = None
    best_mse = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    epochs_run = 0

    try:
        model = build_model(config, head_params)
        load_pretrained_weights(model, config, device, torch)
        model.to(device)

        train_loader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=int(config["finetune"]["batch_size"]),
            shuffle=True,
            drop_last=True,
        )
        validation_loader = DataLoader(
            validation_dataset,
            collate_fn=collate_fn,
            batch_size=validation_batch_size,
            shuffle=False,
            drop_last=False,
        )
        trainer = Trainer(target_type=config["finetune"]["target_type"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["finetune"]["lr"]),
        )

        for epoch in range(max_epochs):
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
            epochs_run = epoch + 1

            if trial is not None:
                trial.report(validation_mse, epoch)

            if validation_mse < best_mse:
                best_mse = validation_mse
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            if (
                trial is not None
                and prune_trials
                and trial.should_prune()
            ):
                import optuna

                raise optuna.TrialPruned()
            if epochs_without_improvement >= patience:
                break

        if not checkpoint_path.exists():
            raise RuntimeError("Training did not produce a checkpoint.")
        model.load_state_dict(torch_load(checkpoint_path, device, torch))
        validation_loss, _, _ = trainer.test_regr(
            model=model,
            loader=validation_loader,
            device=device,
        )
        restored_mse = scalar_float(validation_loss)
        parameter_count = int(sum(p.numel() for p in model.parameters()))
        trainable_parameter_count = int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
        return {
            "validation_mse": restored_mse,
            "validation_rmse": math.sqrt(restored_mse),
            "best_epoch_zero_based": int(best_epoch),
            "epochs_run": int(epochs_run),
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
        }
    finally:
        if not keep_checkpoint:
            checkpoint_path.unlink(missing_ok=True)
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def load_optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "Optuna is required. Install it in the FragNet environment with "
            "'pip install optuna'."
        ) from error
    return optuna


def suggested_head_params(trial: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "h1": trial.suggest_int(
            "h1", args.head_min, args.head_max, step=args.head_step
        ),
        "h2": trial.suggest_int(
            "h2", args.head_min, args.head_max, step=args.head_step
        ),
        "h3": trial.suggest_int(
            "h3", args.head_min, args.head_max, step=args.head_step
        ),
        "h4": trial.suggest_int(
            "h4", args.head_min, args.head_max, step=args.head_step
        ),
        "act": trial.suggest_categorical("act", ACTIVATIONS),
    }


def run_fine_study(
    *,
    broad_dir: Path,
    config: dict[str, Any],
    train_dataset: Any,
    validation_dataset: Any,
    args: argparse.Namespace,
) -> tuple[Any, Path]:
    optuna = load_optuna()
    fine_dir = broad_dir / "fine_optuna"
    fine_dir.mkdir(parents=True, exist_ok=True)
    database_path = fine_dir / "fragnet_fine_optuna.db"
    study_name = "fragnet_fine_optuna"
    existed = database_path.exists()
    if args.skip_search and not existed:
        raise FileNotFoundError(
            "--skip-search was supplied, but no fine study exists at "
            f"{database_path}."
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=8,
            n_warmup_steps=10,
            interval_steps=1,
        )
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
    if not existed:
        # Guarantees an apples-to-apples FTHead3 baseline in the fine study.
        study.enqueue_trial(BASELINE_HEAD)

    finished = sum(
        trial.state.name not in {"RUNNING", "WAITING"}
        for trial in study.trials
    )
    remaining = max(0, args.n_trials - finished)

    if args.skip_search:
        print(f"Loading existing fine study: {database_path}")
    elif remaining:
        print(
            f"Starting/resuming fine search: {finished}/{args.n_trials} "
            f"finished; launching {remaining} trial(s)."
        )

        def objective(trial: Any) -> float:
            head_params = suggested_head_params(trial, args)
            checkpoint = fine_dir / f"trial_{trial.number}.pt"
            try:
                result = run_one_training(
                    config=config,
                    head_params=head_params,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    checkpoint_path=checkpoint,
                    seed=args.seed,
                    max_epochs=args.tuning_epochs,
                    patience=args.tuning_patience,
                    validation_batch_size=args.validation_batch_size,
                    requested_device=args.device,
                    trial=trial,
                    prune_trials=args.prune_trials,
                )
                for key, value in result.items():
                    trial.set_user_attr(key, value)
                return float(result["validation_mse"])
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    trial.set_user_attr("failure", "out of memory")
                    raise optuna.TrialPruned() from error
                raise

        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
        )
    else:
        print(
            f"Fine study already has {finished} finished trial(s); "
            "no new trials are required."
        )

    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not complete:
        raise RuntimeError("The fine study contains no completed trial.")
    trials_csv = fine_dir / "fine_optuna_trials.csv"
    study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    ).to_csv(trials_csv, index=False)
    return study, trials_csv


def complete_trials_in_rank_order(study: Any) -> list[Any]:
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    return sorted(complete, key=lambda trial: (float(trial.value), trial.number))


def confirm_candidates(
    *,
    study: Any,
    broad_dir: Path,
    config: dict[str, Any],
    train_dataset: Any,
    validation_dataset: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ranked = complete_trials_in_rank_order(study)
    fine_dir = broad_dir / "fine_optuna"

    if not args.confirm_top_candidates:
        winner = ranked[0]
        params = dict(winner.params)
        selection = {
            "method": "single-seed Optuna validation winner",
            "selected_trial_number": int(winner.number),
            "selected_mean_validation_mse": float(winner.value),
            "selected_mean_validation_rmse": math.sqrt(float(winner.value)),
            "confirmation_performed": False,
        }
        return params, selection

    candidates = ranked[: min(args.confirm_top_k, len(ranked))]
    rows: list[dict[str, Any]] = []
    print(
        f"Confirming the top {len(candidates)} configuration(s) over "
        f"seeds {args.confirmation_seeds}."
    )
    for candidate in candidates:
        for seed in args.confirmation_seeds:
            checkpoint = (
                fine_dir
                / "confirmation_checkpoints"
                / f"trial_{candidate.number}_seed_{seed}.pt"
            )
            result = run_one_training(
                config=config,
                head_params=dict(candidate.params),
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                checkpoint_path=checkpoint,
                seed=seed,
                max_epochs=args.tuning_epochs,
                patience=args.tuning_patience,
                validation_batch_size=args.validation_batch_size,
                requested_device=args.device,
            )
            rows.append(
                {
                    "trial_number": int(candidate.number),
                    "seed": int(seed),
                    **dict(candidate.params),
                    **result,
                }
            )

    confirmation = pd.DataFrame(rows)
    confirmation_path = fine_dir / "fine_candidate_confirmation.csv"
    confirmation.to_csv(confirmation_path, index=False)
    aggregate = (
        confirmation.groupby("trial_number", as_index=False)
        .agg(
            mean_validation_mse=("validation_mse", "mean"),
            sd_validation_mse=("validation_mse", "std"),
            mean_validation_rmse=("validation_rmse", "mean"),
            mean_best_epoch_zero_based=("best_epoch_zero_based", "mean"),
        )
        .sort_values(
            ["mean_validation_mse", "sd_validation_mse"],
            na_position="last",
        )
    )
    aggregate_path = fine_dir / "fine_candidate_confirmation_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    selected_number = int(aggregate.iloc[0]["trial_number"])
    selected_trial = next(
        trial for trial in candidates if trial.number == selected_number
    )
    selected_row = aggregate.iloc[0]
    sd_value = selected_row["sd_validation_mse"]
    selection = {
        "method": "lowest mean validation MSE over confirmation seeds",
        "confirmation_performed": True,
        "top_k_confirmed": int(len(candidates)),
        "confirmation_seeds": [int(seed) for seed in args.confirmation_seeds],
        "selected_trial_number": selected_number,
        "selected_mean_validation_mse": float(
            selected_row["mean_validation_mse"]
        ),
        "selected_sd_validation_mse": (
            float(sd_value) if pd.notna(sd_value) else None
        ),
        "selected_mean_validation_rmse": float(
            selected_row["mean_validation_rmse"]
        ),
        "confirmation_csv": str(confirmation_path.resolve()),
        "confirmation_summary_csv": str(aggregate_path.resolve()),
    }
    return dict(selected_trial.params), selection


def load_cached_confirmation(
    summary_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Reuse an already completed confirmation on a --skip-search final run."""
    if not args.skip_search or not args.confirm_top_candidates:
        return None
    if not summary_path.exists():
        return None
    previous = read_json(summary_path)
    selected = previous.get("selected_params")
    selection = previous.get("selection")
    if not isinstance(selected, dict) or not isinstance(selection, dict):
        return None
    required = {"h1", "h2", "h3", "h4", "act"}
    if required.difference(selected):
        return None
    if not selection.get("confirmation_performed"):
        return None
    previous_seeds = selection.get("confirmation_seeds")
    if previous_seeds != [int(seed) for seed in args.confirmation_seeds]:
        return None
    if int(selection.get("top_k_confirmed", -1)) != args.confirm_top_k:
        return None
    print(f"Reusing cached multi-seed confirmation from {summary_path}.")
    return dict(selected), dict(selection)


def write_fine_config(
    *,
    broad_dir: Path,
    broad_config: dict[str, Any],
    selected_params: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    config = copy.deepcopy(broad_config)
    model = config["finetune"]["model"]
    for name in ("h1", "h2", "h3", "h4"):
        model[name] = int(selected_params[name])
    model["act"] = str(selected_params["act"])
    model["fthead"] = "FTHead3"
    final_dir = broad_dir / "fine_final_experiment"
    final_dir.mkdir(parents=True, exist_ok=True)
    config["exp_dir"] = str(final_dir.resolve())
    config["finetune"]["n_epochs"] = int(args.final_epochs)
    config["finetune"]["es_patience"] = int(args.final_patience)
    config["finetune"]["chkpoint_name"] = str((final_dir / "ft.pt").resolve())
    config_path = broad_dir / "fragnet_fine_optimised.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path, config


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    true = np.asarray(true, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(true) != len(pred):
        raise ValueError("Prediction and target lengths differ.")
    if len(true) == 0:
        raise ValueError("Cannot score an empty dataset.")
    residual = pred - true
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    r2 = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0
        else math.nan
    )
    pearson = (
        float(np.corrcoef(true, pred)[0, 1])
        if len(true) > 1 and np.std(true) > 0 and np.std(pred) > 0
        else math.nan
    )
    true_rank = pd.Series(true).rank(method="average").to_numpy()
    pred_rank = pd.Series(pred).rank(method="average").to_numpy()
    spearman = (
        float(np.corrcoef(true_rank, pred_rank)[0, 1])
        if len(true) > 1 and np.std(true_rank) > 0 and np.std(pred_rank) > 0
        else math.nan
    )
    return {
        "n": int(len(true)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": finite_or_none(r2),
        "pearson_r": finite_or_none(pearson),
        "spearman_rho": finite_or_none(spearman),
    }


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def evaluate_checkpoint(
    *,
    config: dict[str, Any],
    selected_params: dict[str, Any],
    checkpoint_path: Path,
    validation_dataset: Any,
    test_dataset: Any,
    validation_batch_size: int,
    requested_device: str,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from fragnet.dataset.data import collate_fn
    from fragnet.train.utils import TrainerFineTune as Trainer

    device = choose_device(requested_device, torch)
    model = build_model(config, selected_params)
    model.load_state_dict(torch_load(checkpoint_path, device, torch))
    model.to(device)
    trainer = Trainer(target_type=config["finetune"]["target_type"])
    datasets = {"validation": validation_dataset, "test": test_dataset}
    metrics: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for split, dataset in datasets.items():
            loader = DataLoader(
                dataset,
                collate_fn=collate_fn,
                batch_size=validation_batch_size,
                shuffle=False,
                drop_last=False,
            )
            _, true, pred = trainer.test_regr(
                model=model,
                loader=loader,
                device=device,
            )
            true_array = as_numpy(true)
            pred_array = as_numpy(pred)
            metrics[split] = regression_metrics(true_array, pred_array)
            smiles = [getattr(item, "smiles", "") for item in dataset]
            if len(smiles) != len(true_array):
                smiles = [""] * len(true_array)
            pd.DataFrame(
                {
                    "smiles": smiles,
                    "true": true_array,
                    "prediction": pred_array,
                    "error": pred_array - true_array,
                    "absolute_error": np.abs(pred_array - true_array),
                }
            ).to_csv(output_dir / f"{split}_predictions.csv", index=False)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return metrics


def train_and_evaluate_final(
    *,
    config: dict[str, Any],
    selected_params: dict[str, Any],
    train_dataset: Any,
    validation_dataset: Any,
    test_dataset: Any,
    broad_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    final_dir = broad_dir / "fine_final_experiment"
    checkpoint = final_dir / "ft.pt"
    training = run_one_training(
        config=config,
        head_params=selected_params,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        checkpoint_path=checkpoint,
        seed=args.final_seed,
        max_epochs=args.final_epochs,
        patience=args.final_patience,
        validation_batch_size=args.validation_batch_size,
        requested_device=args.device,
        keep_checkpoint=True,
    )
    metrics = evaluate_checkpoint(
        config=config,
        selected_params=selected_params,
        checkpoint_path=checkpoint,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        validation_batch_size=args.validation_batch_size,
        requested_device=args.device,
        output_dir=final_dir,
    )
    result = {
        "seed": int(args.final_seed),
        "checkpoint": str(checkpoint.resolve()),
        "training": training,
        "metrics": metrics,
        "test_set_evaluated": True,
    }
    write_json(final_dir / "final_metrics.json", result)
    return result


def main() -> int:
    args = parse_args()
    validate_args(args)
    broad_dir = args.broad_dir.expanduser().resolve()
    if not broad_dir.is_dir():
        raise NotADirectoryError(broad_dir)

    config, broad_summary, locked, graph_paths = load_broad_inputs(broad_dir)
    fragnet_root = resolve_fragnet_root(args.fragnet_root, broad_dir)
    configure_fragnet_import(fragnet_root)

    print("Resolved stage-2 inputs:")
    print(f"  broad directory: {broad_dir}")
    print(f"  FragNet root:    {fragnet_root}")
    print(f"  train graphs:    {graph_paths['train']}")
    print(f"  validation:      {graph_paths['val']}")
    print(f"  held-out test:   {graph_paths['test']}")
    print("\nLocked broad hyperparameters:")
    print(f"  learning_rate: {locked['learning_rate']}")
    print(f"  drop_ratio:    {locked['drop_ratio']}")
    print(f"  batch_size:    {locked['batch_size']}")
    print("\nFine search space:")
    print(
        f"  h1, h2, h3, h4: {args.head_min}..{args.head_max} "
        f"in steps of {args.head_step}"
    )
    print(f"  activation: {', '.join(ACTIVATIONS)}")
    print(f"  requested total trials: {args.n_trials}")
    print("  selection data: validation only (test is not loaded)")

    if args.dry_run:
        print("\nDry run complete; no training was started.")
        return 0

    from fragnet.dataset.dataset import load_pickle_dataset

    # Deliberately load only train and validation during all model selection.
    train_dataset = load_pickle_dataset(str(graph_paths["train"]))
    validation_dataset = load_pickle_dataset(str(graph_paths["val"]))
    study, trials_csv = run_fine_study(
        broad_dir=broad_dir,
        config=config,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        args=args,
    )
    ranked = complete_trials_in_rank_order(study)
    single_seed_winner = ranked[0]
    summary_path = broad_dir / "fine_optuna" / "best_fine_hyperparameters.json"
    cached_selection = load_cached_confirmation(summary_path, args)
    if cached_selection is None:
        selected_params, selection = confirm_candidates(
            study=study,
            broad_dir=broad_dir,
            config=config,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            args=args,
        )
    else:
        selected_params, selection = cached_selection
    config_path, final_config = write_fine_config(
        broad_dir=broad_dir,
        broad_config=config,
        selected_params=selected_params,
        args=args,
    )

    fine_summary: dict[str, Any] = {
        "stage": "fine FTHead3 optimisation",
        "method": "seeded Optuna TPE with optional MedianPruner",
        "selection_data": "validation split only",
        "objective": "minimise validation MSE",
        "requested_total_trials": int(args.n_trials),
        "finished_trials": int(
            sum(
                trial.state.name not in {"RUNNING", "WAITING"}
                for trial in study.trials
            )
        ),
        "completed_trials": int(len(ranked)),
        "locked_broad_params": locked,
        "broad_validation_mse": float(broad_summary["best_validation_mse"]),
        "broad_validation_rmse": float(broad_summary["best_validation_rmse"]),
        "fine_search_space": {
            "h1": {
                "low": int(args.head_min),
                "high": int(args.head_max),
                "step": int(args.head_step),
            },
            "h2": {
                "low": int(args.head_min),
                "high": int(args.head_max),
                "step": int(args.head_step),
            },
            "h3": {
                "low": int(args.head_min),
                "high": int(args.head_max),
                "step": int(args.head_step),
            },
            "h4": {
                "low": int(args.head_min),
                "high": int(args.head_max),
                "step": int(args.head_step),
            },
            "act": ACTIVATIONS,
        },
        "single_seed_optuna_winner": {
            "trial_number": int(single_seed_winner.number),
            "validation_mse": float(single_seed_winner.value),
            "validation_rmse": math.sqrt(float(single_seed_winner.value)),
            "params": dict(single_seed_winner.params),
        },
        "selection": selection,
        "selected_params": selected_params,
        "test_set_evaluated": False,
        "study_database": str(
            (broad_dir / "fine_optuna" / "fragnet_fine_optuna.db").resolve()
        ),
        "trials_csv": str(trials_csv.resolve()),
        "fine_optimised_config": str(config_path.resolve()),
    }
    write_json(summary_path, fine_summary)

    print("\nSelected fine hyperparameters:")
    for name in ("h1", "h2", "h3", "h4", "act"):
        print(f"  {name}: {selected_params[name]}")
    print(f"Selection: {selection['method']}")
    print(f"Saved fine configuration: {config_path}")

    if args.train_final:
        print(
            "\nModel selection is complete. Loading the held-out test set "
            "for the final evaluation."
        )
        test_dataset = load_pickle_dataset(str(graph_paths["test"]))
        final_result = train_and_evaluate_final(
            config=final_config,
            selected_params=selected_params,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            broad_dir=broad_dir,
            args=args,
        )
        fine_summary["test_set_evaluated"] = True
        fine_summary["final_training"] = final_result
        write_json(summary_path, fine_summary)
        print("\nFinal metrics:")
        for split in ("validation", "test"):
            values = final_result["metrics"][split]
            print(
                f"  {split}: RMSE={values['rmse']:.6g}, "
                f"MAE={values['mae']:.6g}, R2={values['r2']}, "
                f"Pearson={values['pearson_r']}, "
                f"Spearman={values['spearman_rho']}"
            )
    else:
        print(
            "\nFine search complete. The test set remains untouched. "
            "After inspecting the fine results, run again with "
            "--skip-search --train-final to fit and evaluate the final model."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        raise
