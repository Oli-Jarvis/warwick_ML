#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from fragnet.dataset.data import collate_fn
from fragnet.dataset.dataset import load_pickle_dataset
from fragnet.model.gat.gat2 import FragNetFineTune
from fragnet.train.utils import TrainerFineTune as Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("fragnet_log_P_upconversion_regression"),
    )
    return p.parse_args()


def calc_metrics(true, pred):
    return {
        "n": int(len(true)),
        "rmse": float(mean_squared_error(true, pred) ** 0.5),
        "mae": float(mean_absolute_error(true, pred)),
        "r2": float(r2_score(true, pred)),
        "pearson_r": float(np.corrcoef(true, pred)[0, 1]),
        "spearman_rho": float(pd.Series(true).corr(pd.Series(pred), method="spearman")),
    }


def save_outputs(name, true, pred, smiles, out_dir):
    true = np.asarray(true, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    residual = pred - true

    pd.DataFrame({
        "smiles": smiles,
        "true": true,
        "predicted": pred,
        "residual": residual,
        "absolute_error": np.abs(residual),
    }).to_csv(out_dir / f"{name}_predictions.csv", index=False)

    m = calc_metrics(true, pred)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(true, pred, alpha=0.55, s=18)
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Actual log_P_upconversion")
    ax.set_ylabel("Predicted log_P_upconversion")
    ax.set_title(
        f"FragNet {name}: predicted vs actual\n"
        f"RMSE={m['rmse']:.4g}, MAE={m['mae']:.4g}, R2={m['r2']:.4g}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_predicted_vs_actual.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(pred, residual, alpha=0.55, s=18)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted log_P_upconversion")
    ax.set_ylabel("Residual (predicted - actual)")
    ax.set_title(f"FragNet {name}: residuals")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_residuals.png", dpi=300)
    plt.close(fig)

    return m


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    cfg = OmegaConf.load(results_dir / "fragnet_regression.yaml")
    OmegaConf.resolve(cfg)

    out_dir = results_dir / "preliminary_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = results_dir / "experiment" / "ft.pt"
    snapshot = out_dir / "ft_snapshot.pt"
    shutil.copy2(checkpoint, snapshot)

    device = torch.device("cpu")

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
    model.load_state_dict(torch.load(snapshot, map_location=device))
    model.to(device)

    trainer = Trainer(target_type=cfg.finetune.target_type)
    all_metrics = {}

    for name, dataset_path in (
        ("validation", Path(cfg.finetune.val.path)),
        ("test", Path(cfg.finetune.test.path)),
    ):
        dataset = load_pickle_dataset(dataset_path)
        loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=64,
            shuffle=False,
            drop_last=False,
        )
        score, true, pred = trainer.test(model=model, loader=loader, device=device)
        smiles = [item.smiles for item in loader.dataset]
        all_metrics[name] = save_outputs(name, true, pred, smiles, out_dir)
        all_metrics[name]["fragnet_reported_rmse"] = float(score ** 0.5)

    (out_dir / "regression_metrics.json").write_text(
        json.dumps(all_metrics, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(all_metrics, indent=2))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
