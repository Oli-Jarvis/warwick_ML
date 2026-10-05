#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from chemplot import Plotter
from rdkit import Chem

NAMES = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", type=Path, default=Path("."))
    p.add_argument("--method", choices=NAMES)
    p.add_argument("--run-pattern", default="run_0_seed_*")
    p.add_argument("--input-name", default="full_history.csv")
    p.add_argument("--delimiter", default=";")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--colour-column", default="fitness")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--drop-duplicate-smiles", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("dim_plots"))
    return p.parse_args()


def choose_method(method):
    if method:
        return method
    return {"1": "pca", "2": "tsne", "3": "umap"}[
        input("Choose method: 1=PCA, 2=t-SNE, 3=UMAP: ").strip()
    ]


def load_data(root, pattern, filename, delimiter):
    frames = []
    sep = None if delimiter.lower() == "auto" else delimiter

    for run in sorted(p for p in root.glob(pattern) if p.is_dir()):
        path = run / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="warn")
        df.columns = df.columns.astype(str).str.strip()
        df["source_run"] = run.name
        df["source_row"] = np.arange(len(df))
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No {filename} files found in {root}")

    data = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Combined {len(data):,} rows from {len(frames)} runs.")
    return data


def find_column(data, requested, alternatives=()):
    lookup = {str(c).strip().lower(): c for c in data.columns}
    for name in (requested, *alternatives):
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise KeyError(
        f"Missing column: {requested}. GGS analysis requires a generated SMILES column. Available columns: {', '.join(map(str, data.columns))}"
    )


def prepare_data(data, smiles_col, colour_col, drop_duplicates):
    smiles_source = find_column(
        data, smiles_col, ("SMILES", "canonical_smiles", "canonical smiles", "decoded_smiles")
    )
    colour_source = find_column(data, colour_col)

    data = data.copy()
    if smiles_source != smiles_col:
        data[smiles_col] = data[smiles_source]
    if colour_source != colour_col:
        data[colour_col] = data[colour_source]

    data[colour_col] = pd.to_numeric(data[colour_col], errors="coerce")
    data = data.dropna(subset=[smiles_col, colour_col])
    data["chemplot_smiles"] = data[smiles_col].astype(str).str.strip().str.replace("[Au]", "", regex=False)
    data = data[data["chemplot_smiles"].map(lambda s: bool(s) and Chem.MolFromSmiles(s) is not None)]

    if drop_duplicates:
        data = data.drop_duplicates("chemplot_smiles")

    data = data.reset_index(drop=True)
    if len(data) < 5:
        raise ValueError("At least five valid molecules are required.")

    print(f"Usable molecules: {len(data):,}")
    return data


def embed(data, method, colour_col, random_state):
    plotter = Plotter.from_smiles(
        data["chemplot_smiles"].tolist(),
        target=data[colour_col].tolist(),
        target_type="R",
        sim_type="structural",
    )

    if method == "pca":
        raw = plotter.pca()
    elif method == "tsne":
        raw = plotter.tsne(random_state=random_state)
    else:
        raw = plotter.umap(random_state=random_state)

    if raw is None:
        raw = getattr(plotter, "df_plot_xy", None)
    if raw is None:
        raise RuntimeError("ChemPlot returned no coordinates.")

    xy = pd.DataFrame(raw)
    xy = xy[[c for c in xy.columns if str(c).lower() != "target"][:2]].apply(pd.to_numeric)
    xy.columns = [f"{NAMES[method]}_1", f"{NAMES[method]}_2"]

    if len(xy) != len(data) or xy.isna().any().any():
        raise RuntimeError("Invalid ChemPlot coordinates.")

    return pd.concat([data, xy.reset_index(drop=True)], axis=1)


def save_outputs(data, method, smiles_col, colour_col, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = f"{NAMES[method]}_1", f"{NAMES[method]}_2"
    stem = output_dir / f"full_GGS_{method}_embedding"

    data.to_csv(stem.with_suffix(".csv"), index=False)

    hover = [c for c in [smiles_col, "encoding", "source_run", "source_row", colour_col, "SA", "N_rot", "log_P_upconversion", "P_upconversion"] if c in data]
    fig = px.scatter(data, x=x, y=y, color=colour_col, hover_data=hover, render_mode="webgl", title=f"Complete GGS chemical space: {NAMES[method]}")
    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(template="plotly_white")
    fig.write_html(stem.with_suffix(".html"), include_plotlyjs="cdn")

    plt.figure(figsize=(10, 8))
    points = plt.scatter(data[x], data[y], c=data[colour_col], s=8, alpha=0.75, linewidths=0)
    plt.colorbar(points, label=colour_col)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Complete GGS chemical space: {NAMES[method]}")
    plt.tight_layout()
    plt.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close()

    print(f"Saved: {stem}.csv, .html, .png")


def main():
    args = get_args()
    root = args.root.expanduser().resolve()
    method = choose_method(args.method)
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    data = load_data(root, args.run_pattern, args.input_name, args.delimiter)
    data = prepare_data(data, args.smiles_column, args.colour_column, args.drop_duplicate_smiles)
    data = embed(data, method, args.colour_column, args.random_state)
    save_outputs(data, method, args.smiles_column, args.colour_column, output_dir)


if __name__ == "__main__":
    main()
