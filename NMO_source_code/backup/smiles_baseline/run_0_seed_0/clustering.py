#!/usr/bin/env python3
"""
ChemPlot structural chemical-space analysis for one NMO run/seed.

This script:
1. Reads one full_history.csv (comma or semicolon separated).
2. Uses ChemPlot structural similarity from the SMILES strings.
3. Runs ChemPlot's 2D UMAP.
4. Runs DBSCAN and BIRCH on standardised UMAP coordinates.
5. Produces:
   - fitness scatter plot
   - DBSCAN cluster plot
   - BIRCH cluster plot
   - molecule-density contour plot
   - smoothed fitness contour plot
   - CSV containing coordinates and labels
   - cluster summary CSV

No HTML is produced.

Typical use:
    python chemplot_clustering.py full_history.csv

Optional tuning:
    python chemplot_clustering.py full_history.csv \
        --dbscan-eps 0.35 \
        --dbscan-min-samples 10 \
        --birch-threshold 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chemplot import Plotter
from scipy.stats import gaussian_kde
from sklearn.cluster import Birch, DBSCAN
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ChemPlot structural UMAP, clustering and contour analysis "
            "for one NMO seed."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chemplot_cluster_contour_results"),
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--fitness-column", default="fitness")
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--dbscan-eps", type=float, default=0.35)
    parser.add_argument("--dbscan-min-samples", type=int, default=10)

    parser.add_argument("--birch-threshold", type=float, default=0.25)
    parser.add_argument("--birch-branching-factor", type=int, default=50)
    parser.add_argument(
        "--birch-n-clusters",
        type=int,
        default=0,
        help="Set to 0 to let BIRCH determine the number automatically.",
    )

    parser.add_argument("--contour-neighbours", type=int, default=25)
    parser.add_argument("--grid-size", type=int, default=250)
    parser.add_argument("--mask-spacing-multiplier", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=300)

    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find input file: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        # Automatically detect commas, semicolons, tabs, etc.
        return pd.read_csv(path, sep=None, engine="python")

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError("Input must be a CSV or Parquet file.")


def clean_smiles_for_chemplot(smiles: str) -> str:
    """
    Remove explicit gold atoms before ChemPlot/RDKit processing.

    The original SMILES remains unchanged in the output data.
    """
    return str(smiles).replace("[Au]", "")


def prepare_data(
    df: pd.DataFrame,
    smiles_column: str,
    fitness_column: str,
) -> pd.DataFrame:
    missing = {
        smiles_column,
        fitness_column,
    }.difference(df.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    output = df.copy()

    output[fitness_column] = pd.to_numeric(
        output[fitness_column],
        errors="coerce",
    )

    output = output.dropna(
        subset=[smiles_column, fitness_column]
    ).copy()

    output["chemplot_smiles"] = output[smiles_column].map(
        clean_smiles_for_chemplot
    )

    output = output[
        output["chemplot_smiles"].str.strip().ne("")
    ].reset_index(drop=True)

    if len(output) < 5:
        raise ValueError(
            f"Only {len(output)} usable molecules remain; at least 5 are needed."
        )

    return output


def extract_umap_coordinates(
    plotter: Plotter,
    random_state: int,
) -> pd.DataFrame:
    """
    Run ChemPlot UMAP and normalise the returned coordinate column names.
    """
    coordinates = plotter.umap(random_state=random_state)

    if coordinates is None:
        # Fallback for ChemPlot versions that store rather than return the data.
        coordinates = getattr(plotter, "df_plot_xy", None)

    if coordinates is None:
        raise RuntimeError(
            "ChemPlot did not return or expose its UMAP coordinates."
        )

    coordinates = pd.DataFrame(coordinates).reset_index(drop=True)

    # Identify the two coordinate columns. Ignore a possible target column.
    candidate_columns = [
        column
        for column in coordinates.columns
        if str(column).lower() != "target"
    ]

    if len(candidate_columns) < 2:
        raise RuntimeError(
            "Could not identify two UMAP coordinate columns. "
            f"ChemPlot returned columns: {list(coordinates.columns)}"
        )

    xy = coordinates[candidate_columns[:2]].copy()
    xy.columns = ["UMAP_1", "UMAP_2"]

    xy["UMAP_1"] = pd.to_numeric(xy["UMAP_1"], errors="raise")
    xy["UMAP_2"] = pd.to_numeric(xy["UMAP_2"], errors="raise")

    return xy


def make_grid(
    coordinates: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    x_padding = max((x.max() - x.min()) * 0.03, 1e-6)
    y_padding = max((y.max() - y.min()) * 0.03, 1e-6)

    x_values = np.linspace(
        x.min() - x_padding,
        x.max() + x_padding,
        grid_size,
    )
    y_values = np.linspace(
        y.min() - y_padding,
        y.max() + y_padding,
        grid_size,
    )

    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    return xx, yy, grid_points


def observed_space_mask(
    coordinates: np.ndarray,
    grid_points: np.ndarray,
    grid_shape: tuple[int, int],
    spacing_multiplier: float,
) -> tuple[np.ndarray, float]:
    """
    Mask contour regions that are far from any observed molecule.
    """
    point_model = NearestNeighbors(n_neighbors=2).fit(coordinates)
    point_distances, _ = point_model.kneighbors(coordinates)

    nearest_other = point_distances[:, 1]
    nearest_other = nearest_other[nearest_other > 0]

    if len(nearest_other) == 0:
        threshold = np.inf
    else:
        threshold = (
            spacing_multiplier * float(np.median(nearest_other))
        )

    grid_model = NearestNeighbors(n_neighbors=1).fit(coordinates)
    grid_distances, _ = grid_model.kneighbors(grid_points)

    mask = grid_distances.reshape(grid_shape) <= threshold
    return mask, threshold


def save_fitness_scatter(
    data: pd.DataFrame,
    fitness_column: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        data["UMAP_1"],
        data["UMAP_2"],
        c=data[fitness_column],
        cmap="viridis",
        s=10,
        alpha=0.75,
        edgecolors="none",
    )

    fig.colorbar(scatter, ax=ax, label="Fitness")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Fitness across ChemPlot structural UMAP space")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_cluster_scatter(
    data: pd.DataFrame,
    cluster_column: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = np.sort(data[cluster_column].unique())
    cmap = plt.get_cmap("tab20", max(len(labels), 1))

    for index, label in enumerate(labels):
        selected = data[cluster_column] == label
        display_name = "Noise" if label == -1 else f"Cluster {label}"

        ax.scatter(
            data.loc[selected, "UMAP_1"],
            data.loc[selected, "UMAP_2"],
            s=10,
            alpha=0.75,
            label=display_name,
            color=cmap(index),
            edgecolors="none",
        )

    if len(labels) <= 20:
        ax.legend(
            loc="best",
            fontsize=8,
            markerscale=1.5,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_density_contour(
    coordinates: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    grid_points: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    kde = gaussian_kde(coordinates.T)
    density = kde(grid_points.T).reshape(xx.shape)
    density = np.ma.masked_where(~mask, density)

    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(
        xx,
        yy,
        density,
        levels=30,
        cmap="viridis",
    )

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=3,
        alpha=0.18,
        color="black",
    )

    fig.colorbar(
        contour,
        ax=ax,
        label="Estimated molecule density",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        "Molecule-density contour in ChemPlot structural UMAP space"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_fitness_contour(
    coordinates: np.ndarray,
    fitness: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    grid_points: np.ndarray,
    mask: np.ndarray,
    requested_neighbours: int,
    output_path: Path,
    dpi: int,
) -> int:
    neighbour_count = min(
        max(2, requested_neighbours),
        len(coordinates),
    )

    interpolator = KNeighborsRegressor(
        n_neighbors=neighbour_count,
        weights="distance",
    )
    interpolator.fit(coordinates, fitness)

    predicted = interpolator.predict(grid_points).reshape(xx.shape)
    predicted = np.ma.masked_where(~mask, predicted)

    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(
        xx,
        yy,
        predicted,
        levels=30,
        cmap="viridis",
    )

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=fitness,
        cmap="viridis",
        s=6,
        alpha=0.35,
        edgecolors="none",
    )

    fig.colorbar(
        contour,
        ax=ax,
        label="Estimated local fitness",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        "Smoothed fitness contour in ChemPlot structural UMAP space"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return neighbour_count


def summarise_clusters(
    data: pd.DataFrame,
    cluster_column: str,
    fitness_column: str,
) -> pd.DataFrame:
    summary = (
        data.groupby(cluster_column)[fitness_column]
        .agg(
            molecule_count="size",
            mean_fitness="mean",
            median_fitness="median",
            maximum_fitness="max",
            minimum_fitness="min",
        )
        .reset_index()
        .rename(columns={cluster_column: "cluster"})
    )

    summary.insert(0, "method", cluster_column)
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.input_file}")
    raw = load_table(args.input_file)
    data = prepare_data(
        raw,
        args.smiles_column,
        args.fitness_column,
    )

    print(f"Usable molecules: {len(data):,}")
    print("Constructing ChemPlot structural representation...")

    plotter = Plotter.from_smiles(
        data["chemplot_smiles"].tolist(),
        target=data[args.fitness_column].tolist(),
        target_type="R",
        sim_type="structural",
    )

    print("Running ChemPlot UMAP...")
    coordinates = extract_umap_coordinates(
        plotter,
        args.random_state,
    )

    if len(coordinates) != len(data):
        raise RuntimeError(
            "ChemPlot returned a different number of coordinates "
            f"({len(coordinates)}) from input molecules ({len(data)}). "
            "This usually means one or more SMILES were rejected internally. "
            "Remove invalid SMILES before running the script."
        )

    data = pd.concat(
        [data.reset_index(drop=True), coordinates],
        axis=1,
    )

    embedding = data[["UMAP_1", "UMAP_2"]].to_numpy()
    scaled_embedding = StandardScaler().fit_transform(embedding)

    print("Running DBSCAN...")
    data["DBSCAN_cluster"] = DBSCAN(
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
    ).fit_predict(scaled_embedding)

    print("Running BIRCH...")
    birch_n_clusters = (
        None
        if args.birch_n_clusters <= 0
        else args.birch_n_clusters
    )

    data["BIRCH_cluster"] = Birch(
        threshold=args.birch_threshold,
        branching_factor=args.birch_branching_factor,
        n_clusters=birch_n_clusters,
    ).fit_predict(scaled_embedding)

    print("Saving scatter plots...")
    save_fitness_scatter(
        data,
        args.fitness_column,
        args.output_dir / "chemplot_umap_fitness.png",
        args.dpi,
    )

    save_cluster_scatter(
        data,
        "DBSCAN_cluster",
        "DBSCAN clusters in ChemPlot structural UMAP space",
        args.output_dir / "dbscan_clusters.png",
        args.dpi,
    )

    save_cluster_scatter(
        data,
        "BIRCH_cluster",
        "BIRCH clusters in ChemPlot structural UMAP space",
        args.output_dir / "birch_clusters.png",
        args.dpi,
    )

    print("Building contour grid...")
    xx, yy, grid_points = make_grid(
        embedding,
        args.grid_size,
    )

    mask, mask_threshold = observed_space_mask(
        embedding,
        grid_points,
        xx.shape,
        args.mask_spacing_multiplier,
    )

    print("Saving molecule-density contour...")
    save_density_contour(
        embedding,
        xx,
        yy,
        grid_points,
        mask,
        args.output_dir / "molecule_density_contour.png",
        args.dpi,
    )

    print("Saving fitness contour...")
    effective_neighbours = save_fitness_contour(
        embedding,
        data[args.fitness_column].to_numpy(dtype=float),
        xx,
        yy,
        grid_points,
        mask,
        args.contour_neighbours,
        args.output_dir / "fitness_contour.png",
        args.dpi,
    )

    output_csv = args.output_dir / "chemplot_umap_results.csv"
    data.to_csv(output_csv, index=False)

    summaries = pd.concat(
        [
            summarise_clusters(
                data,
                "DBSCAN_cluster",
                args.fitness_column,
            ),
            summarise_clusters(
                data,
                "BIRCH_cluster",
                args.fitness_column,
            ),
        ],
        ignore_index=True,
    )
    summaries.to_csv(
        args.output_dir / "cluster_summary.csv",
        index=False,
    )

    dbscan_labels = data["DBSCAN_cluster"].to_numpy()
    dbscan_clusters = len(set(dbscan_labels)) - (
        1 if -1 in dbscan_labels else 0
    )
    dbscan_noise = int(np.sum(dbscan_labels == -1))
    birch_clusters = int(data["BIRCH_cluster"].nunique())

    settings = {
        "input_file": str(args.input_file),
        "representation": "ChemPlot structural similarity",
        "valid_molecules": int(len(data)),
        "random_state": args.random_state,
        "dbscan_eps_on_standardised_umap": args.dbscan_eps,
        "dbscan_min_samples": args.dbscan_min_samples,
        "dbscan_cluster_count_excluding_noise": dbscan_clusters,
        "dbscan_noise_count": dbscan_noise,
        "birch_threshold_on_standardised_umap": args.birch_threshold,
        "birch_cluster_count": birch_clusters,
        "fitness_contour_neighbours": effective_neighbours,
        "contour_mask_distance_threshold": mask_threshold,
    }

    with open(
        args.output_dir / "run_settings.txt",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(settings, handle, indent=2)
        handle.write("\n")

    print("\nFinished.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(
        f"DBSCAN: {dbscan_clusters} clusters and "
        f"{dbscan_noise} noise molecules"
    )
    print(f"BIRCH: {birch_clusters} clusters")
    print(f"Coordinate data: {output_csv}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
