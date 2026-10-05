#!/usr/bin/env python3
"""
Analyse saved dimensionality-reduction coordinates for the SMILES runs.

Compatible inputs:
    dim_plots/full_fixed_pca_embedding.csv
    dim_plots/full_fixed_tsne_embedding.csv
    dim_plots/full_fixed_umap_embedding.csv

Behaviour:
    - PCA:
        * plots Bonding_PC_1 against Structural_PC_1, with the selected
          property on the z axis in the interactive plots
        * clusters using Bonding_PC_1-3, Structural_PC_1-3, and the
          selected property
        * scales each descriptor block by the standard deviation of its
          first PC, and scales the property to a comparable contribution
        * uses BIRCH subclusters followed by agglomerative clustering
    - t-SNE and UMAP:
        * plots and clusters using their two saved coordinates

    By default all usable molecules are clustered, matching the paper.
    Property filtering remains available as an optional pre-clustering step.

Outputs:
    analysis_plots/<method>/<property>_all/
    or, when optional filtering is used:
    analysis_plots/<method>/<property>_<threshold_type>_<threshold>/
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.preprocessing import StandardScaler


EMBEDDING_FILES = {
    "1": ("pca", Path("dim_plots/full_fixed_pca_embedding.csv")),
    "2": ("tsne", Path("dim_plots/full_fixed_tsne_embedding.csv")),
    "3": ("umap", Path("dim_plots/full_fixed_umap_embedding.csv")),
}


def choose_embedding() -> tuple[str, Path]:
    print("\nChoose dimensionality reduction:")
    print("1 = PCA")
    print("2 = t-SNE")
    print("3 = UMAP")

    choice = input("Choice: ").strip()

    if choice not in EMBEDDING_FILES:
        raise SystemExit("Choice must be 1, 2 or 3.")

    method, path = EMBEDDING_FILES[choice]

    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    return method, path


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise KeyError(
        f"Could not find any of {candidates}. Available columns:\n"
        + ", ".join(map(str, df.columns))
    )


def find_embedding_columns(
    df: pd.DataFrame,
    method: str,
) -> tuple[str, str]:
    normalised = {
        re.sub(r"[^a-z0-9]+", "", str(column).lower()): column
        for column in df.columns
    }

    if method == "pca":
        bonding_pc1 = normalised.get("bondingpc1")
        structural_pc1 = normalised.get("structuralpc1")

        if bonding_pc1 is None or structural_pc1 is None:
            raise KeyError(
                "The PCA CSV must contain Bonding_PC_1 and Structural_PC_1. "
                "Regenerate it using the paper-style dimensionality-reduction script."
            )

        return bonding_pc1, structural_pc1

    aliases = {
        "tsne": ["tsne"],
        "umap": ["umap"],
    }[method]

    candidate_pairs = []

    for prefix in aliases:
        candidate_pairs.extend([
            (f"{prefix}1", f"{prefix}2"),
            (f"{prefix}01", f"{prefix}02"),
            (f"{prefix}x", f"{prefix}y"),
        ])

    for first, second in candidate_pairs:
        if first in normalised and second in normalised:
            return normalised[first], normalised[second]

    raise KeyError(
        f"Could not identify the first two {method.upper()} coordinate columns."
    )


def find_pca_clustering_columns(
    df: pd.DataFrame,
    n_components: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Return the requested number of bonding and structural PCA columns.
    """
    normalised = {
        re.sub(r"[^a-z0-9]+", "", str(column).lower()): column
        for column in df.columns
    }

    requested_bonding = [
        f"bondingpc{i}" for i in range(1, n_components + 1)
    ]
    requested_structural = [
        f"structuralpc{i}" for i in range(1, n_components + 1)
    ]
    requested = [*requested_bonding, *requested_structural]

    missing = [name for name in requested if name not in normalised]

    if missing:
        raise KeyError(
            f"The PCA CSV must contain Bonding_PC_1-{n_components} and "
            f"Structural_PC_1-{n_components}. Missing: " + ", ".join(missing)
        )

    return (
        [normalised[name] for name in requested_bonding],
        [normalised[name] for name in requested_structural],
    )


def make_paper_clustering_matrix(
    df: pd.DataFrame,
    bonding_columns: list[str],
    structural_columns: list[str],
    property_col: str,
    property_weight: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build the paper-style PCA/property clustering matrix.

    The first three PCs from each descriptor retain their PCA variance
    hierarchy. Each descriptor block is centred and divided by the standard
    deviation of its first PC. The property is z-scored and can then be given
    an additional user-selected weight. With the default weight of 1.0,
    Bonding PC1, Structural PC1, and the property each have unit standard
    deviation and comparable influence on Euclidean distances, without
    artificially inflating PC2 and PC3.
    """
    if property_weight <= 0:
        raise ValueError("The property weight must be greater than zero.")

    bonding = df[bonding_columns].to_numpy(dtype=float)
    structural = df[structural_columns].to_numpy(dtype=float)
    property_values = df[property_col].to_numpy(dtype=float)

    bonding_pc1_sd = float(np.std(bonding[:, 0]))
    structural_pc1_sd = float(np.std(structural[:, 0]))
    property_sd = float(np.std(property_values))

    scales = {
        bonding_columns[0]: bonding_pc1_sd,
        structural_columns[0]: structural_pc1_sd,
        property_col: property_sd,
    }
    invalid = [
        name
        for name, scale in scales.items()
        if not np.isfinite(scale) or scale <= 0
    ]
    if invalid:
        raise ValueError(
            "Cannot scale constant or non-finite clustering references: "
            + ", ".join(map(str, invalid))
        )

    bonding_scaled = (
        bonding - np.mean(bonding, axis=0, keepdims=True)
    ) / bonding_pc1_sd
    structural_scaled = (
        structural - np.mean(structural, axis=0, keepdims=True)
    ) / structural_pc1_sd
    property_scaled = (
        (property_values - np.mean(property_values)) / property_sd
    )[:, None] * property_weight

    matrix = np.column_stack([
        bonding_scaled,
        structural_scaled,
        property_scaled,
    ])

    scaling_rows = []
    for column_index, column in enumerate(bonding_columns):
        scaling_rows.append({
            "feature": column,
            "group": "bonding",
            "centering_mean": float(df[column].mean()),
            "group_reference": bonding_columns[0],
            "divisor": bonding_pc1_sd,
            "additional_weight": 1.0,
            "scaled_standard_deviation": float(
                np.std(matrix[:, column_index])
            ),
        })
    for column_index, column in enumerate(structural_columns):
        matrix_index = len(bonding_columns) + column_index
        scaling_rows.append({
            "feature": column,
            "group": "structural",
            "centering_mean": float(df[column].mean()),
            "group_reference": structural_columns[0],
            "divisor": structural_pc1_sd,
            "additional_weight": 1.0,
            "scaled_standard_deviation": float(np.std(matrix[:, matrix_index])),
        })
    scaling_rows.append({
        "feature": property_col,
        "group": "property",
        "centering_mean": float(df[property_col].mean()),
        "group_reference": property_col,
        "divisor": property_sd,
        "additional_weight": property_weight,
        "scaled_standard_deviation": float(np.std(matrix[:, -1])),
    })

    return matrix, pd.DataFrame(scaling_rows)


def safe_name(value: float) -> str:
    return (
        f"{value:.10g}"
        .replace("-", "minus_")
        .replace(".", "p")
        .replace("+", "plus_")
    )


def molecule_png_data_uri(
    smiles: str,
    size: tuple[int, int] = (260, 180),
) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None

    if mol is None:
        return ""

    image = Draw.MolToImage(mol, size=size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None

    if mol is None:
        return ""

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""

    return Chem.MolToSmiles(scaffold, canonical=True)


def save_density_map(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    property_col: str,
    threshold_type: str | None,
    threshold: float | None,
    output: Path,
) -> None:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    if len(df) < 3:
        raise ValueError("At least three molecules are required for the density map.")

    x_padding = max(np.ptp(x) * 0.05, 1e-9)
    y_padding = max(np.ptp(y) * 0.05, 1e-9)

    x_grid = np.linspace(
        np.min(x) - x_padding,
        np.max(x) + x_padding,
        250,
    )
    y_grid = np.linspace(
        np.min(y) - y_padding,
        np.max(y) + y_padding,
        250,
    )

    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    samples = np.vstack([x, y])

    try:
        density = gaussian_kde(samples)(positions).reshape(xx.shape)
    except np.linalg.LinAlgError:
        scale = np.maximum(np.std(samples, axis=1, keepdims=True), 1.0)
        jitter = np.random.default_rng(0).normal(
            scale=1e-9 * scale,
            size=samples.shape,
        )
        density = gaussian_kde(samples + jitter)(positions).reshape(xx.shape)

    if threshold_type == "min" and threshold is not None:
        condition = f"{property_col} ≥ {threshold:.6g}"
    elif threshold_type == "max" and threshold is not None:
        condition = f"{property_col} ≤ {threshold:.6g}"
    else:
        condition = f"all usable {property_col} values"

    plt.figure(figsize=(10, 8))
    filled = plt.contourf(
        xx,
        yy,
        density,
        levels=30,
        cmap="viridis",
    )
    plt.contour(
        xx,
        yy,
        density,
        levels=10,
        linewidths=0.5,
    )
    plt.scatter(
        x,
        y,
        s=3,
        alpha=0.18,
    )
    plt.colorbar(filled, label="KDE density")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(
        f"Density of retained molecules\n"
        f"{condition} ({len(df):,} molecules)"
    )
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def save_3d_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    property_col: str,
    smiles_col: str,
    cluster_col: str,
    title: str,
    output: Path,
) -> None:
    plot_df = df.copy()
    plot_df[cluster_col] = plot_df[cluster_col].astype(str)

    hover = {
        smiles_col: True,
        property_col: ":.5g",
        cluster_col: True,
        x_col: ":.5g",
        y_col: ":.5g",
    }

    for extra in [
        "SA",
        "sa",
        "N_rot",
        "n_rot",
        "log_P_upconversion",
        "P_upconversion",
    ]:
        if extra in plot_df.columns and extra not in hover:
            hover[extra] = True

    figure = px.scatter_3d(
        plot_df,
        x=x_col,
        y=y_col,
        z=property_col,
        color=cluster_col,
        hover_data=hover,
        title=title,
        opacity=0.75,
    )
    figure.update_traces(marker={"size": 3})
    figure.update_layout(legend_title_text="Cluster")
    figure.write_html(output, include_plotlyjs="cdn")


def save_scaffold_report(
    df: pd.DataFrame,
    smiles_col: str,
    property_col: str,
    cluster_col: str,
    output: Path,
) -> None:
    work = df[[smiles_col, property_col, cluster_col]].copy()
    work["scaffold"] = work[smiles_col].map(scaffold_smiles)

    rows = []

    for cluster, group in work.groupby(cluster_col, sort=True):
        scaffold_counts = group["scaffold"].value_counts(dropna=False)

        if scaffold_counts.empty:
            dominant = ""
            count = 0
        else:
            dominant = scaffold_counts.index[0]
            count = int(scaffold_counts.iloc[0])

        percentage = 100.0 * count / len(group)
        representative = dominant or str(group.iloc[0][smiles_col])

        rows.append({
            "drawing": molecule_png_data_uri(representative),
            "cluster": cluster,
            "dominant_scaffold": (
                dominant
                if dominant
                else "(acyclic / no Murcko scaffold)"
            ),
            "percentage": percentage,
            "molecules": len(group),
            "mean": group[property_col].mean(),
            "sd": group[property_col].std(ddof=1),
        })

    summary = pd.DataFrame(rows).sort_values(
        "mean",
        ascending=False,
    )

    html_rows = []

    for _, row in summary.iterrows():
        image_html = (
            f'<img src="{row["drawing"]}" width="260" height="180">'
            if row["drawing"]
            else "Drawing unavailable"
        )

        sd_text = (
            f"{row['sd']:.6g}"
            if pd.notna(row["sd"])
            else "N/A"
        )

        html_rows.append(
            "<tr>"
            f"<td>{image_html}</td>"
            f"<td>{row['cluster']}</td>"
            f"<td><code>{row['dominant_scaffold']}</code></td>"
            f"<td>{row['percentage']:.2f}</td>"
            f"<td>{int(row['molecules'])}</td>"
            f"<td>{row['mean']:.6g}</td>"
            f"<td>{sd_text}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{cluster_col} scaffold summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{
    border: 1px solid #ccc;
    padding: 8px;
    text-align: center;
}}
th {{
    background: #f2f2f2;
    position: sticky;
    top: 0;
}}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>{cluster_col} scaffold summary</h1>
<table>
<thead>
<tr>
<th>Drawing</th>
<th>Cluster</th>
<th>Dominant scaffold</th>
<th>% of cluster</th>
<th>Molecules</th>
<th>Mean {property_col}</th>
<th>SD {property_col}</th>
</tr>
</thead>
<tbody>
{''.join(html_rows)}
</tbody>
</table>
</body>
</html>
"""

    output.write_text(html, encoding="utf-8")


def save_representative_molecules_report(
    df: pd.DataFrame,
    smiles_col: str,
    property_col: str,
    cluster_col: str,
    output: Path,
    molecules_per_cluster: int = 20,
) -> None:
    try:
        sa_col = find_column(df, ["SA", "sa"])
    except KeyError:
        sa_col = None

    sections = []

    for cluster, group in df.groupby(cluster_col, sort=True):
        unique_group = group.drop_duplicates(subset=[smiles_col])

        selected = unique_group.sort_values(
            "distance_to_cluster_centroid",
            ascending=True,
        ).head(molecules_per_cluster)

        cards = []

        for _, row in selected.iterrows():
            smiles = str(row[smiles_col])
            image_uri = molecule_png_data_uri(
                smiles,
                size=(300, 210),
            )

            image_html = (
                f'<img src="{image_uri}" width="300" height="210">'
                if image_uri
                else '<div class="missing">Drawing unavailable</div>'
            )

            sa_html = (
                f'<p><strong>SA:</strong> {row[sa_col]:.6g}</p>'
                if sa_col is not None and pd.notna(row[sa_col])
                else ""
            )

            cards.append(
                '<div class="card">'
                f'{image_html}'
                f'{sa_html}'
                f'<p><strong>{property_col}:</strong> '
                f'{row[property_col]:.6g}</p>'
                f'<p><strong>Centroid distance:</strong> '
                f'{row["distance_to_cluster_centroid"]:.6g}</p>'
                '</div>'
            )

        sections.append(
            '<section>'
            f'<h2>Cluster {cluster} — {len(group):,} molecules</h2>'
            f'<div class="gallery">{"".join(cards)}</div>'
            '</section>'
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{cluster_col} representative molecules</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 24px;
    background: #fafafa;
}}
section {{
    margin-bottom: 36px;
    padding: 18px;
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
}}
.gallery {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
}}
.card {{
    width: 320px;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 6px;
    background: white;
}}
.card img {{
    display: block;
    margin: auto;
}}
.missing {{
    width: 300px;
    height: 210px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #eee;
}}
</style>
</head>
<body>
<h1>{cluster_col} representative molecules</h1>
<p>
The {molecules_per_cluster} unique molecules closest to the centroid in the
weighted clustering space are shown for each cluster, where available.
</p>
{''.join(sections)}
</body>
</html>
"""

    output.write_text(html, encoding="utf-8")


def main() -> None:
    method, input_path = choose_embedding()
    data = pd.read_csv(input_path)

    smiles_col = find_column(
        data,
        [
            "smiles",
            "SMILES",
            "canonical_smiles",
            "canonical smiles",
            "encoding",
        ],
    )

    x_col, y_col = find_embedding_columns(data, method)

    if method == "pca":
        pca_clustering_components = int(
            input("Number of PCA components per descriptor for clustering [3]: ").strip()
            or "3"
        )
        if pca_clustering_components < 1:
            raise SystemExit("The number of PCA clustering components must be at least 1.")
        bonding_columns, structural_columns = find_pca_clustering_columns(
            data,
            pca_clustering_components,
        )
        clustering_columns = [*bonding_columns, *structural_columns]
    else:
        bonding_columns = []
        structural_columns = []
        clustering_columns = [x_col, y_col]

    print("\nAvailable numeric columns:")
    numeric_columns = list(data.select_dtypes(include=np.number).columns)
    print(", ".join(map(str, numeric_columns)))

    requested_property = (
        input("\nProperty to analyse [fitness]: ").strip()
        or "fitness"
    )
    property_col = find_column(data, [requested_property])

    if method == "pca" and property_col not in clustering_columns:
        clustering_columns = [*clustering_columns, property_col]

    filter_choice = input(
        "\nFilter molecules by the property before clustering? [y/N]: "
    ).strip().lower()

    if filter_choice in {"", "n", "no"}:
        threshold_type = None
        threshold = None
        comparison = None
    elif filter_choice in {"y", "yes"}:
        print("\nChoose threshold type:")
        print("1 = Minimum value — retain molecules at or above the threshold")
        print("2 = Maximum value — retain molecules at or below the threshold")

        threshold_choice = input("Choice [1]: ").strip() or "1"

        if threshold_choice == "1":
            threshold_type = "min"
            comparison = ">="
        elif threshold_choice == "2":
            threshold_type = "max"
            comparison = "<="
        else:
            raise SystemExit("Threshold choice must be 1 or 2.")

        threshold_text = input(
            f"Enter the {threshold_type} {property_col} threshold: "
        ).strip()

        if not threshold_text:
            raise SystemExit("A threshold value is required.")

        threshold = float(threshold_text)
    else:
        raise SystemExit("Filter choice must be y or n.")

    required_columns = list(dict.fromkeys([
        smiles_col,
        x_col,
        y_col,
        property_col,
        *clustering_columns,
    ]))

    work = data.dropna(subset=required_columns).copy()

    if threshold_type == "min" and threshold is not None:
        work = work[work[property_col] >= threshold].copy()
    elif threshold_type == "max" and threshold is not None:
        work = work[work[property_col] <= threshold].copy()

    if len(work) < 3:
        raise SystemExit(
            "Fewer than three usable molecules remain after filtering."
        )

    print(f"\nLoaded rows: {len(data):,}")
    if threshold_type is None:
        print(f"Retained rows: {len(work):,} (no property filter)")
    else:
        print(
            f"Retained rows: {len(work):,} "
            f"({property_col} {comparison} {threshold:.6g})"
        )
    print(f"Plotting coordinates: {x_col}, {y_col}")
    print(f"Clustering dimensions: {len(clustering_columns)}")

    if method == "pca":
        print(
            "PCA clustering columns: "
            + ", ".join(map(str, clustering_columns))
        )

    filter_stem = (
        f"{property_col}_all"
        if threshold_type is None
        else f"{property_col}_{threshold_type}_{safe_name(threshold)}"
    )
    output_dir = Path("analysis_plots") / method / filter_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "pca":
        property_weight = float(
            input(
                "\nProperty weight relative to Bonding PC1 and "
                "Structural PC1 [1.0]: "
            ).strip()
            or "1.0"
        )
        try:
            scaled_matrix, scaling_table = make_paper_clustering_matrix(
                work,
                bonding_columns,
                structural_columns,
                property_col,
                property_weight,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        # UMAP and t-SNE coordinate scales are arbitrary. Standardising
        # their two coordinates keeps the BIRCH threshold
        # meaningful and prevents one axis from dominating distances.
        clustering_matrix = work[clustering_columns].to_numpy(dtype=float)
        scaled_matrix = StandardScaler().fit_transform(clustering_matrix)
        scaling_table = pd.DataFrame({
            "feature": clustering_columns,
            "group": method,
            "centering_mean": np.mean(clustering_matrix, axis=0),
            "group_reference": clustering_columns,
            "divisor": np.std(clustering_matrix, axis=0),
            "additional_weight": 1.0,
            "scaled_standard_deviation": np.std(scaled_matrix, axis=0),
        })

    birch_threshold = float(
        input("\nBIRCH threshold [0.50]: ").strip()
        or "0.50"
    )

    final_clusters = int(
        input("Final number of agglomerative clusters [20]: ").strip()
        or "20"
    )

    if final_clusters < 2:
        raise SystemExit("The final number of clusters must be at least 2.")

    birch_model = Birch(
        threshold=birch_threshold,
        n_clusters=None,
    )
    birch_subcluster_labels = birch_model.fit_predict(scaled_matrix)
    subcluster_centres = birch_model.subcluster_centers_

    if len(subcluster_centres) < final_clusters:
        raise SystemExit(
            f"BIRCH produced only {len(subcluster_centres)} subclusters, "
            f"which is fewer than the requested {final_clusters} final clusters. "
            "Lower the BIRCH threshold or request fewer final clusters."
        )

    agglomerative_labels = AgglomerativeClustering(
        n_clusters=final_clusters,
        linkage="ward",
    ).fit_predict(subcluster_centres)

    work["BIRCH_Agglomerative_cluster"] = agglomerative_labels[
        birch_subcluster_labels
    ]

    cluster_labels = work["BIRCH_Agglomerative_cluster"].to_numpy(dtype=int)
    centroid_distances = np.empty(len(work), dtype=float)
    for cluster in np.unique(cluster_labels):
        member_mask = cluster_labels == cluster
        final_centroid = np.mean(scaled_matrix[member_mask], axis=0)
        centroid_distances[member_mask] = np.linalg.norm(
            scaled_matrix[member_mask] - final_centroid,
            axis=1,
        )
    work["distance_to_cluster_centroid"] = centroid_distances

    print(f"BIRCH subclusters: {len(subcluster_centres)}")
    print(f"Final agglomerative clusters: {final_clusters}")

    cluster_col = "BIRCH_Agglomerative_cluster"
    work.to_csv(
        output_dir / "birch_agglomerative_clustered_molecules.csv",
        index=False,
    )
    scaling_table.to_csv(
        output_dir / "clustering_feature_scaling.csv",
        index=False,
    )
    (
        work.groupby(cluster_col)[property_col]
        .agg(["size", "mean", "std", "min", "max"])
        .reset_index()
        .to_csv(
            output_dir / "birch_agglomerative_cluster_summary.csv",
            index=False,
        )
    )

    save_density_map(
        work,
        x_col,
        y_col,
        property_col,
        threshold_type,
        threshold,
        output_dir
        / f"{method}_{property_col}_{threshold_type or 'all'}_kde_density.png",
    )

    save_3d_plot(
        work,
        x_col,
        y_col,
        property_col,
        smiles_col,
        cluster_col,
        f"{method.upper()} embedding — BIRCH + agglomerative clusters",
        output_dir / "birch_agglomerative_3d.html",
    )

    save_scaffold_report(
        work,
        smiles_col,
        property_col,
        cluster_col,
        output_dir / "birch_agglomerative_scaffolds.html",
    )

    save_representative_molecules_report(
        work,
        smiles_col,
        property_col,
        cluster_col,
        output_dir / "birch_agglomerative_representative_molecules.html",
        molecules_per_cluster=20,
    )

    print("\nSaved:")
    for output in sorted(output_dir.iterdir()):
        print(f"  {output}")


