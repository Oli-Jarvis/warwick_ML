#!/usr/bin/env python3
"""
Analyse saved ChemPlot dimensionality-reduction coordinates for the SMILES runs.

Run from:
    ~/plots_warwick/backup/smiles_baseline2

Usage:
    python analyse_dim.py

Inputs:
    dim_plots/full_fixed_pca_embedding.csv
    dim_plots/full_fixed_tsne_embedding.csv
    dim_plots/full_fixed_umap_embedding.csv

Outputs are written beneath:
    analysis_plots/<method>/<property>_threshold_<value>/
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
from sklearn.cluster import Birch, DBSCAN
from sklearn.preprocessing import StandardScaler


EMBEDDING_FILES = {
    "1": ("pca", Path("dim_plots/combined_GGS_SMILES_pca.csv")),
    "2": ("tsne", Path("dim_plots/combined_GGS_SMILES_tsne.csv")),
    "3": ("umap", Path("dim_plots/combined_GGS_SMILES_umap.csv")),
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


def find_embedding_columns(df: pd.DataFrame, method: str) -> tuple[str, str]:
    normalised = {
        re.sub(r"[^a-z0-9]+", "", str(column).lower()): column
        for column in df.columns
    }

    method_aliases = {
        "pca": ["pca"],
        "tsne": ["tsne", "t_sne"],
        "umap": ["umap"],
    }[method]

    pairs = []
    for prefix in method_aliases:
        prefix = re.sub(r"[^a-z0-9]+", "", prefix)
        pairs.extend([
            (f"{prefix}1", f"{prefix}2"),
            (f"{prefix}x", f"{prefix}y"),
            (f"{prefix}01", f"{prefix}02"),
        ])

    pairs.extend([
        ("dim1", "dim2"),
        ("component1", "component2"),
        ("x", "y"),
    ])

    for a, b in pairs:
        if a in normalised and b in normalised:
            return normalised[a], normalised[b]

    numeric = [
        c for c in df.select_dtypes(include=np.number).columns
        if str(c).lower() not in {"fitness", "sa", "n_rot", "logp"}
    ]
    if len(numeric) >= 2:
        print(
            f"Warning: embedding columns were not named conventionally. "
            f"Using {numeric[-2]!r} and {numeric[-1]!r}."
        )
        return numeric[-2], numeric[-1]

    raise KeyError("Could not identify the two embedding-coordinate columns.")


def safe_name(value: float) -> str:
    return str(value).replace("-", "minus_").replace(".", "p")


def molecule_png_data_uri(smiles: str, size: tuple[int, int] = (260, 180)) -> str:
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
    threshold_type: str,
    threshold: float,
    output: Path,
) -> None:
    """Save a conventional 2D KDE density plot of the retained molecules."""
    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)

    if len(df) < 3:
        raise ValueError("At least three molecules are required for a KDE plot.")

    x_padding = max(np.ptp(x) * 0.05, 1e-9)
    y_padding = max(np.ptp(y) * 0.05, 1e-9)

    x_grid = np.linspace(np.min(x) - x_padding, np.max(x) + x_padding, 250)
    y_grid = np.linspace(np.min(y) - y_padding, np.max(y) + y_padding, 250)
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

    performance = "high-performing" if threshold_type == "min" else "low-performing"
    condition = (
        f"{property_col} ≥ {threshold:.6g}"
        if threshold_type == "min"
        else f"{property_col} ≤ {threshold:.6g}"
    )

    plt.figure(figsize=(10, 8))
    filled = plt.contourf(xx, yy, density, levels=30, cmap="viridis")
    plt.contour(xx, yy, density, levels=10, linewidths=0.5)
    plt.scatter(x, y, s=3, alpha=0.18)
    plt.colorbar(filled, label="KDE density")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(
        f"Density of {performance} regions\n"
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

    for extra in ["SA", "sa", "N_rot", "n_rot", "log_P_upconversion"]:
        if extra in plot_df.columns and extra not in hover:
            hover[extra] = True

    fig = px.scatter_3d(
        plot_df,
        x=x_col,
        y=y_col,
        z=property_col,
        color=cluster_col,
        hover_data=hover,
        title=title,
        opacity=0.75,
    )
    fig.update_traces(marker={"size": 3})
    fig.update_layout(legend_title_text="Cluster")
    fig.write_html(output, include_plotlyjs="cdn")


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
        representative = dominant
        if not representative:
            representative = str(group.iloc[0][smiles_col])

        rows.append({
            "Drawing": molecule_png_data_uri(representative),
            "Cluster": cluster,
            "Dominant scaffold": dominant if dominant else "(acyclic / no Murcko scaffold)",
            "% of cluster": percentage,
            "Molecules": len(group),
            f"Mean {property_col}": group[property_col].mean(),
            f"SD {property_col}": group[property_col].std(ddof=1),
        })

    summary = pd.DataFrame(rows)

    html_rows = []
    for _, row in summary.iterrows():
        image_html = (
            f'<img src="{row["Drawing"]}" width="260" height="180">'
            if row["Drawing"] else "Drawing unavailable"
        )
        html_rows.append(
            "<tr>"
            f"<td>{image_html}</td>"
            f"<td>{row['Cluster']}</td>"
            f"<td><code>{row['Dominant scaffold']}</code></td>"
            f"<td>{row['% of cluster']:.2f}</td>"
            f"<td>{int(row['Molecules'])}</td>"
            f"<td>{row[f'Mean {property_col}']:.6g}</td>"
            f"<td>{row[f'SD {property_col}']:.6g}</td>"
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
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
th {{ background: #f2f2f2; position: sticky; top: 0; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>{cluster_col} scaffold summary</h1>
<p>Dominant scaffold means the most frequent Bemis–Murcko scaffold in that cluster.</p>
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
    molecules_per_cluster: int = 4,
) -> None:
    """Save a gallery of randomly selected molecules from every cluster."""
    sections = []

    for cluster, group in df.groupby(cluster_col, sort=True):
        unique_group = group.drop_duplicates(subset=[smiles_col])

        if "SA" not in unique_group.columns:
            raise ValueError("Representative molecule report requires an 'SA' column.")

        sampled = (
            unique_group
            .sort_values("SA", ascending=True)
            .head(molecules_per_cluster)
        )

        cards = []
        for _, row in sampled.iterrows():
            smiles = str(row[smiles_col])
            image_uri = molecule_png_data_uri(smiles, size=(300, 210))
            image_html = (
                f'<img src="{image_uri}" width="300" height="210">'
                if image_uri
                else '<div class="missing">Drawing unavailable</div>'
            )

            cards.append(
                '<div class="card">'
                f'{image_html}'
                f'<p><strong>SA:</strong> {row["SA"]:.6g}</p>'
                f'<p><strong>{property_col}:</strong> {row[property_col]:.6g}</p>'
                '</div>'
            )

        note = ""
        if len(group) < molecules_per_cluster:
            note = (
                f"<p><em>This cluster contains only {len(group)} molecule(s), "
                f"so all available molecules are shown.</em></p>"
            )

        sections.append(
            f'<section>'
            f'<h2>Cluster {cluster} — {len(group):,} molecules</h2>'
            f'{note}'
            f'<div class="gallery">{"".join(cards)}</div>'
            f'</section>'
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
.smiles {{
    overflow-wrap: anywhere;
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
Randomly selected molecules from each cluster.
Where possible, {molecules_per_cluster} unique molecules are shown per cluster.
</p>
{''.join(sections)}
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    method, input_path = choose_embedding()
    df = pd.read_csv(input_path)

    smiles_col = find_column(
        df, ["smiles", "SMILES", "canonical_smiles", "canonical smiles", "encoding"]
    )
    x_col, y_col = find_embedding_columns(df, method)

    print("\nAvailable numeric columns:")
    numeric_columns = list(df.select_dtypes(include=np.number).columns)
    print(", ".join(map(str, numeric_columns)))

    property_name = input("\nProperty to analyse [fitness]: ").strip() or "fitness"
    property_col = find_column(df, [property_name])

    print("\nChoose threshold type:")
    print("1 = Minimum value — retain and highlight high-performing molecules")
    print("2 = Maximum value — retain and highlight low-performing molecules")
    threshold_choice = input("Choice [1]: ").strip() or "1"

    if threshold_choice == "1":
        threshold_type = "min"
        threshold_label = "minimum"
    elif threshold_choice == "2":
        threshold_type = "max"
        threshold_label = "maximum"
    else:
        raise SystemExit("Threshold choice must be 1 or 2.")

    threshold_text = input(
        f"Enter the {threshold_label} {property_col} value: "
    ).strip()
    if not threshold_text:
        raise SystemExit("A threshold value is required.")

    threshold = float(threshold_text)

    required = [smiles_col, x_col, y_col, property_col]
    work = df.dropna(subset=required).copy()

    if threshold_type == "min":
        work = work[work[property_col] >= threshold].copy()
    else:
        work = work[work[property_col] <= threshold].copy()

    if len(work) < 3:
        raise SystemExit("Fewer than three usable molecules remain after filtering.")

    print(f"\nLoaded {len(df):,} rows from {input_path}")
    print(f"Usable rows after filtering: {len(work):,}")
    print(f"Embedding coordinates: {x_col}, {y_col}")

    output_dir = (
        Path("analysis_plots")
        / method
        / f"{property_col}_{threshold_type}_{safe_name(threshold)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    coords = work[[x_col, y_col]].to_numpy(float)
    scaled = StandardScaler().fit_transform(coords)

    dbscan_eps = float(input("\nDBSCAN eps [0.30]: ").strip() or "0.30")
    dbscan_min_samples = int(input("DBSCAN min_samples [10]: ").strip() or "10")
    birch_threshold = float(input("BIRCH threshold [0.50]: ").strip() or "0.50")
    birch_clusters_text = input(
        "BIRCH final number of clusters [None = automatic]: "
    ).strip()
    birch_clusters = int(birch_clusters_text) if birch_clusters_text else None

    work["DBSCAN_cluster"] = DBSCAN(
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
        n_jobs=-1,
    ).fit_predict(scaled)

    work["BIRCH_cluster"] = Birch(
        threshold=birch_threshold,
        n_clusters=birch_clusters,
    ).fit_predict(scaled)

    save_density_map(
        work,
        x_col,
        y_col,
        property_col,
        threshold_type,
        threshold,
        output_dir / f"{method}_{property_col}_{threshold_type}_kde_density.png",
    )

    for label, cluster_col in [
        ("dbscan", "DBSCAN_cluster"),
        ("birch", "BIRCH_cluster"),
    ]:
        # Cluster CSV output disabled to save disk space.

        save_3d_plot(
            work,
            x_col,
            y_col,
            property_col,
            smiles_col,
            cluster_col,
            f"{method.upper()} embedding — {label.upper()} clusters",
            output_dir / f"{label}_3d.html",
        )

        save_scaffold_report(
            work,
            smiles_col,
            property_col,
            cluster_col,
            output_dir / f"{label}_scaffolds.html",
        )

        save_representative_molecules_report(
            work,
            smiles_col,
            property_col,
            cluster_col,
            output_dir / f"{label}_representative_molecules.html",
            molecules_per_cluster=4,
        )

    print("\nSaved:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
