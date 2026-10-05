#!/usr/bin/env python3
"""Compare one proposed molecule with a molecular reference set.

The input format is detected automatically:

* FragNet_embedding_* columns -> cosine similarity in the trained model's
  embedding space.
* A SMILES column (including the dimensionality-reduction output CSV) ->
  Morgan fingerprint Tanimoto similarity.

The script scores the full reference set, saves the top-k neighbours, reports
the nearest-neighbour and mean top-k similarities, and creates a bar chart. If
the reference file contains PCA_1/PCA_2, it also highlights the neighbours on
the existing dimensionality-reduction plane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


EMBEDDING_PREFIX = "FragNet_embedding_"
SMILES_CANDIDATES = (
    "smiles",
    "chemplot_smiles",
    "SMILES",
    "canonical_smiles",
    "canonical smiles",
    "encoding",
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the nearest and top-k training-set neighbours of a proposed "
            "molecule using Morgan-Tanimoto or FragNet embedding similarity."
        )
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help=(
            "Reference CSV: FragNet molecular embeddings, FragNet train.csv, "
            "or combined_labelled_pca_embedding.csv."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("auto", "morgan", "embedding"),
        default="auto",
        help="Similarity method. Auto uses embeddings when present, otherwise Morgan.",
    )
    query = parser.add_mutually_exclusive_group()
    query.add_argument(
        "--query-smiles",
        help=(
            "Proposed molecule as SMILES. For embedding similarity, the same "
            "SMILES must occur in --query-file or --reference."
        ),
    )
    query.add_argument(
        "--query-index",
        type=int,
        default=0,
        help=(
            "Zero-based row in --query-file, or in the filtered reference when "
            "no query file is supplied. Default: 0 (useful as an example)."
        ),
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        help=(
            "Optional CSV containing the proposed molecule and, for embedding "
            "mode, its FragNet_embedding_* columns."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--reference-split",
        default="train",
        help=(
            "If the reference has a split column, keep this split. Use 'all' "
            "to disable filtering. Default: train."
        ),
    )
    parser.add_argument("--smiles-column", help="Override SMILES column detection.")
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include exact copies of the query in the reference results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("molecule_similarity"),
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, sep=None, engine="python", on_bad_lines="warn")
    data.columns = data.columns.astype(str).str.strip()
    if data.empty:
        raise ValueError(f"No rows found in {path}")
    return data


def find_smiles_column(data: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in data.columns:
            raise KeyError(
                f"SMILES column {requested!r} not found. Available columns: "
                + ", ".join(data.columns)
            )
        return requested

    lookup = {column.lower(): column for column in data.columns}
    for candidate in SMILES_CANDIDATES:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def embedding_columns(data: pd.DataFrame) -> list[str]:
    columns = [
        column for column in data.columns if column.startswith(EMBEDDING_PREFIX)
    ]
    return sorted(columns)


def canonical_smiles(value) -> str | None:
    if pd.isna(value):
        return None
    molecule = Chem.MolFromSmiles(str(value).strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def resolve_method(args: argparse.Namespace, reference: pd.DataFrame) -> str:
    if args.method != "auto":
        return args.method
    return "embedding" if embedding_columns(reference) else "morgan"


def filter_reference(
    reference: pd.DataFrame,
    requested_split: str,
) -> pd.DataFrame:
    reference = reference.copy()
    reference["_reference_row"] = np.arange(len(reference))
    if "split" in reference.columns and requested_split.lower() != "all":
        keep = reference["split"].astype(str).str.lower() == requested_split.lower()
        reference = reference.loc[keep].copy()
        if reference.empty:
            raise ValueError(f"No reference rows have split={requested_split!r}")
        print(f"Reference split: {requested_split} ({len(reference):,} rows)")
    elif "split" not in reference.columns:
        print(
            "Reference has no split column; using all rows in the supplied file "
            f"({len(reference):,})."
        )
    else:
        print(f"Using all reference splits ({len(reference):,} rows).")
    return reference.reset_index(drop=True)


def select_query_row(
    args: argparse.Namespace,
    reference: pd.DataFrame,
    query_data: pd.DataFrame,
    query_smiles_column: str | None,
) -> tuple[pd.Series | None, int | None]:
    if args.query_smiles is not None:
        if query_smiles_column is None:
            return None, None
        wanted = canonical_smiles(args.query_smiles)
        if wanted is None:
            raise ValueError("--query-smiles is not valid RDKit SMILES.")
        canonical = query_data[query_smiles_column].map(canonical_smiles)
        matches = np.flatnonzero(canonical.to_numpy() == wanted)
        if not len(matches):
            return None, None
        position = int(matches[0])
        if len(matches) > 1:
            print(
                f"Query SMILES occurs {len(matches)} times; using the first "
                "matching embedding/row."
            )
        return query_data.iloc[position], position

    index = int(args.query_index)
    if index < 0 or index >= len(query_data):
        raise IndexError(
            f"--query-index {index} is outside the available range "
            f"0..{len(query_data) - 1}."
        )
    return query_data.iloc[index], index


def query_smiles_value(
    args: argparse.Namespace,
    query_row: pd.Series | None,
    smiles_column: str | None,
) -> str | None:
    if args.query_smiles is not None:
        return args.query_smiles
    if query_row is not None and smiles_column and smiles_column in query_row.index:
        return str(query_row[smiles_column])
    return None


def exclude_query_copies(
    table: pd.DataFrame,
    similarities: np.ndarray,
    query_canonical: str | None,
    reference_smiles_column: str | None,
    query_row: pd.Series | None,
    same_source: bool,
    include_self: bool,
) -> np.ndarray:
    keep = np.ones(len(table), dtype=bool)
    if include_self:
        return keep

    if query_canonical is not None and reference_smiles_column is not None:
        reference_canonical = table[reference_smiles_column].map(canonical_smiles)
        keep &= reference_canonical.to_numpy() != query_canonical

    if same_source and query_row is not None and "_reference_row" in query_row.index:
        keep &= table["_reference_row"].to_numpy() != query_row["_reference_row"]

    keep &= np.isfinite(similarities)
    return keep


def score_morgan(
    args: argparse.Namespace,
    reference: pd.DataFrame,
    query_smiles: str | None,
    reference_smiles_column: str | None,
    query_row: pd.Series | None,
    same_source: bool,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    if reference_smiles_column is None:
        raise KeyError("Morgan mode requires a SMILES column in the reference.")
    if query_smiles is None:
        raise ValueError(
            "Morgan mode requires --query-smiles or a query row with SMILES."
        )

    query_molecule = Chem.MolFromSmiles(str(query_smiles).strip())
    if query_molecule is None:
        raise ValueError("The proposed molecule is not valid RDKit SMILES.")
    query_canonical = Chem.MolToSmiles(query_molecule, canonical=True)
    query_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        query_molecule,
        radius=args.morgan_radius,
        nBits=args.morgan_bits,
    )

    valid_rows = []
    fingerprints = []
    for position, value in enumerate(reference[reference_smiles_column]):
        molecule = Chem.MolFromSmiles(str(value).strip()) if not pd.isna(value) else None
        if molecule is None:
            continue
        valid_rows.append(position)
        fingerprints.append(
            AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                radius=args.morgan_radius,
                nBits=args.morgan_bits,
            )
        )

    if not fingerprints:
        raise ValueError("The reference contains no valid molecular SMILES.")
    table = reference.iloc[valid_rows].reset_index(drop=True)
    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(query_fingerprint, fingerprints),
        dtype=float,
    )
    keep = exclude_query_copies(
        table,
        similarities,
        query_canonical,
        reference_smiles_column,
        query_row,
        same_source,
        args.include_self,
    )
    return table.loc[keep].reset_index(drop=True), similarities[keep], query_canonical


def score_embeddings(
    args: argparse.Namespace,
    reference: pd.DataFrame,
    query_data: pd.DataFrame,
    query_row: pd.Series | None,
    query_smiles: str | None,
    reference_smiles_column: str | None,
    same_source: bool,
) -> tuple[pd.DataFrame, np.ndarray, str | None]:
    columns = embedding_columns(reference)
    if not columns:
        raise KeyError(
            f"Embedding mode requires columns beginning with {EMBEDDING_PREFIX!r}."
        )
    missing = [column for column in columns if column not in query_data.columns]
    if missing:
        raise KeyError(
            "Query file does not contain the same FragNet embedding columns. "
            f"First missing column: {missing[0]}"
        )
    if query_row is None:
        raise ValueError(
            "That SMILES is not present in an embedding file. Generate its "
            "embedding with the same ft.pt model, then pass that CSV with "
            "--query-file."
        )

    matrix = reference[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    query_vector = pd.to_numeric(query_row[columns], errors="coerce").to_numpy(float)
    if not np.isfinite(query_vector).all() or np.linalg.norm(query_vector) == 0:
        raise ValueError("The query embedding contains invalid values or has zero norm.")

    norms = np.linalg.norm(matrix, axis=1)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(norms) & (norms > 0)
    table = reference.loc[valid].reset_index(drop=True)
    matrix = matrix[valid]
    similarities = (matrix @ query_vector) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    )

    query_canonical = canonical_smiles(query_smiles) if query_smiles else None
    keep = exclude_query_copies(
        table,
        similarities,
        query_canonical,
        reference_smiles_column,
        query_row,
        same_source,
        args.include_self,
    )
    return table.loc[keep].reset_index(drop=True), similarities[keep], query_canonical


def ranked_neighbours(
    table: pd.DataFrame,
    similarities: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if not len(table):
        raise ValueError("No reference molecules remain after excluding the query.")
    actual_k = min(top_k, len(table))
    order = np.argsort(-similarities, kind="stable")[:actual_k]

    preferred_metadata = [
        "smiles",
        "chemplot_smiles",
        "split",
        "split_row",
        "dataset_label",
        "source_run",
        "source_row",
        "PCA_1",
        "PCA_2",
    ]
    keep_columns = [column for column in preferred_metadata if column in table.columns]
    result = table.iloc[order][keep_columns].copy().reset_index(drop=True)
    result.insert(0, "similarity", similarities[order])
    result.insert(0, "rank", np.arange(1, actual_k + 1))
    return result


def save_similarity_bar(
    neighbours: pd.DataFrame,
    smiles_column: str | None,
    method: str,
    output_path: Path,
) -> None:
    labels = []
    for _, row in neighbours.iterrows():
        if smiles_column and smiles_column in row.index:
            text = str(row[smiles_column])
            text = text if len(text) <= 55 else text[:52] + "..."
        else:
            text = "reference molecule"
        labels.append(f"{int(row['rank'])}. {text}")

    plot_data = neighbours.iloc[::-1]
    plot_labels = labels[::-1]
    fig_height = max(5.0, 0.48 * len(neighbours) + 1.7)
    plt.figure(figsize=(11, fig_height))
    plt.barh(plot_labels, plot_data["similarity"], color="#386cb0")
    plt.xlabel("Cosine similarity" if method == "embedding" else "Morgan Tanimoto similarity")
    plt.title(f"Top-{len(neighbours)} nearest reference molecules")
    for index, value in enumerate(plot_data["similarity"]):
        plt.text(float(value), index, f" {float(value):.3f}", va="center")
    lower = min(0.0, float(neighbours["similarity"].min()) - 0.05)
    plt.xlim(lower, max(1.02, float(neighbours["similarity"].max()) + 0.08))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dimensionality_plot(
    reference: pd.DataFrame,
    neighbours: pd.DataFrame,
    query_row: pd.Series | None,
    output_path: Path,
) -> bool:
    if not {"PCA_1", "PCA_2"}.issubset(reference.columns):
        return False

    plt.figure(figsize=(10, 8))
    colours = {"smiles": "#1f77b4", "ggs": "#ff7f0e", "DFT": "#2ca02c"}
    if "dataset_label" in reference.columns:
        for label, group in reference.groupby("dataset_label", sort=False):
            plt.scatter(
                group["PCA_1"],
                group["PCA_2"],
                s=7,
                alpha=0.22,
                linewidths=0,
                color=colours.get(str(label), "#999999"),
                label=str(label),
            )
    else:
        plt.scatter(
            reference["PCA_1"],
            reference["PCA_2"],
            s=7,
            alpha=0.22,
            linewidths=0,
            color="#999999",
            label="reference",
        )

    if {"PCA_1", "PCA_2"}.issubset(neighbours.columns):
        points = plt.scatter(
            neighbours["PCA_1"],
            neighbours["PCA_2"],
            c=neighbours["similarity"],
            cmap="viridis",
            s=80,
            edgecolors="black",
            linewidths=0.8,
            label="top-k neighbours",
            zorder=4,
        )
        plt.colorbar(points, label="Similarity")
        for _, row in neighbours.iterrows():
            plt.annotate(
                str(int(row["rank"])),
                (row["PCA_1"], row["PCA_2"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    if query_row is not None and {"PCA_1", "PCA_2"}.issubset(query_row.index):
        if pd.notna(query_row["PCA_1"]) and pd.notna(query_row["PCA_2"]):
            plt.scatter(
                [query_row["PCA_1"]],
                [query_row["PCA_2"]],
                marker="*",
                s=260,
                color="#d62728",
                edgecolors="black",
                linewidths=1,
                label="query molecule",
                zorder=5,
            )

    plt.xlabel("Bonding PC1")
    plt.ylabel("Structural PC1")
    plt.title("Nearest neighbours on the existing dimensionality-reduction plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def main() -> None:
    args = get_args()
    reference_path = args.reference.expanduser().resolve()
    reference_unfiltered = read_table(reference_path)
    reference = filter_reference(reference_unfiltered, args.reference_split)
    method = resolve_method(args, reference)

    if args.query_file:
        query_path = args.query_file.expanduser().resolve()
        query_data = read_table(query_path)
        same_source = query_path == reference_path
        if same_source:
            query_data = reference
    else:
        query_data = reference
        same_source = True

    reference_smiles_column = find_smiles_column(reference, args.smiles_column)
    query_smiles_column = find_smiles_column(query_data, args.smiles_column)
    query_row, _ = select_query_row(
        args,
        reference,
        query_data,
        query_smiles_column,
    )
    query_smiles = query_smiles_value(args, query_row, query_smiles_column)

    if method == "morgan":
        scored_table, similarities, query_identifier = score_morgan(
            args,
            reference,
            query_smiles,
            reference_smiles_column,
            query_row,
            same_source,
        )
        method_label = "Morgan fingerprint Tanimoto"
    else:
        scored_table, similarities, query_identifier = score_embeddings(
            args,
            reference,
            query_data,
            query_row,
            query_smiles,
            reference_smiles_column,
            same_source,
        )
        method_label = "FragNet embedding cosine"

    neighbours = ranked_neighbours(scored_table, similarities, args.top_k)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    neighbours_path = output_dir / "top_k_neighbours.csv"
    summary_path = output_dir / "similarity_summary.csv"
    bar_path = output_dir / "top_k_similarity.png"
    dim_path = output_dir / "top_k_on_dimensionality_reduction.png"

    neighbours.to_csv(neighbours_path, index=False)
    top_scores = neighbours["similarity"].to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "method": method_label,
                "query_smiles": query_identifier,
                "reference_file": str(reference_path),
                "reference_molecules_scored": len(scored_table),
                "top_k": len(neighbours),
                "nearest_neighbour_similarity": top_scores[0],
                "mean_top_k_similarity": top_scores.mean(),
                "median_top_k_similarity": np.median(top_scores),
                "lowest_top_k_similarity": top_scores.min(),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)
    save_similarity_bar(neighbours, reference_smiles_column, method, bar_path)
    made_dimensionality_plot = save_dimensionality_plot(
        reference,
        neighbours,
        query_row,
        dim_path,
    )

    print("\nSimilarity analysis complete")
    print(f"Method: {method_label}")
    print(f"Reference molecules scored: {len(scored_table):,}")
    print(f"Nearest-neighbour similarity: {top_scores[0]:.4f}")
    print(f"Mean top-{len(neighbours)} similarity: {top_scores.mean():.4f}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {neighbours_path}")
    print(f"Saved: {bar_path}")
    if made_dimensionality_plot:
        print(f"Saved: {dim_path}")


if __name__ == "__main__":
    main()
