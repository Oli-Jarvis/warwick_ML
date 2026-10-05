from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from chemplot import Plotter


PARQUET_FILE = Path("nmo.parquet")

OUTPUT_PNG = "nmo_full_chemical_space.png"
OUTPUT_HTML = "nmo_full_chemical_space.html"
OUTPUT_CSV = "nmo_full_chemical_space_data.csv"

RANDOM_SEED = 42


def clean_smiles_for_chemplot(smiles: str) -> str:
    """
    Remove explicit gold atoms before ChemPlot fingerprint generation.

    The original NMO SMILES is preserved in the output CSV. The cleaned
    version is used only to construct RDKit/ChemPlot fingerprints.
    """
    return smiles.replace("[Au]", "")


def load_nmo_data() -> pd.DataFrame:
    """
    Load only the columns needed for the chemical-space plot.
    """

    if not PARQUET_FILE.exists():
        raise FileNotFoundError(
            f"Could not find '{PARQUET_FILE}'. "
            "Run this script from the directory containing nmo.parquet."
        )

    columns = [
        "task",
        "hash",
        "molecular_junction__smiles",
        "molecular_junction__fitness",
        "molecular_junction__sa_score",
    ]

    df = pd.read_parquet(
        PARQUET_FILE,
        columns=columns,
    )

    # Remove rows that cannot be plotted.
    df = df.dropna(
        subset=[
            "task",
            "molecular_junction__smiles",
        ]
    ).copy()

    # Create a separate SMILES column for ChemPlot.
    df["chemplot_smiles"] = df[
        "molecular_junction__smiles"
    ].map(clean_smiles_for_chemplot)

    # Remove any molecules that became empty after cleaning.
    df = df[
        df["chemplot_smiles"].str.strip().ne("")
    ].reset_index(drop=True)

    return df


def create_plotter(df: pd.DataFrame) -> Plotter:
    """
    Convert SMILES into structural fingerprints and create a ChemPlot object.
    """

    smiles = df["chemplot_smiles"].tolist()
    task_labels = df["task"].tolist()

    return Plotter.from_smiles(
        smiles,
        target=task_labels,
        target_type="C",
        sim_type="structural",
    )


def save_static_plot(plotter: Plotter) -> None:
    """
    Save the UMAP chemical-space embedding as a PNG image.
    """

    print("Creating static plot...")

    plotter.visualize_plot(
        size=14,
        remove_outliers=False,
    )

    plt.title(
        "NMO chemical space by benchmark task",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved static plot: {OUTPUT_PNG}")


def save_interactive_plot(plotter: Plotter) -> None:
    """
    Save the interactive Bokeh plot directly through ChemPlot's
    filename argument.
    """

    print("Creating interactive plot...")

    try:
        plotter.interactive_plot(
            size=900,
            kind="scatter",
            remove_outliers=False,
            is_colored=True,
            clusters=False,
            filename=OUTPUT_HTML,
            show_plot=False,
            title="NMO chemical space by benchmark task",
        )

        if Path(OUTPUT_HTML).exists():
            print(f"Saved interactive plot: {OUTPUT_HTML}")
        else:
            print(
                "ChemPlot completed without raising an error, but "
                f"'{OUTPUT_HTML}' was not found."
            )

    except Exception as error:
        print(f"Could not save interactive plot: {error}")


def main() -> None:
    print("Loading NMO dataset...")

    try:
        df = load_nmo_data()
    except Exception as error:
        print(f"Failed to load dataset: {error}")
        sys.exit(1)

    print(f"Number of plotted molecules: {len(df):,}")

    print("\nTask counts:")
    print(df["task"].value_counts().to_string())

    # Save the metadata for all plotted molecules.
    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(f"\nSaved plotted molecule data: {OUTPUT_CSV}")

    print("\nConstructing structural chemical space...")
    print(
        f"This may take several minutes for {len(df):,} molecules."
    )

    try:
        plotter = create_plotter(df)
    except Exception as error:
        print(f"Failed to construct chemical fingerprints: {error}")
        sys.exit(1)

    print("Running UMAP...")

    try:
        plotter.umap(
        )
    except Exception as error:
        print(f"UMAP failed: {error}")
        sys.exit(1)

    save_static_plot(plotter)
    save_interactive_plot(plotter)

    print("\nFinished.")


if __name__ == "__main__":
    main()
