from pathlib import Path
import argparse
import sys

import pandas as pd
import plotly.express as px
from chemplot import Plotter
from rdkit import Chem


def clean_smiles_for_chemplot(smiles: str) -> str:
    """
    Remove explicit gold atoms before ChemPlot fingerprint generation.

    The original NMO SMILES is preserved in the exported CSV.
    The cleaned version is used only for ChemPlot/RDKit.
    """
    return str(smiles).replace("[Au]", "").strip()


def load_history(
    input_file: Path,
    delimiter: str,
    min_fitness: float,
) -> pd.DataFrame:
    """
    Load the optimisation-history CSV and prepare the working ChemPlot SMILES.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find '{input_file}'."
        )

    df = pd.read_csv(
        input_file,
        sep=delimiter,
        engine="python",
        on_bad_lines="warn",
    )

    df.columns = df.columns.astype(str).str.strip()

    required = [
        "encoding",
        "smiles",
        "fitness",
        "SA",
        "N_rot",
        "log_P_upconversion",
        "log_P_upconversion_scaled",
        "molecular_length",
        "surface_area",
        "P_upconversion",
        "hl_gaps",
        "oracle_calls",
        "created_by",
        "step",
        "generation",
        "mutation_stats",
        "crossover_stats",
        "hash_values",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df = df.copy()
    df.insert(0, "source_row", df.index + 2)

    numeric_columns = [
        "fitness",
        "SA",
        "N_rot",
        "log_P_upconversion",
        "log_P_upconversion_scaled",
        "molecular_length",
        "surface_area",
        "P_upconversion",
        "hl_gaps",
        "oracle_calls",
        "step",
        "generation",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    text_columns = [
        "encoding",
        "smiles",
        "created_by",
        "mutation_stats",
        "crossover_stats",
        "hash_values",
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    df = df.dropna(
        subset=["encoding", "fitness"]
    ).copy()

    df = df[
        df["encoding"].ne("")
        & (df["fitness"] > min_fitness)
    ].copy()

    # IMPORTANT:
    # In full_history.csv, "encoding" is the clean molecular SMILES.
    # The "smiles" column is the expanded Au-contact representation used
    # internally by NMO and should not be passed to ChemPlot for drawing.
    df["chemplot_smiles"] = df["encoding"].astype(str).str.strip()

    df = df[
        df["chemplot_smiles"].ne("")
    ].copy()

    valid_mask = df["chemplot_smiles"].map(
        lambda value: (
            Chem.MolFromSmiles(str(value)) is not None
        )
    )

    invalid_count = int((~valid_mask).sum())

    if invalid_count:
        print(
            f"Discarding {invalid_count:,} rows whose cleaned "
            "SMILES could not be parsed by RDKit."
        )

    return df[valid_mask].reset_index(drop=True)


def create_plotter(
    df: pd.DataFrame,
) -> Plotter:
    """
    Build the ChemPlot object using the same approach as the working script.
    """

    smiles = df["chemplot_smiles"].tolist()
    fitness = df["fitness"].tolist()

    return Plotter.from_smiles(
        smiles,
        target=fitness,
        target_type="R",
        sim_type="structural",
    )


def save_chemplot_html(
    plotter: Plotter,
    output_html: Path,
) -> None:
    """
    Let ChemPlot write its own interactive HTML directly.

    This is intentionally the same export route as the original working script.
    """
    print("Creating the native ChemPlot interactive plot...")

    plotter.interactive_plot(
        size=900,
        kind="scatter",
        remove_outliers=False,
        is_colored=True,
        clusters=False,
        filename=str(output_html),
        show_plot=False,
        title="NMO chemical space coloured by fitness",
    )

    if not output_html.exists():
        raise RuntimeError(
            "ChemPlot completed without raising an error, "
            f"but '{output_html}' was not created."
        )

    print(f"Saved ChemPlot plot: {output_html}")


def save_fitness_plot(
    df: pd.DataFrame,
    coordinates: pd.DataFrame,
    output_html: Path,
) -> None:
    """
    Create a separate 3D Plotly fitness landscape from the same UMAP.
    """

    plot_df = pd.concat(
        [
            df.reset_index(drop=True),
            coordinates,
        ],
        axis=1,
    )

    figure = px.scatter_3d(
        plot_df,
        x="UMAP-1",
        y="UMAP-2",
        z="fitness",
        color="fitness",
        custom_data=[
            "encoding",
            "smiles",
            "chemplot_smiles",
            "SA",
            "N_rot",
            "log_P_upconversion",
            "log_P_upconversion_scaled",
            "molecular_length",
            "surface_area",
            "P_upconversion",
            "hl_gaps",
            "oracle_calls",
            "created_by",
            "step",
            "generation",
            "mutation_stats",
            "crossover_stats",
            "hash_values",
            "source_row",
        ],
        color_continuous_scale="Viridis",
        title=(
            "Fitness above the same ChemPlot UMAP chemical space"
        ),
    )

    figure.update_traces(
        marker={
            "size": 4,
            "opacity": 0.82,
        },
        hovertemplate=(
            "<b>Fitness:</b> %{z:.6f}<br>"
            "<b>SA:</b> %{customdata[3]:.4f}<br>"
            "<b>Rotatable bonds:</b> %{customdata[4]}<br>"
            "<b>logP:</b> %{customdata[8]:.4f}<br>"
            "<b>Scaled logP:</b> %{customdata[8]:.4f}<br>"
            "<b>Molecular length:</b> %{customdata[8]:.4f}<br>"
            "<b>Surface area:</b> %{customdata[8]:.4f}<br>"
            "<b>P upconversion:</b> %{customdata[10]:.6g}<br>"
            "<b>HL gap:</b> %{customdata[10]:.6g}<br>"
            "<b>Oracle calls:</b> %{customdata[14]}<br>"
            "<b>Created by:</b> %{customdata[14]}<br>"
            "<b>Step:</b> %{customdata[14]}<br>"
            "<b>Generation:</b> %{customdata[14]}<br>"
            "<b>Hash:</b> %{customdata[18]}<br>"
            "<b>CSV row:</b> %{customdata[18]}<br>"
            "<b>Encoding:</b> %{customdata[0]}<br><b>Expanded NMO SMILES:</b> %{customdata[1]}"
            "<extra></extra>"
        ),
    )

    figure.update_layout(
        scene={
            "xaxis_title": "ChemPlot UMAP-1",
            "yaxis_title": "ChemPlot UMAP-2",
            "zaxis_title": "Fitness",
        },
        height=900,
        margin={
            "l": 0,
            "r": 0,
            "t": 70,
            "b": 0,
        },
    )

    figure.write_html(
        output_html,
        include_plotlyjs=True,
        full_html=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    print(f"Saved 3D fitness plot: {output_html}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ChemPlot's native interactive 2D plot first, "
            "then create a separate 3D fitness landscape from the "
            "same UMAP coordinates."
        )
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default="full_history.csv",
    )

    parser.add_argument(
        "--delimiter",
        default=";",
    )

    parser.add_argument(
        "--min-fitness",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--prefix",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_file = Path(args.input_file)
    prefix = args.prefix or input_file.stem

    output_directory = input_file.parent

    chemplot_html = (
        output_directory
        / f"{prefix}_chemplot_interactive.html"
    )

    fitness_html = (
        output_directory
        / f"{prefix}_fitness_3d.html"
    )

    output_csv = (
        output_directory
        / f"{prefix}_plotted_data.csv"
    )

    try:
        print("Loading optimisation history...")

        df = load_history(
            input_file=input_file,
            delimiter=args.delimiter,
            min_fitness=args.min_fitness,
        )

        if len(df) < 3:
            raise RuntimeError(
                "Fewer than three valid molecules remain "
                "after filtering."
            )

        print(f"Number of plotted molecules: {len(df):,}")

        df.to_csv(
            output_csv,
            index=False,
        )

        print(f"Saved plotted data: {output_csv}")

        print("Constructing structural chemical space...")

        plotter = create_plotter(df)

        print("Running ChemPlot UMAP...")

        # ChemPlot returns the calculated coordinates directly.
        # Capturing this return value avoids relying on private,
        # version-specific attributes inside the Plotter object.
        coordinates = plotter.umap()

        if not isinstance(coordinates, pd.DataFrame):
            coordinates = pd.DataFrame(coordinates)

        coordinates = coordinates.iloc[:, :2].copy()
        coordinates.columns = ["UMAP-1", "UMAP-2"]
        coordinates = coordinates.reset_index(drop=True)

        if len(coordinates) != len(df):
            raise RuntimeError(
                "ChemPlot returned a different number of UMAP "
                "coordinates from plotted molecules."
            )

        # Important: ChemPlot writes its own HTML directly so the
        # native molecular hover images remain intact.
        save_chemplot_html(
            plotter=plotter,
            output_html=chemplot_html,
        )

        save_fitness_plot(
            df=df,
            coordinates=coordinates,
            output_html=fitness_html,
        )

    except Exception as error:
        print(f"Failed: {error}")
        sys.exit(1)

    print()
    print("Finished.")
    print(f"2D ChemPlot: {chemplot_html}")
    print(f"3D fitness:  {fitness_html}")


if __name__ == "__main__":
    main()
