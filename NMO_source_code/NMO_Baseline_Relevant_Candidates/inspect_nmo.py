import sys
from pathlib import Path

import numpy as np
import pandas as pd


PARQUET_FILE = "nmo.parquet"
TOP_N = 10

IMPORTANT_COLUMNS = [
    "task",
    "hash",
    "molecular_junction__smiles",
    "molecular_junction__fitness",
    "molecular_junction__sa_score",
]

TASK_PROPERTY_COLUMNS = [
    "phonon_transport__kappa",
    "electronic_transport__conductance",
    "electronic_transport__seebeck",
    "electronic_transport__kappa_el",
    "electronic_transport__zt",
    "upconversion__log_p_upconversion",
    "upconversion__log_p_upconversion_scaled",
    "upconversion__molecular_length",
    "upconversion__surface_area",
]


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def shorten(value, max_length: int = 120) -> str:
    """Return a compact description of a potentially large dataset value."""

    if value is None:
        return "None"

    if isinstance(value, np.ndarray):
        return f"array with shape {value.shape}"

    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} with length {len(value)}"

    if isinstance(value, dict):
        return f"dictionary with {len(value)} entries"

    try:
        if pd.isna(value):
            return "NaN"
    except (TypeError, ValueError):
        pass

    text = str(value).replace("\n", "\\n")

    if len(text) > max_length:
        return text[:max_length] + "..."

    return text


def get_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return only columns that exist in the DataFrame."""
    return [column for column in columns if column in df.columns]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the Parquet dataset with a useful error message."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find '{file_path}'. "
            "Make sure the script is in the same directory as nmo.parquet."
        )

    return pd.read_parquet(file_path)


def print_dataset_summary(df: pd.DataFrame) -> None:
    print_section("DATASET SUMMARY")
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Memory used by DataFrame: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def print_task_counts(df: pd.DataFrame) -> None:
    print_section("NUMBER OF CANDIDATES IN EACH TASK")

    if "task" not in df.columns:
        print("The dataset does not contain a 'task' column.")
        return

    counts = df["task"].value_counts(dropna=False)

    for task, count in counts.items():
        print(f"{str(task):25s} {count:6,d}")


def print_column_summary(df: pd.DataFrame) -> None:
    print_section("COLUMN SUMMARY")

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null": [int(df[column].notna().sum()) for column in df.columns],
            "missing": [int(df[column].isna().sum()) for column in df.columns],
        }
    )

    print(summary.to_string(index=False))


def print_first_rows(df: pd.DataFrame) -> None:
    print_section("FIRST FIVE CANDIDATES")

    columns = get_existing_columns(df, IMPORTANT_COLUMNS)

    if not columns:
        print("None of the expected summary columns were found.")
        return

    print(
        df[columns]
        .head(5)
        .to_string(index=False, max_colwidth=80)
    )


def print_top_overall(df: pd.DataFrame) -> pd.DataFrame:
    print_section(f"TOP {TOP_N} CANDIDATES BY FITNESS")

    fitness_column = "molecular_junction__fitness"

    if fitness_column not in df.columns:
        print(f"Column '{fitness_column}' was not found.")
        return pd.DataFrame()

    top = (
        df.dropna(subset=[fitness_column])
        .sort_values(fitness_column, ascending=False)
        .head(TOP_N)
    )

    columns = get_existing_columns(df, IMPORTANT_COLUMNS)

    if top.empty:
        print("No non-null fitness values were found.")
    else:
        print(top[columns].to_string(index=False, max_colwidth=80))

    return top


def print_top_by_task(df: pd.DataFrame) -> pd.DataFrame:
    print_section(f"TOP {TOP_N} CANDIDATES FOR EACH TASK")

    task_column = "task"
    fitness_column = "molecular_junction__fitness"

    required = [task_column, fitness_column]

    if any(column not in df.columns for column in required):
        print("The required task or fitness column was not found.")
        return pd.DataFrame()

    valid = df.dropna(subset=[fitness_column]).copy()

    top_by_task = (
        valid.sort_values(
            [task_column, fitness_column],
            ascending=[True, False],
        )
        .groupby(task_column, group_keys=False)
        .head(TOP_N)
    )

    display_columns = get_existing_columns(
        df,
        [
            "molecular_junction__fitness",
            "molecular_junction__sa_score",
            "molecular_junction__smiles",
            "hash",
        ],
    )

    for task, task_df in top_by_task.groupby(task_column):
        print(f"\n--- {task} ---")
        print(task_df[display_columns].to_string(index=False, max_colwidth=80))

    return top_by_task


def print_task_property_summary(df: pd.DataFrame) -> None:
    print_section("TASK-SPECIFIC PROPERTY SUMMARY")

    available_properties = get_existing_columns(df, TASK_PROPERTY_COLUMNS)

    if not available_properties:
        print("No expected task-specific property columns were found.")
        return

    for column in available_properties:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        print(f"\n{column}")

        if values.empty:
            print("  No numerical values")
            continue

        print(f"  Count:  {len(values):,}")
        print(f"  Minimum: {values.min():.6g}")
        print(f"  Mean:    {values.mean():.6g}")
        print(f"  Median:  {values.median():.6g}")
        print(f"  Maximum: {values.max():.6g}")


def print_first_candidate_summary(df: pd.DataFrame) -> None:
    print_section("FIRST CANDIDATE: COMPACT FIELD SUMMARY")

    if df.empty:
        print("The dataset is empty.")
        return

    first_row = df.iloc[0]

    for column, value in first_row.items():
        print(f"{column}: {shorten(value)}")


def save_summary_files(
    df: pd.DataFrame,
    top_overall: pd.DataFrame,
    top_by_task: pd.DataFrame,
) -> None:
    print_section("SAVING SUMMARY FILES")

    important_columns = get_existing_columns(df, IMPORTANT_COLUMNS)

    summary_file = "nmo_important_columns.csv"
    df[important_columns].to_csv(summary_file, index=False)
    print(f"Saved: {summary_file}")

    if not top_overall.empty:
        overall_file = "nmo_top_10_overall.csv"
        top_overall[important_columns].to_csv(overall_file, index=False)
        print(f"Saved: {overall_file}")

    if not top_by_task.empty:
        task_file = "nmo_top_10_by_task.csv"
        top_by_task[important_columns].to_csv(task_file, index=False)
        print(f"Saved: {task_file}")


def inspect_row(df: pd.DataFrame, row_number: int) -> None:
    """Print a compact summary of a chosen row."""

    print_section(f"CANDIDATE AT ROW {row_number}")

    if row_number < 0 or row_number >= len(df):
        print(f"Invalid row number. Choose a value from 0 to {len(df) - 1}.")
        return

    row = df.iloc[row_number]

    for column, value in row.items():
        print(f"{column}: {shorten(value)}")


def main() -> None:
    file_path = Path(PARQUET_FILE)

    try:
        print(f"Loading {file_path}...")
        df = load_dataset(file_path)
    except Exception as error:
        print(f"Error loading dataset: {error}")
        sys.exit(1)

    print_dataset_summary(df)
    print_task_counts(df)
    print_column_summary(df)
    print_first_rows(df)

    top_overall = print_top_overall(df)
    top_by_task = print_top_by_task(df)

    print_task_property_summary(df)
    print_first_candidate_summary(df)

    save_summary_files(df, top_overall, top_by_task)

    # Optional: run `python inspect_nmo.py 25` to inspect row 25.
    if len(sys.argv) > 1:
        try:
            selected_row = int(sys.argv[1])
            inspect_row(df, selected_row)
        except ValueError:
            print("\nRow number must be an integer.")

    print_section("FINISHED")
    print("Dataset loaded and summaries created successfully.")
    print()
    print("To inspect a particular candidate, run:")
    print("    python inspect_nmo.py ROW_NUMBER")
    print()
    print("For example:")
    print("    python inspect_nmo.py 25")


if __name__ == "__main__":
    main()
