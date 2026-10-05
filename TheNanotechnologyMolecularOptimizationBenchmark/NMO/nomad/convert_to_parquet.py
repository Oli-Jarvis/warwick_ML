"""
NMO Benchmark — YAML > Parquet converter for Hugging Face
==========================================================
Flattens the nested NMOEntry YAML schema into a columnar Parquet file.

Naming convention:  section__field
  e.g.  benchmark_context.model_name  >  benchmark_context__model_name
        phonon_transport.kappa        >  phonon_transport__kappa
        structure.positions           "oracle_calls"  structure__positions   (list<list<float>>)

Usage
-----
  # Convert a folder of YAML files:
  python nmo_yaml_to_parquet.py --input "data/raw/**/*.yaml" --output nmo.parquet

  # Convert and push to the Hugging Face Hub:
  python nmo_yaml_to_parquet.py --input "data/raw/**/*.yaml" --output nmo.parquet \
      --push-to-hub your-username/nmo-benchmark
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


# ---------------------------------------------------------------------------
# Schema helpers — maps section key > list of (field, pyarrow type)
# Array fields use pa.list_() so they survive round-trips without squeezing.
# ---------------------------------------------------------------------------

SECTION_SCHEMAS: dict[str, list[tuple[str, pa.DataType]]] = {
    "": [  # top-level NMOEntry fields
        ("task",  pa.string()),
        ("hash",  pa.string()),
    ],
    "benchmark_context": [
        ("model_name",      pa.string()),
        ("model_version",   pa.string()),
        ("model_url",       pa.string()),
        ("hyperparameters", pa.string()),
        ("grammar",         pa.string()),
    ],
    "calculation_settings": [
        ("geometry_method",             pa.string()),
        ("hessian_method",              pa.string()),
        ("electronic_structure_method", pa.string()),
        ("transport_method",            pa.string()),
        ("upconversion_method",         pa.string()),
        ("temperature",                 pa.float64()),
        ("oracle_config",               pa.string()),
    ],
    "molecular_junction": [
        ("smiles",      pa.string()),
        ("encoding",    pa.string()),
        ("sa_score",    pa.float64()),
        ("hl_gap",      pa.float64()),
        ("oracle_call", pa.int64()),
        ("fitness",     pa.float64()),
    ],
    "structure": [
        ("species",   pa.list_(pa.string())),
        ("positions", pa.list_(pa.list_(pa.float64()))),  # N×3
    ],
    "phonon_transport": [
        ("electrode_model", pa.string()),
        ("kappa",           pa.float64()),
        ("energy",          pa.list_(pa.float64())),
        ("transmission",    pa.list_(pa.float64())),
    ],
    "electronic_transport": [
        ("electrode_geometry", pa.string()),
        ("electrode_method",   pa.string()),
        ("conductance",        pa.float64()),
        ("seebeck",            pa.float64()),
        ("kappa_el",           pa.float64()),
        ("zt",                 pa.float64()),
        ("energy",             pa.list_(pa.float64())),
        ("transmission",       pa.list_(pa.float64())),
    ],
    "upconversion": [
        ("log_p_upconversion",        pa.float64()),
        ("log_p_upconversion_scaled", pa.float64()),
        ("molecular_length",          pa.float64()),
        ("surface_area",              pa.float64()),
    ],
}


def _col_name(section: str, field: str) -> str:
    return field if section == "" else f"{section}__{field}"


def build_schema() -> pa.Schema:
    fields = []
    for section, field_types in SECTION_SCHEMAS.items():
        for field, dtype in field_types:
            fields.append(pa.field(_col_name(section, field), dtype, nullable=True))
    return pa.schema(fields)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _to_native(val: Any) -> Any:
    """Convert numpy scalars/arrays to Python native types for PyArrow."""
    if val is None:
        return None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def extract_entry(entry: dict) -> dict:
    """
    Flatten one NMOEntry dict into a flat {col_name: value} dict.
    Missing optional sections produce None values.
    """
    row: dict[str, Any] = {}

    # Top-level scalars
    for field, _ in SECTION_SCHEMAS[""]:
        row[field] = _to_native(entry.get(field))

    # Sub-sections
    for section, field_types in SECTION_SCHEMAS.items():
        if section == "":
            continue
        sub = entry.get(section) or {}
        for field, _ in field_types:
            col = _col_name(section, field)
            raw = sub.get(field)

            if isinstance(raw, np.ndarray):
                raw = raw.tolist()
            elif isinstance(raw, list):
                raw = [_to_native(v) for v in raw]

            row[col] = raw

    return row


# ---------------------------------------------------------------------------
# YAML loading — supports single-document and multi-document files
# ---------------------------------------------------------------------------

def load_yaml_file(path: Path) -> list[dict]:
    """Return a list of NMOEntry dicts from a single YAML file.

    Handles plain dicts, lists of dicts, and NOMAD archive files
    (where the entry lives under a top-level 'data' key).
    """
    with open(path, "r", encoding="utf-8") as fh:
        docs = list(yaml.safe_load_all(fh))

    entries = []
    for doc in docs:
        if doc is None:
            continue
        # NOMAD .archive.yaml — unwrap the 'data' section
        if isinstance(doc, dict) and "data" in doc:
            doc = doc["data"]
            # drop the m_def pointer, it's not a data field
            if isinstance(doc, dict):
                doc.pop("m_def", None)
        if isinstance(doc, list):
            entries.extend(doc)
        elif isinstance(doc, dict):
            entries.append(doc)
    return entries


# ---------------------------------------------------------------------------
# Main conversion — batched + parallel
# ---------------------------------------------------------------------------

def _process_file(path: str) -> list[dict]:
    """Worker: load one YAML file and return extracted rows. Must be top-level for pickling."""
    try:
        return [extract_entry(e) for e in load_yaml_file(Path(path))]
    except Exception as exc:
        print(f"  [WARN] {path}: {exc}")
        return []


def _rows_to_batch(rows: list[dict], schema: pa.Schema) -> pa.RecordBatch:
    columns = {f.name: [] for f in schema}
    for row in rows:
        for col in columns:
            columns[col].append(row.get(col))
    arrays = [pa.array(columns[f.name], type=f.type) for f in schema]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def convert(
    input_glob: str,
    output_path: str,
    batch_size: int = 500,
    workers: int = 4,
) -> int:
    """
    Convert YAML files to a single Parquet file.

    Processes files in parallel (workers) and flushes to disk every
    batch_size files so memory stays flat regardless of total count.

    Returns the total number of rows written.
    """
    import glob as _glob
    from multiprocessing import Pool
    from multiprocessing.pool import AsyncResult

    schema = build_schema()
    input_paths = sorted(_glob.glob(input_glob, recursive=True))

    if not input_paths:
        raise FileNotFoundError(f"No YAML files matched: {input_glob!r}")

    n = len(input_paths)
    print(f"Found {n} YAML file(s) | batch_size={batch_size} | workers={workers}")

    total_rows = 0
    writer = pq.ParquetWriter(output_path, schema, compression="snappy")

    with Pool(processes=workers) as pool:
        # Slide a window of batch_size files through the list
        for batch_start in range(0, n, batch_size):
            batch_paths = input_paths[batch_start : batch_start + batch_size]
            batch_end   = batch_start + len(batch_paths)

            # Parse all files in the batch in parallel
            results: list[list[dict]] = pool.map(_process_file, batch_paths)

            # Flatten into one list of rows
            rows = [row for file_rows in results for row in file_rows]

            if rows:
                record_batch = _rows_to_batch(rows, schema)
                writer.write_batch(record_batch)
                total_rows += len(rows)

            print(f"  [{batch_end}/{n}] +{len(rows)} rows  (total {total_rows})")

    writer.close()
    print(f"\nDone — {total_rows} rows > {output_path}")
    return total_rows


# ---------------------------------------------------------------------------
# Optional: push to Hugging Face Hub
# ---------------------------------------------------------------------------

def push_to_hub(parquet_path: str, repo_id: str) -> None:
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Run: pip install datasets huggingface_hub")

    print(f"\nPushing to Hub: {repo_id}")
    ds = Dataset.from_parquet(parquet_path)
    ds.push_to_hub(repo_id)
    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NMO YAML files to Parquet.")
    parser.add_argument(
        "--input", required=True,
        help="Glob pattern for YAML files, e.g. 'data/raw/**/*.yaml'",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output Parquet file path, e.g. nmo.parquet",
    )
    parser.add_argument(
        "--push-to-hub", default=None, metavar="REPO_ID",
        help="Hugging Face repo to push to, e.g. 'your-username/nmo-benchmark'",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, metavar="N",
        help="Files per batch before flushing to disk (default: 500)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Parallel worker processes for YAML parsing (default: 4)",
    )
    args = parser.parse_args()

    convert(args.input, args.output, batch_size=args.batch_size, workers=args.workers)

    if args.push_to_hub:
        push_to_hub(args.output, args.push_to_hub)