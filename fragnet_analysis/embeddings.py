#!/usr/bin/env python3
"""Generate a plain FragNet fragment and fragment-connection HTML report.

Typical usage, inside the Python environment that already contains FragNet:

    python embeddings.py --top-n 10

The generated report is an ordinary offline HTML file; it does not require
Streamlit, an internet connection, or any JavaScript packages.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("fragnet_combined_selected_stereo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an interactive, offline HTML report showing which FragNet "
            "fragments and fragment connections increase predicted P."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fragnet-dir", type=Path, default=Path("smiles_baseline2/FragNet"))
    parser.add_argument("--config", type=Path, help="Override the YAML configuration path.")
    parser.add_argument("--checkpoint", type=Path, help="Override the ft.pt checkpoint path.")
    parser.add_argument("--split", default="test", help="CSV split name: test, val, or train.")
    parser.add_argument("--target", default="log_P_upconversion")
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--rank-by",
        choices=("observed", "predicted"),
        default="observed",
        help="Select molecules by observed target or by model prediction.",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=None,
        help="Exclude molecules whose absolute prediction error exceeds this value.",
    )
    parser.add_argument(
        "--property-selector",
        default="Solubility",
        help=(
            "FragNet's internal regression-code selector. Its default does NOT "
            "change the target: predictions always use the supplied checkpoint."
        ),
    )
    parser.add_argument("--fragmentation", default="brics")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a clearly labelled synthetic report to test the renderer.",
    )
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    if args.max_error is not None and args.max_error < 0:
        parser.error("--max-error must be non-negative")
    return args


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def svg_data_url(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def image_data_url(image: Any) -> str | None:
    if image is None:
        return None
    if isinstance(image, str) and "<svg" in image:
        return svg_data_url(image)
    if hasattr(image, "save"):
        output = io.BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    return None


def parse_connection(value: Any) -> tuple[int, int] | None:
    try:
        decoded = ast.literal_eval(value) if isinstance(value, str) else value
        if len(decoded) != 2:
            return None
        return int(decoded[0]), int(decoded[1])
    except (TypeError, ValueError, SyntaxError, IndexError):
        return None


def fragment_motif(mol: Any, atom_ids: list[int], chem: Any) -> str:
    valid_atoms = [
        int(index)
        for index in atom_ids
        if 0 <= int(index) < mol.GetNumAtoms()
        and mol.GetAtomWithIdx(int(index)).GetAtomicNum() > 1
    ]
    if not valid_atoms:
        valid_atoms = [int(index) for index in atom_ids if 0 <= int(index) < mol.GetNumAtoms()]
    if not valid_atoms:
        return "[unknown fragment]"
    try:
        return chem.MolFragmentToSmiles(mol, atomsToUse=valid_atoms, canonical=True)
    except Exception:
        symbols = [mol.GetAtomWithIdx(index).GetSymbol() for index in valid_atoms]
        return "-".join(symbols) or "[unknown fragment]"


def motif_image(motif: str, chem: Any, drawer_module: Any) -> str | None:
    try:
        molecule = chem.MolFromSmiles(motif)
        if molecule is None:
            return None
        drawer = drawer_module.MolDraw2DSVG(440, 260)
        options = drawer.drawOptions()
        options.padding = 0.18
        drawer_module.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        return svg_data_url(drawer.GetDrawingText())
    except Exception:
        return None


def signed_molecule_image(
    smiles: str,
    fragments: list[dict[str, Any]],
    chem: Any,
    drawer_module: Any,
) -> str | None:
    try:
        molecule = chem.MolFromSmiles(smiles)
        if molecule is None:
            return None

        drawer = drawer_module.MolDraw2DSVG(520, 300)
        options = drawer.drawOptions()
        options.padding = 0.12
        drawer_module.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        return svg_data_url(drawer.GetDrawingText())
    except Exception:
        return None


def choose_columns(frame: Any, args: argparse.Namespace) -> tuple[str, str]:
    if args.smiles_column:
        smiles_column = args.smiles_column
    else:
        candidates = {str(column).lower(): str(column) for column in frame.columns}
        smiles_column = next(
            (candidates[key] for key in ("smiles", "canonical_smiles", "smile") if key in candidates),
            "",
        )
    if not smiles_column or smiles_column not in frame.columns:
        raise ValueError(
            "Could not find a SMILES column. Available columns: "
            f"{list(frame.columns)}. Specify --smiles-column COLUMN."
        )
    target_column = args.target
    if target_column == "log_P_upconversion" and target_column not in frame.columns and "y" in frame.columns:
        target_column = "y"
    if target_column not in frame.columns:
        raise ValueError(
            f"Target column {args.target!r} was not found. "
            f"Available columns: {list(frame.columns)}. Specify --target COLUMN."
        )
    return smiles_column, target_column


def build_real_dataset(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    fragnet_paths = [args.fragnet_dir.expanduser().resolve(),
                     (root.parent / args.fragnet_dir).expanduser().resolve()]
    for candidate in fragnet_paths:
        if (candidate / "fragnet").is_dir():
            sys.path.insert(0, str(candidate))
            break
    try:
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        from fragnet.vizualize.model_attr import get_attr_image
        from fragnet.vizualize.viz import FragNetVizApp, get_atoms_in_frags, get_frags
    except ImportError as error:
        raise SystemExit(
            f"Missing dependency: {error}\n"
            "Activate the Python environment used to train FragNet, for example:\n"
            "    source ~/.env/fragnet/bin/activate\n"
            "Then run this script again from the FragNet repository if necessary."
        ) from error

    config = (args.config or root / "fragnet_selected.yaml").expanduser().resolve()
    checkpoint = (args.checkpoint or root / "experiment" / "ft.pt").expanduser().resolve()
    split = root / "splits" / f"{args.split}.csv"

    missing = [str(path) for path in (config, checkpoint, split) if not path.is_file()]
    if missing:
        raise SystemExit("Required FragNet run files were not found:\n  " + "\n  ".join(missing))

    frame = pd.read_csv(split)
    smiles_column, target_column = choose_columns(frame, args)
    frame = frame.dropna(subset=[smiles_column, target_column]).copy()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=[target_column])
    if frame.empty:
        raise SystemExit(f"No usable molecules were found in {split}.")

    print(f"Loading trained checkpoint: {checkpoint}", flush=True)
    try:
        viz = FragNetVizApp(str(config), str(checkpoint))
    except Exception as error:
        raise SystemExit(
            f"Could not load the model through FragNetVizApp: {error}\n"
            "Check that fragnet_selected.yaml describes the architecture used for ft.pt "
            "and that you are using the same FragNet checkout as during training."
        ) from error

    prediction_cache: dict[str, float] = {}
    failures: list[str] = []
    if args.rank_by == "predicted":
        print(f"Ranking {len(frame)} molecules by predicted {target_column} ...", flush=True)
        predictions = []
        for smiles in frame[smiles_column].astype(str):
            try:
                prediction = float(viz.calc_weights(smiles))
                prediction_cache[smiles] = prediction
                predictions.append(prediction)
            except Exception as error:
                failures.append(f"Prediction failed for {smiles}: {error}")
                predictions.append(float("nan"))
        frame["_fragnet_prediction"] = predictions
        ranked = frame.dropna(subset=["_fragnet_prediction"]).sort_values(
            "_fragnet_prediction", ascending=False
        )
    else:
        ranked = frame.sort_values(target_column, ascending=False)

    molecules = []
    supports_signed_connections = hasattr(viz, "calc_fbond_contributions")
    for _, row in ranked.iterrows():
        if len(molecules) >= args.top_n:
            break
        smiles = str(row[smiles_column]).strip()
        actual = float(row[target_column])
        print(f"Explaining molecule {len(molecules) + 1}/{args.top_n}: {smiles}", flush=True)

        try:
            # calc_weights also populates viz.mol and viz.data_item; call it even
            # if a prediction was cached during predicted-value ranking.
            prediction = float(viz.calc_weights(smiles))
            prediction_cache[smiles] = prediction
            error_value = abs(prediction - actual)
            if args.max_error is not None and error_value > args.max_error:
                print(f"  Skipping: absolute error {error_value:.3f} exceeds threshold.", flush=True)
                continue

            fragment_attention_image = None
            fragment_attention = {}
            attention_connections = {}
            try:
                fragment_png, _, weight_frame, connection_frame, atoms_in_frags = (
                    viz.frag_weight_highlight()
                )
                fragment_attention_image = image_data_url(fragment_png)
                for _, weight_row in weight_frame.iterrows():
                    fragment_attention[int(weight_row["fragment"])] = number(weight_row["weight"])
                for _, connection_row in connection_frame.iterrows():
                    pair = parse_connection(connection_row["connection"])
                    if pair is not None:
                        attention_connections[tuple(sorted(pair))] = number(connection_row["weight"])
            except Exception as attention_error:
                graph, _ = get_frags(smiles)
                atoms_in_frags = get_atoms_in_frags(graph)
                try:
                    values = viz.summed_attn_weights_frags.sum(1).detach().cpu().tolist()
                    fragment_attention.update({index: number(value) for index, value in enumerate(values)})
                except Exception:
                    pass
                failures.append(f"Attention rendering failed for {smiles}: {attention_error}")

            # "Solubility" selects FragNet's existing *generic fine-tuned
            # regression path*. The actual model and target are determined by
            # config/checkpoint, not by this legacy selector label.
            attribution_image, _, raw_contributions = get_attr_image(
                smiles,
                str(config),
                str(checkpoint),
                args.property_selector,
                frag_type=args.fragmentation,
            )

            fragment_map = {}
            for raw_id, raw_atoms in atoms_in_frags.items():
                fragment_id = int(raw_id)
                atom_ids = [int(value) for value in raw_atoms]
                fragment_map[fragment_id] = {
                    "id": fragment_id,
                    "atoms": atom_ids,
                    "motif": fragment_motif(viz.mol, atom_ids, Chem),
                    "attention": fragment_attention.get(fragment_id),
                    "contribution": None,
                }

            for contribution in raw_contributions:
                fragment_id = int(contribution["fragment_index"])
                atom_ids = [int(value) for value in contribution.get("atoms", [])]
                entry = fragment_map.setdefault(
                    fragment_id,
                    {
                        "id": fragment_id,
                        "atoms": atom_ids,
                        "motif": fragment_motif(viz.mol, atom_ids, Chem),
                        "attention": fragment_attention.get(fragment_id),
                        "contribution": None,
                    },
                )
                entry["contribution"] = number(contribution.get("contribution"))

            fragments = sorted(
                fragment_map.values(),
                key=lambda item: item["contribution"] if item["contribution"] is not None else -math.inf,
                reverse=True,
            )
            for fragment in fragments:
                fragment["image"] = motif_image(fragment["motif"], Chem, rdMolDraw2D)

            signed_connections = {}
            connection_warning = None
            if supports_signed_connections:
                try:
                    connection_frame = viz.calc_fbond_contributions(
                        viz.data_item, args.property_selector
                    )
                    for _, connection_row in connection_frame.iterrows():
                        pair = tuple(
                            sorted((int(connection_row["begin_index"]), int(connection_row["end_index"])))
                        )
                        signed_connections[pair] = number(connection_row["attr"])
                except Exception as connection_error:
                    connection_warning = str(connection_error)
                    failures.append(f"Signed connection masking failed for {smiles}: {connection_error}")

            connections = []
            for begin, end in sorted(set(attention_connections) | set(signed_connections)):
                left = fragment_map.get(begin, {}).get("motif", f"fragment {begin}")
                right = fragment_map.get(end, {}).get("motif", f"fragment {end}")
                connections.append(
                    {
                        "begin": begin,
                        "end": end,
                        "left": left,
                        "right": right,
                        "label": " ↔ ".join(sorted((left, right))),
                        "attention": attention_connections.get((begin, end)),
                        "contribution": signed_connections.get((begin, end)),
                    }
                )
            connections.sort(
                key=lambda item: (
                    item["contribution"] is not None,
                    item["contribution"] if item["contribution"] is not None else item["attention"] or 0,
                ),
                reverse=True,
            )

            molecules.append(
                {
                    "smiles": smiles,
                    "actual": actual,
                    "predicted": prediction,
                    "error": error_value,
                    "image": signed_molecule_image(smiles, fragments, Chem, rdMolDraw2D)
                    or image_data_url(attribution_image),
                    "attention_image": fragment_attention_image,
                    "fragments": fragments,
                    "connections": connections,
                    "connection_warning": connection_warning,
                }
            )
        except Exception as error:
            failures.append(f"Explanation failed for {smiles}: {error}")
            print(f"  Skipping molecule after explanation error: {error}", file=sys.stderr, flush=True)

    if not molecules:
        preview = "\n  ".join(failures[:5]) or "No molecules met the filtering criteria."
        raise SystemExit(
            "No molecules could be explained. Check your checkpoint/configuration and "
            "consider removing --max-error. First issues:\n  " + preview
        )

    return {
        "title": "FragNet high-P structure–property report",
        "target": target_column,
        "generated_at": dt.datetime.now().strftime("%d %b %Y, %H:%M"),
        "checkpoint": str(checkpoint),
        "split": str(split),
        "rank_by": args.rank_by,
        "max_error": args.max_error,
        "dataset_size": int(len(frame)),
        "demo": False,
        "molecules": molecules,
        "warnings": failures[:30],
        "signed_connections_available": any(
            connection.get("contribution") is not None
            for molecule in molecules
            for connection in molecule["connections"]
        ),
    }


def demo_svg(label: str, positive: bool = True) -> str:
    color = "#aaa"
    markup = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="300" '
        'viewBox="0 0 520 300">'
        '<rect width="520" height="300" fill="#ffffff"/>'
        '<path d="M110 150 L155 110 L205 130 L210 180 L165 205 L115 185 Z '
        'M205 130 L275 112 L325 152 L375 124 M325 152 L350 206" '
        'fill="none" stroke="#334155" stroke-width="7" stroke-linejoin="round"/>'
        f'<circle cx="275" cy="112" r="30" fill="{color}" fill-opacity=".43"/>'
        '<text x="268" y="121" font-size="26" font-family="Arial" fill="#0f172a">N</text>'
        '</svg>'
    )
    return svg_data_url(markup)


def build_demo_dataset(args: argparse.Namespace) -> dict[str, Any]:
    definitions = [
        ("Nc1ccc(S)cc1", 3.82, 3.71, [("N", 0.88), ("c1ccccc1", 0.33), ("S", 0.61)]),
        ("Nc1ccc(O)cc1", 3.47, 3.42, [("N", 0.75), ("c1ccccc1", 0.28), ("O", -0.14)]),
        ("CNc1ccncc1", 3.31, 3.39, [("CN", 0.59), ("c1ccncc1", 0.48)]),
        ("Nc1ccc(N)cc1", 3.08, 2.99, [("N", 0.71), ("c1ccccc1", 0.22), ("N", 0.45)]),
        ("Sc1ccncc1", 2.92, 2.88, [("S", 0.55), ("c1ccncc1", 0.31)]),
    ]
    molecules = []
    for smiles, actual, predicted, items in definitions[: args.top_n]:
        fragments = [
            {
                "id": index,
                "atoms": [index, index + 1],
                "motif": motif,
                "attention": 1.8 + contribution,
                "contribution": contribution,
                "image": demo_svg(motif, positive=contribution >= 0),
            }
            for index, (motif, contribution) in enumerate(items)
        ]
        connections = [
            {
                "begin": index,
                "end": index + 1,
                "left": items[index][0],
                "right": items[index + 1][0],
                "label": " ↔ ".join(sorted((items[index][0], items[index + 1][0]))),
                "attention": 2.3 + index * 0.2,
                "contribution": 0.41 - index * 0.17,
            }
            for index in range(len(items) - 1)
        ]
        molecules.append(
            {
                "smiles": smiles,
                "actual": actual,
                "predicted": predicted,
                "error": abs(actual - predicted),
                "image": demo_svg(smiles),
                "attention_image": None,
                "fragments": fragments,
                "connections": connections,
                "connection_warning": None,
            }
        )
    return {
        "title": "FragNet high-P structure–property report",
        "target": args.target,
        "generated_at": dt.datetime.now().strftime("%d %b %Y, %H:%M"),
        "checkpoint": "SYNTHETIC DEMONSTRATION — not a trained checkpoint",
        "split": "SYNTHETIC DEMONSTRATION — not real model predictions",
        "rank_by": "observed",
        "max_error": None,
        "dataset_size": len(definitions),
        "demo": True,
        "molecules": molecules,
        "warnings": [],
        "signed_connections_available": True,
    }


def summarize(dataset: dict[str, Any]) -> None:
    grouped_fragments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_connections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for molecule_index, molecule in enumerate(dataset["molecules"]):
        for fragment in molecule["fragments"]:
            grouped_fragments[fragment["motif"]].append({**fragment, "molecule": molecule_index})
        for connection in molecule["connections"]:
            grouped_connections[connection["label"]].append({**connection, "molecule": molecule_index})

    def compile_group(label: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        values = [entry["contribution"] for entry in entries if entry.get("contribution") is not None]
        attentions = [entry["attention"] for entry in entries if entry.get("attention") is not None]
        return {
            "label": label,
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "positive_fraction": sum(value > 0 for value in values) / len(values) if values else None,
            "support": len({entry["molecule"] for entry in entries}),
            "occurrences": len(entries),
            "attention": statistics.fmean(attentions) if attentions else None,
            "image": next((entry.get("image") for entry in entries if entry.get("image")), None),
            "molecules": sorted({entry["molecule"] for entry in entries}),
        }

    def ranking(item: dict[str, Any]) -> tuple[bool, float, int]:
        return (item["mean"] is not None, item["mean"] or item["attention"] or 0, item["support"])

    dataset["fragment_summary"] = sorted(
        (compile_group(label, entries) for label, entries in grouped_fragments.items()),
        key=ranking,
        reverse=True,
    )
    fragment_lookup = {}
    for index, fragment in enumerate(dataset["fragment_summary"], start=1):
        fragment["display_name"] = f"Fragment {index}"
        fragment_lookup[fragment["label"]] = fragment

    for molecule in dataset["molecules"]:
        for fragment in molecule["fragments"]:
            fragment["display_name"] = fragment_lookup[fragment["motif"]]["display_name"]
        for connection in molecule["connections"]:
            left = fragment_lookup.get(connection["left"], {})
            right = fragment_lookup.get(connection["right"], {})
            connection["left_name"] = left.get("display_name", f"Fragment {connection['begin'] + 1}")
            connection["right_name"] = right.get("display_name", f"Fragment {connection['end'] + 1}")
            connection["left_image"] = left.get("image")
            connection["right_image"] = right.get("image")
            connection["display_name"] = f"{connection['left_name']} ↔ {connection['right_name']}"

    dataset["connection_summary"] = sorted(
        (compile_group(label, entries) for label, entries in grouped_connections.items()),
        key=ranking,
        reverse=True,
    )
    for connection in dataset["connection_summary"]:
        example = grouped_connections[connection["label"]][0]
        left = fragment_lookup.get(example["left"], {})
        right = fragment_lookup.get(example["right"], {})
        connection["left_name"] = left.get("display_name", "Fragment")
        connection["right_name"] = right.get("display_name", "Fragment")
        connection["left_image"] = left.get("image")
        connection["right_image"] = right.get("image")
        connection["display_name"] = f"{connection['left_name']} ↔ {connection['right_name']}"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FragNet · High-P fragment explorer</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; color: #222; background: white; font: 15px/1.5 Arial, sans-serif; }
    .wrap { max-width: 1150px; margin: auto; padding: 24px; }
    h1 { font-size: 28px; margin: 0 0 10px; }
    h2 { font-size: 19px; margin: 0 0 8px; }
    .eyebrow, .label, .caption, .section-name, .sub, .legend { color: #555; }
    .section-name { margin: 18px 0 8px; font-weight: bold; }
    .formula, .legend, .banner { margin: 12px 0; }
    .stats, .toolbar, .tabs, .filters, .chips, .connection-pair, .connection-chip-images, .metric-row {
      display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
    }
    .stats { margin: 20px 0; }
    .stat { min-width: 150px; }
    .stat .value { font-size: 20px; font-weight: bold; }
    .panel, .card { border: 1px solid #bbb; padding: 16px; margin: 14px 0; }
    .toolbar { justify-content: space-between; margin: 18px 0; }
    button, input, select { color: #222; background: white; border: 1px solid #999; padding: 6px; font: inherit; }
    button { cursor: pointer; }
    button.active { border: 2px solid #222; font-weight: bold; }
    .filter-input { min-width: 190px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 9px; text-align: left; vertical-align: middle; border-bottom: 1px solid #ccc; }
    .motif-cell, .connection-pair, .connection-unit { display: flex; gap: 10px; align-items: center; }
    .connection-unit { flex-direction: column; }
    .thumb { width: 180px; height: 110px; object-fit: contain; }
    .connection-thumb { width: 135px; height: 90px; object-fit: contain; }
    .connector { font-size: 20px; }
    .fragment-name, .signed { font-weight: bold; }
    .chip { display: inline-flex; flex-direction: column; gap: 4px; align-items: center; padding: 8px; border: 1px solid #ddd; }
    .fragment-chip img { width: 145px; height: 95px; object-fit: contain; }
    .connection-chip img { width: 105px; height: 75px; object-fit: contain; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .card { margin: 0; }
    .card-id, .panel-head { display: flex; justify-content: space-between; }
    .structure { display: block; width: 100%; height: 260px; object-fit: contain; }
    .metric-row { margin: 10px 0; }
    .metric strong { display: block; }
    .path { overflow-wrap: anywhere; }
    .empty { padding: 20px; }
    .hidden { display: none !important; }
    @media (max-width: 750px) { .grid { grid-template-columns: 1fr; } .wrap { padding: 14px; } }
  </style>
</head>
<body><main class="wrap">
  <div id="demo-banner" class="banner hidden"><strong>Synthetic demonstration.</strong> These values are not real model predictions. Run the generator without <code>--demo</code> to analyse your checkpoint.</div>
  <header>
    <div class="eyebrow">FragNet · structure–property interpretation</div>
    <h1>What drives high P?</h1>
    <div class="sub">Find recurring molecular fragments and fragment–fragment connections that increase the model’s predicted <span id="target-name"></span>. Rankings use signed masking contributions, not attention alone.</div>
    <div class="formula">Contribution = prediction<sub>full</sub> − prediction<sub>masked</sub> <strong>positive → increases P</strong></div>
  </header>
  <section class="stats" id="stats"></section>
  <div id="connection-note" class="banner hidden">This FragNet checkout did not return signed fragment-connection contributions. Connection rows are labelled <strong>attention only</strong>; they must not be interpreted as increasing or decreasing P.</div>
  <div class="toolbar">
    <div class="tabs">
      <button class="active" data-tab="fragments">Fragments</button>
      <button data-tab="connections">Fragment connections</button>
      <button data-tab="molecules">Individual molecules</button>
    </div>
    <div class="filters">
      <input class="filter-input" id="search" placeholder="Search fragment number…">
      <select id="support" aria-label="Minimum molecule support"><option value="1">Support ≥ 1</option><option value="2">Support ≥ 2</option><option value="3">Support ≥ 3</option><option value="5">Support ≥ 5</option><option value="10">Support ≥ 10</option></select>
      <label class="toggle"><input type="checkbox" id="positive" checked> Positive only</label>
    </div>
  </div>
  <section id="fragments-pane" class="panel"></section>
  <section id="connections-pane" class="panel hidden"></section>
  <section id="molecules-pane" class="grid hidden"></section>
  <div class="legend">Positive contributions indicate fragments predicted to increase P; negative contributions indicate a decrease. Attention-only values describe model focus, without a direction.</div>
  <details id="metadata"><summary>Model, data source and diagnostics</summary><div id="metadata-body"></div></details>
</main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const state = {tab:'fragments', search:'', support:1, positive:true};
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const fmt = value => value == null ? '—' : Number(value).toFixed(3);
  const signed = value => value == null ? '<span class="neutral">—</span>' : `<span class="signed ${value >= 0 ? 'pos':'neg'}">${value > 0 ? '+':''}${fmt(value)}</span>`;
  const badge = value => value == null ? '<span class="pill neutral">attention only</span>' : value > 0 ? '<span class="pill pos">increases P</span>' : value < 0 ? '<span class="pill neg">decreases P</span>' : '<span class="pill neutral">neutral</span>';
  const pct = value => value == null ? '—' : `${Math.round(value * 100)}%`;
  const searchMatch = text => String(text).toLowerCase().includes(state.search);

  $('target-name').textContent = data.target;
  if (data.demo) $('demo-banner').classList.remove('hidden');
  if (!data.signed_connections_available) $('connection-note').classList.remove('hidden');
  const strongest = data.fragment_summary.find(row => row.mean != null && row.mean > 0);
  const bestPrediction = Math.max(...data.molecules.map(molecule => molecule.predicted));
  $('stats').innerHTML = [
    ['Molecules explained', data.molecules.length],
    ['Highest predicted P', fmt(bestPrediction)],
    ['Recurring fragment types', data.fragment_summary.length],
    ['Strongest fragment', strongest ? `${escape(strongest.display_name)} <span class="pos">+${fmt(strongest.mean)}</span>` : '—'],
  ].map(([label,value]) => `<article class="stat"><div class="label">${label}</div><div class="value">${value}</div></article>`).join('');

  function summaryRows(rows) {
    return rows.filter(row => row.support >= state.support && searchMatch(row.display_name) &&
      (!state.positive || row.mean == null || row.mean > 0));
  }

  function renderSummary(kind) {
    const isConnection = kind === 'connections';
    const rows = summaryRows(isConnection ? data.connection_summary : data.fragment_summary);
    const heading = isConnection ? 'Fragment combinations associated with high P' : 'Fragments associated with high P';
    const table = rows.length ? `<div class="table-wrap"><table><thead><tr>
      <th>${isConnection ? 'Fragment combination' : 'Fragment motif'}</th><th>Mean contribution</th>
      <th>Median</th><th>Direction</th><th>Positive fraction</th><th>Molecules</th><th>Attention</th>
      </tr></thead><tbody>${rows.map(row => `<tr><td><div class="motif-cell">
      ${isConnection ? `<div class="connection-pair"><div class="connection-unit">${row.left_image ? `<img class="connection-thumb" src="${row.left_image}" alt="${escape(row.left_name)} structure">` : ''}<span class="fragment-name">${escape(row.left_name)}</span></div><span class="connector">↔</span><div class="connection-unit">${row.right_image ? `<img class="connection-thumb" src="${row.right_image}" alt="${escape(row.right_name)} structure">` : ''}<span class="fragment-name">${escape(row.right_name)}</span></div></div>` : `${row.image ? `<img class="thumb" src="${row.image}" alt="${escape(row.display_name)} structure">` : ''}<span class="fragment-name">${escape(row.display_name)}</span>`}
      </div></td><td>${signed(row.mean)}</td>
      <td>${signed(row.median)}</td><td>${badge(row.mean)}</td><td><div>${pct(row.positive_fraction)}</div>
      </td>
      <td>${row.support} <span class="label">/ ${data.molecules.length}</span></td><td>${fmt(row.attention)}</td></tr>`).join('')}</tbody></table></div>` :
      '<div class="empty">No entries match these filters. Try reducing the support threshold or switching off “Positive only”.</div>';
    $(kind + '-pane').innerHTML = `<div class="panel-head"><div><h2>${heading}</h2><div class="caption">${rows.length} displayed · support counts distinct molecules</div></div></div>${table}`;
  }

  function moleculeMatches(molecule) {
    if (!state.search) return true;
    return molecule.fragments.some(fragment => searchMatch(fragment.display_name)) ||
      molecule.connections.some(connection => searchMatch(connection.display_name));
  }

  function renderMolecules() {
    const rows = data.molecules.filter(moleculeMatches);
    $('molecules-pane').innerHTML = rows.length ? rows.map((molecule,index) => {
      const fragments = molecule.fragments.filter(fragment => !state.positive || (fragment.contribution ?? 0) > 0).slice(0,6);
      const connections = molecule.connections.filter(connection => !state.positive || connection.contribution == null || connection.contribution > 0).slice(0,5);
      return `<article class="card"><div class="card-top"><div class="card-id"><span class="eyebrow">High-P molecule ${index + 1}</span><span class="pill ${molecule.error <= 0.3 ? 'pos':'neutral'}">error ${fmt(molecule.error)}</span></div></div>
      ${molecule.image ? `<img class="structure" src="${molecule.image}" alt="Molecule structure">` : '<div class="empty">Structure image unavailable</div>'}
      <div class="card-content"><div class="metric-row">
      <div class="metric"><span class="label">Observed P</span><strong>${fmt(molecule.actual)}</strong></div>
      <div class="metric"><span class="label">Predicted P</span><strong>${fmt(molecule.predicted)}</strong></div>
      <div class="metric"><span class="label">Fragments</span><strong>${molecule.fragments.length}</strong></div></div>
      <div class="section-name">${state.positive ? 'Property-increasing' : 'Signed'} fragments</div>
      <div class="chips">${fragments.length ? fragments.map(fragment => `<span class="chip fragment-chip">${fragment.image ? `<img src="${fragment.image}" alt="${escape(fragment.display_name)} structure">` : ''}<span class="fragment-name">${escape(fragment.display_name)}</span>${signed(fragment.contribution)}</span>`).join('') : '<span class="caption">No positive fragments identified.</span>'}</div>
      <div class="section-name">Fragment connections</div><div class="chips">${connections.length ? connections.map(connection => `<span class="chip connection-chip"><span class="connection-chip-images">${connection.left_image ? `<img src="${connection.left_image}" alt="${escape(connection.left_name)} structure">` : ''}<span class="connector">↔</span>${connection.right_image ? `<img src="${connection.right_image}" alt="${escape(connection.right_name)} structure">` : ''}</span><span class="fragment-name">${escape(connection.display_name)}</span>${connection.contribution == null ? '<span class="neutral">attention only</span>' : signed(connection.contribution)}</span>`).join('') : '<span class="caption">No matching fragment connections.</span>'}</div>
      ${molecule.connection_warning ? `<div class="caption" style="margin-top:10px">Signed connection masking unavailable for this molecule.</div>` : ''}</div></article>`;
    }).join('') : '<div class="empty">No molecules match this search.</div>';
  }

  function render() {
    renderSummary('fragments'); renderSummary('connections'); renderMolecules();
    ['fragments','connections','molecules'].forEach(tab => {
      $(tab + '-pane').classList.toggle('hidden', state.tab !== tab);
      document.querySelector(`[data-tab="${tab}"]`).classList.toggle('active', state.tab === tab);
    });
  }

  document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => {state.tab = button.dataset.tab; render();}));
  $('search').addEventListener('input', event => {state.search = event.target.value.toLowerCase().trim(); render();});
  $('support').addEventListener('change', event => {state.support = Number(event.target.value); render();});
  $('positive').addEventListener('change', event => {state.positive = event.target.checked; render();});
  $('metadata-body').innerHTML = `<p><strong>Generated:</strong> ${escape(data.generated_at)}<br><strong>Selection:</strong> highest ${escape(data.rank_by)} ${escape(data.target)}<br><strong>Source molecules:</strong> ${data.dataset_size}<br><strong>Maximum prediction error:</strong> ${data.max_error == null ? 'not filtered' : escape(data.max_error)}</p><p><strong>Checkpoint:</strong><br><span class="path">${escape(data.checkpoint)}</span></p><p><strong>Input split:</strong><br><span class="path">${escape(data.split)}</span></p>${data.warnings.length ? `<p><strong>Non-fatal warnings:</strong> ${data.warnings.length}</p>`:''}`;
  render();
})();
</script>
</body>
</html>
"""


def render_html(dataset: dict[str, Any]) -> str:
    serialized = json.dumps(dataset, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Prevent user-controlled SMILES or paths from terminating the JSON script.
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return HTML_TEMPLATE.replace("__REPORT_DATA__", serialized)


def main() -> int:
    args = parse_args()
    dataset = build_demo_dataset(args) if args.demo else build_real_dataset(args)
    summarize(dataset)
    if args.output is not None:
        destination = args.output.expanduser().resolve()
    elif args.demo:
        destination = Path.cwd() / "fragnet_fragment_demo.html"
    else:
        destination = args.root.expanduser().resolve() / "explainability" / "fragnet_fragment_report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html(dataset), encoding="utf-8")

    positive_fragments = sum(
        1 for fragment in dataset["fragment_summary"] if (fragment.get("mean") or 0) > 0
    )
    positive_connections = sum(
        1 for connection in dataset["connection_summary"] if (connection.get("mean") or 0) > 0
    )
    print(f"Saved {destination}")
    print(f"Molecules: {len(dataset['molecules'])}; positive fragments: {positive_fragments}; "
          f"positive connections: {positive_connections}")
    if not dataset["signed_connections_available"]:
        print("Connection contributions unavailable; attention-only values are labelled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

