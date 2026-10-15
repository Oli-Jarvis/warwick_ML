#!/usr/bin/env python3
"""Map scored molecules onto the fixed dim4/analyse4 reference without refitting."""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import warnings

ROOT = Path(__file__).resolve().parent


def load_reference(directory):
    import numpy as np
    from packaging.version import Version
    from rdkit import rdBase
    import sklearn

    with (directory / "reference.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("format") != "dim4_analyse4_v1":
        raise ValueError("Use the reference exported by the revised dim4.py and analyse4.py")
    versions = dict(
        rdkit=rdBase.rdkitVersion, sklearn=sklearn.__version__, dscribe=importlib.metadata.version("dscribe")
    )
    if {k: Version(v) for k, v in versions.items()} != {
        k: Version(v) for k, v in bundle["versions"].items()
    }:
        raise RuntimeError(f'Use the reference environment: expected {bundle["versions"]}, found {versions}')
    expected = dict(soap_r_cut=5.0, soap_n_max=4, soap_l_max=3, soap_average="inner")
    metadata = bundle["pca_provenance"]["descriptor_metadata"]
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Unsupported reference descriptor setting: {key}")
    if bundle["property"] != "log_P_upconversion" or bundle["assignment"] != "nearest_birch_subcluster":
        raise ValueError("Unexpected reference property or cluster assignment rule")
    assigned = bundle["subcluster_labels"][bundle["birch"].predict(bundle["reference_matrix"])]
    if not np.array_equal(assigned, bundle["reference"].cluster.to_numpy()):
        raise ValueError("Saved model does not reproduce the reference cluster assignments")
    # Randomized PCA fit_transform and transform can differ; retain that provenance.
    discrepancy = bundle["pca_provenance"]["transform_max_abs_difference"]
    for name, offset in [("bonding", 0), ("structural", bundle["components"])]:
        error = np.asarray(discrepancy[name])[: bundle["components"]]
        if (error / np.asarray(bundle["scales"])[offset : offset + len(error)] > 0.01).any():
            warnings.warn(
                f"{name} PCA transform differs from saved reference scores by over 1% of the block scale"
            )
    return bundle


def read_candidates(path, property_column, keep_duplicates=False):
    import numpy as np
    import pandas as pd

    with path.open(encoding="utf-8-sig") as handle:
        header = handle.readline()
    separator = csv.Sniffer().sniff(header, delimiters=",;\t").delimiter
    frame = pd.read_csv(path, sep=separator, encoding="utf-8-sig", low_memory=False)
    frame.columns = frame.columns.str.strip()
    if not {"smiles", property_column}.issubset(frame):
        raise ValueError(f"Input needs smiles and {property_column}")
    frame["_row"] = np.arange(len(frame))
    frame["log_P_upconversion"] = pd.to_numeric(frame[property_column], errors="coerce")
    good = np.isfinite(frame.log_P_upconversion)
    if "prediction_ok" in frame:
        good &= frame.prediction_ok.astype(str).str.lower().isin(["true", "1", "1.0"])
    errors = [
        dict(row=int(row["_row"]), smiles=row["smiles"], error="Failed prediction or non-finite property")
        for _, row in frame.loc[~good].iterrows()
    ]
    accepted = frame.loc[good].copy()
    duplicates = 0
    if not keep_duplicates:
        duplicates = int(accepted.duplicated("smiles").sum())
        accepted = accepted.drop_duplicates("smiles", keep="first")
    return accepted.reset_index(drop=True), errors, len(frame), duplicates


def describe(frame, output, bundle, save_conformers=False):
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from ase import Atoms
    from dscribe.descriptors import SOAP

    molecules, systems, selected, errors = [], [], [], []
    with (Chem.SDWriter(str(output / "generated_conformers.sdf")) if save_conformers
          else contextlib.nullcontext()) as writer:
        for index, row in frame.iterrows():
            try:
                if not isinstance(row.smiles, str) or not row.smiles.strip():
                    raise ValueError("Missing/non-text SMILES")
                # Match dim4: remove literal [Au], then ETKDGv3 and MMFF/UFF.
                smiles = row.smiles.strip().replace("[Au]", "")
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or not mol.GetNumAtoms():
                    raise ValueError("Invalid chemical-space SMILES")
                mol = Chem.AddHs(mol)
                symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                if not set(symbols).issubset(bundle["species"]):
                    raise ValueError("Element outside reference SOAP species")
                params = AllChem.ETKDGv3()
                params.randomSeed = int(bundle["random_state"])
                if AllChem.EmbedMolecule(mol, params) != 0:
                    raise ValueError("Conformer generation failed")
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol)
                else:
                    AllChem.UFFOptimizeMolecule(mol)
                system = Atoms(
                    symbols=symbols, positions=np.asarray(mol.GetConformer().GetPositions(), dtype=float)
                )
            except (ValueError, RuntimeError, TypeError) as exc:
                errors.append(dict(row=int(row["_row"]), smiles=row.smiles, error=str(exc)))
                continue
            mol.SetIntProp("source_row", int(row["_row"]))
            if writer is not None:
                writer.write(mol)
            molecules.append(mol)
            systems.append(system)
            selected.append(index)
    if not selected:
        return None, errors
    clean = frame.loc[selected].copy().reset_index(drop=True)
    clean["chemplot_smiles"] = clean.smiles.str.strip().str.replace("[Au]", "", regex=False)
    soap = SOAP(
        species=bundle["species"], periodic=False, r_cut=5.0, n_max=4, l_max=3, average="inner", sparse=False
    )
    structural = np.asarray(soap.create(systems), dtype=np.float32)
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(bundle["descriptor_names"])
    bonding = np.asarray([calculator.CalcDescriptors(Chem.RemoveHs(mol)) for mol in molecules], dtype=float)
    bonding[~np.isfinite(bonding)] = np.nan
    rows, columns = np.where(np.isnan(bonding))
    bonding[rows, columns] = np.asarray(bundle["medians"])[columns]
    bonding = bonding[:, np.asarray(bundle["descriptor_mask"], dtype=bool)]
    if not np.isfinite(structural).all() or not np.isfinite(bonding).all():
        raise ValueError("Non-finite candidate descriptors")
    return (clean, bonding, structural), errors


def transform_coordinates(bundle, bonding, structural, prop):
    import numpy as np
    from sklearn.metrics import pairwise_distances_argmin_min

    pcs = []
    for name, values in [("bonding", bonding), ("structural", structural)]:
        fit = bundle[name]
        pcs.append(fit["pca"].transform(fit["scaler"].transform(values))[:, : bundle["components"]])
    coords = np.column_stack([*pcs, prop])
    matrix = (coords - bundle["mean"]) / bundle["scales"]
    subclusters = bundle["birch"].predict(matrix)
    distance = np.linalg.norm(matrix - bundle["centres"][subclusters], axis=1)
    nearest, ref_distance = pairwise_distances_argmin_min(matrix, bundle["reference_matrix"])
    return coords, bundle["subcluster_labels"][subclusters], distance, nearest, ref_distance


def save_plot(reference_dir, candidates, output):
    import plotly.graph_objects as go

    # Reuse the original analyse4 figure, including cluster colours and axes.
    text = (reference_dir / "reference.html").read_text()
    start = text.rfind("Plotly.newPlot(")
    if start < 0:
        raise ValueError("Reference HTML has no Plotly figure")
    decoder = json.JSONDecoder()
    start = text.index("[", start)
    data, length = decoder.raw_decode(text[start:])
    start = text.index("{", start + length)
    layout, _ = decoder.raw_decode(text[start:])
    figure = go.Figure(data=data, layout=layout)
    figure.add_trace(
        go.Scatter3d(
            x=candidates.Bonding_PC_1,
            y=candidates.Structural_PC_1,
            z=candidates.log_P_upconversion,
            mode="markers",
            name="FragNet candidates",
            customdata=candidates[["smiles", "cluster"]].to_numpy(),
            hovertemplate="SMILES=%{customdata[0]}<br>Cluster=%{customdata[1]}<br>Predicted log P=%{z}<extra></extra>",
            marker=dict(size=7, color="red", symbol="diamond"),
        )
    )
    figure.write_html(str(output), include_plotlyjs=True)



def save_cluster_report(bundle, candidates, output, representatives=20):
    import html
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    reference = bundle["reference"].reset_index(drop=True)
    count = bundle["components"]
    columns = ([f"Bonding_PC_{i+1}" for i in range(count)] +
               [f"Structural_PC_{i+1}" for i in range(count)] + ["log_P_upconversion"])
    if not {"smiles", "cluster", *columns}.issubset(candidates):
        raise ValueError("Report input must be the mapped candidates_in_space.csv")
    matrix = (candidates[columns].to_numpy(float) - bundle["mean"]) / bundle["scales"]
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite mapped coordinates")
    expected = bundle["subcluster_labels"][bundle["birch"].predict(matrix)]
    if not np.array_equal(expected, candidates.cluster.to_numpy()):
        raise ValueError("Candidate cluster labels disagree with this reference")

    def gallery(frame, distances, source):
        cards = []
        for (_, row), distance in zip(frame.iterrows(), distances):
            mol = Chem.MolFromSmiles(str(row.smiles))
            if mol is None:
                drawing = "<p>Drawing unavailable</p>"
            else:
                drawer = rdMolDraw2D.MolDraw2DSVG(280, 190)
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                drawer.FinishDrawing()
                drawing = drawer.GetDrawingText()
            sa = f"<br>SA: {row['SA']:.3g}" if 'SA' in row and np.isfinite(row['SA']) else ""
            cards.append(f'<article>{drawing}<p>{source} log P: {row.log_P_upconversion:.4g}'
                         f'<br>Centroid distance: {distance:.4g}{sa}</p><details><summary>SMILES</summary>'
                         f'<code>{html.escape(str(row.smiles))}</code></details></article>')
        return '<div class="gallery">' + ''.join(cards) + '</div>' if cards else '<p>None assigned.</p>'

    sections = []
    for label, group in reference.groupby("cluster", sort=True):
        centroid = bundle["reference_matrix"][group.index].mean(axis=0)
        distance = np.linalg.norm(bundle["reference_matrix"][group.index] - centroid, axis=1)
        group = group.assign(_centroid_distance=distance)
        selected = group.drop_duplicates("smiles").sort_values("_centroid_distance", kind="stable").head(representatives)
        mask = candidates.cluster.to_numpy() == label
        generated = candidates.loc[mask].copy()
        generated["_centroid_distance"] = np.linalg.norm(matrix[mask] - centroid, axis=1)
        generated = generated.drop_duplicates("smiles").sort_values("_centroid_distance", kind="stable")
        sections.append(f'<section><h2>Cluster {html.escape(str(label))}</h2>'
                        f'<p>{len(group):,} reference rows; {group.smiles.nunique():,} unique reference SMILES; '
                        f'{len(generated):,} unique generated SMILES.</p>'
                        '<h3>Representative reference molecules</h3>' +
                        gallery(selected, selected._centroid_distance, "Reference xTB") +
                        '<details><summary>Assigned generated molecules</summary>' +
                        gallery(generated, generated._centroid_distance, "FragNet predicted") + '</details></section>')
    output.write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Cluster representatives and generated molecules</title>
<style>body{font:15px/1.5 Arial,sans-serif;color:#222;max-width:1250px;margin:30px auto;padding:0 18px}
section{border-top:1px solid #aaa;margin-top:28px;padding-top:12px}.gallery{display:flex;flex-wrap:wrap;gap:16px}
article{width:280px;border:1px solid #ddd;padding:10px}svg{max-width:100%;height:auto}
h1{font-size:25px}h2{font-size:21px}h3{font-size:17px}summary{cursor:pointer;padding:8px 0}
code{overflow-wrap:anywhere;font-size:12px}</style>
<h1>Cluster representatives and generated molecules</h1>
''' + f'<p>Up to {representatives} unique reference molecules nearest each final cluster centroid are shown. '
                      'Distances use the saved scaled clustering space, including the property. '
                      'Centroids use reference molecules only. Generated molecules retain their nearest-BIRCH-subcluster '
                      'assignments and are displayed by distance to the reference cluster centroid, not by performance. '
                      'Reference properties are xTB-derived; generated properties are FragNet predictions. '
                      'These examples describe cluster centres, not every structural motif within a cluster.</p>' +
                      ''.join(sections) + '</html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "chemical_space_reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--property-column", default="fragnet_log_P_upconversion")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--save-conformers", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Build cluster report from an existing candidates_in_space.csv")
    parser.add_argument("--representatives", type=int, default=20)
    args = parser.parse_args()
    if args.representatives < 1:
        parser.error("--representatives must be positive")
    if not args.report_only and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    import pandas as pd

    bundle = load_reference(args.reference)
    if args.report_only:
        report = args.output / "cluster_report.html"
        if report.exists():
            raise FileExistsError(f"Report already exists: {report}")
        summary_path = args.input.parent / "projection_summary.json"
        if summary_path.exists():
            recorded = json.loads(summary_path.read_text()).get("reference_sha256")
            actual = hashlib.sha256((args.reference / "reference.pkl").read_bytes()).hexdigest()
            if recorded and recorded != actual:
                raise ValueError("Reference differs from the original projection")
        args.output.mkdir(parents=True, exist_ok=True)
        save_cluster_report(bundle, pd.read_csv(args.input), report, args.representatives)
        print(f"Report: {report}")
        return
    frame, errors, total, duplicates = read_candidates(args.input, args.property_column, args.keep_duplicates)
    args.output.mkdir(parents=True, exist_ok=True)
    described, descriptor_errors = describe(frame, args.output, bundle, args.save_conformers)
    errors.extend(descriptor_errors)
    if errors:
        pd.DataFrame(errors, columns=["row", "smiles", "error"]).to_csv(
            args.output / "projection_rejected.csv", index=False
        )
    if described is None:
        raise ValueError("No candidates could be projected; see projection_rejected.csv")
    clean, bonding, structural = described
    coords, labels, distances, nearest, ref_distances = transform_coordinates(
        bundle, bonding, structural, clean.log_P_upconversion.to_numpy(float)
    )
    result = clean.rename(columns={"_row": "source_row"}).copy()
    count = bundle["components"]
    for j in range(count):
        result[f"Bonding_PC_{j+1}"] = coords[:, j]
        result[f"Structural_PC_{j+1}"] = coords[:, count + j]
    result["PCA_1"] = coords[:, 0]
    result["PCA_2"] = coords[:, count]
    result["cluster"] = labels
    result["distance_to_birch_subcluster"] = distances
    result["nearest_reference_distance"] = ref_distances
    result["nearest_reference_smiles"] = bundle["reference"].iloc[nearest].smiles.to_numpy()
    result["nearest_reference_log_P_upconversion"] = (
        bundle["reference"].iloc[nearest].log_P_upconversion.to_numpy()
    )
    result.to_csv(args.output / "candidates_in_space.csv", index=False)
    save_plot(args.reference, result, args.output / "candidates_in_space.html")
    save_cluster_report(bundle, result, args.output / "cluster_report.html", args.representatives)
    summary = dict(
        input=str(args.input.resolve()),
        input_rows=total,
        duplicate_rows_omitted=duplicates,
        rejected_rows=len(errors),
        projected_molecules=len(result),
        reference=str(args.reference.resolve()),
        original_clusters_preserved=True,
        reference_sha256=hashlib.sha256((args.reference / "reference.pkl").read_bytes()).hexdigest(),
        components_per_descriptor=count,
        property_weight=bundle["weight"],
        candidate_assignment=bundle["assignment"],
        candidate_descriptors_recalculated=True,
        transform_max_abs_difference=bundle["pca_provenance"]["transform_max_abs_difference"],
    )
    (args.output / "projection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
