#!/usr/bin/env python3
"""Explain generated FragNet predictions using the generation model and graphs."""
import argparse
import contextlib
import csv
import html
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'fragnet_log_P_upconversion'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, nargs='+',
                   help='One or more fragnet_predictions.csv files, or full_history_generated.csv')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--root', type=Path, default=ROOT / 'fragnet_config_details')
    p.add_argument('--fragnet-dir', type=Path, default=ROOT / 'fragnet_source_code')
    p.add_argument('--top-n', type=int, default=0, help='Maximum molecules per 20%% tail; default 0 analyses all')
    p.add_argument('--select', choices=['both', 'high', 'low'], default='both')
    p.add_argument('--min-support', type=int, default=2, help='Distinct molecules per summary motif')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--collect-only', action='store_true', help='Build shared CSV without running explanations')
    args = p.parse_args()
    if args.top_n < 0 or args.min_support < 1 or args.threads < 1:
        p.error('Require top-n >= 0, min-support >= 1 and threads >= 1')
    if args.top_n == 0 and args.select != 'both':
        p.error('--select high/low requires a positive --top-n')
    return args


def read_candidates(path, identity):
    accepted, excluded, seen = [], [], {}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'smiles', TARGET, 'prediction_ok'} <= set(reader.fieldnames or []):
            raise ValueError(f'Input must contain smiles, {TARGET} and prediction_ok')
        for line, row in enumerate(reader, 2):
            try:
                if row['prediction_ok'].strip().lower() not in {'true', '1', '1.0'}:
                    raise ValueError('Prediction failed: ' + row.get('prediction_error', ''))
                value = float(row[TARGET])
                if not math.isfinite(value):
                    raise ValueError('Non-finite prediction')
                smiles = identity(row['smiles'])[0]
            except (ValueError, TypeError) as exc:
                excluded.append(dict(line=line, smiles=row.get('smiles', ''), reason=str(exc)))
                continue
            if smiles in seen:
                if not math.isclose(value, seen[smiles], rel_tol=1e-5, abs_tol=1e-4):
                    raise ValueError(f'Conflicting predictions for the same molecule at line {line}')
                excluded.append(dict(line=line, smiles=smiles, reason='Duplicate canonical isomeric SMILES'))
                continue
            seen[smiles] = value
            accepted.append(dict(smiles=smiles, saved_prediction=value, source_line=line))
    if not accepted:
        raise ValueError('No successful finite predictions in input')
    return sorted(accepted, key=lambda r: (r['saved_prediction'], r['smiles'])), excluded


def select_candidates(rows, top_n, selection):
    ranked = sorted(rows, key=lambda r: (r['saved_prediction'], r['smiles']))
    n = len(ranked) // 5
    labelled = [dict(r, cohort=('low' if i < n else
                               'high' if n and i >= len(ranked)-n else 'middle'))
                for i, r in enumerate(ranked)]
    if top_n == 0:
        return labelled
    if n == 0:
        raise ValueError('At least five unique molecules are required for 20% tail selection')
    low = labelled[:min(n, top_n)]
    high = labelled[-min(n, top_n):][::-1]
    return high + low if selection == 'both' else high if selection == 'high' else low


class CaptureCreator:
    """Retain the exact molecule used by the shared graph builder for drawings."""
    def __init__(self, creator):
        self.creator = creator

    def __getattr__(self, name):
        return getattr(self.creator, name)

    def create_data_point(self, args):
        self.mol = args[2]
        return self.creator.create_data_point(args)


def evaluate(predictor, batch):
    with predictor.torch.inference_mode():
        result = predictor.model(batch).reshape(-1)
    if result.numel() != 1 or not predictor.torch.isfinite(result).all():
        raise ValueError('Expected one finite model prediction')
    return float(result.item())



def evaluate_with_embedding(predictor, batch):
    vectors = []
    def capture(module, inputs):
        vectors.extend(inputs[0].detach().cpu().tolist())
    hook = predictor.model.fthead.register_forward_pre_hook(capture)
    try:
        prediction = evaluate(predictor, batch)
    finally:
        hook.remove()
    if len(vectors) != 1 or not vectors[0] or not all(math.isfinite(x) for x in vectors[0]):
        raise ValueError('Expected one finite pooled molecular embedding at the regression-head input')
    return prediction, vectors[0]


def fragment_prediction(predictor, batch, atom_ids):
    # Match the existing attribution: mask final atom states before pooling.
    def mask(module, inputs, outputs):
        atoms = outputs[0].clone()
        atoms[atom_ids] = 0
        return (atoms, *outputs[1:])
    hook = predictor.model.pretrain.register_forward_hook(mask)
    try:
        return evaluate(predictor, batch)
    finally:
        hook.remove()


def connection_prediction(predictor, batch, pair_index):
    layers = list(predictor.model.pretrain.layers)
    if not layers or any(not hasattr(layer, 'frag_bond_mask') for layer in layers):
        raise RuntimeError('This model does not support signed fragment-connection masking')
    previous = [layer.frag_bond_mask for layer in layers]
    try:
        for layer in layers:
            layer.frag_bond_mask = pair_index
        return evaluate(predictor, batch)
    finally:
        for layer, value in zip(layers, previous):
            layer.frag_bond_mask = value


def connection_pairs(batch):
    edges = batch['frag_index'].t().tolist()
    if batch['x_frags'].shape[0] == 1:
        return []  # The single-fragment self connection is not a chemical bond.
    if len(edges) % 2 or len(edges) != batch['node_features_fbonds'].shape[0]:
        raise ValueError('Unexpected fragment-connection layout; cannot safely mask')
    pairs = []
    for i in range(0, len(edges), 2):
        a, b = edges[i]
        if edges[i + 1] != [b, a] or a == b:
            raise ValueError('Expected consecutive forward/reverse fragment connections')
        pairs.append((i // 2, a, b))
    return pairs


def motif(mol, atoms, chem):
    selected = set(atoms)
    cuts = [b.GetIdx() for b in mol.GetBonds()
            if (b.GetBeginAtomIdx() in selected) != (b.GetEndAtomIdx() in selected)]
    cut = chem.FragmentOnBonds(mol, cuts, dummyLabels=[(0, 0)] * len(cuts)) if cuts else mol
    for group in chem.GetMolFrags(cut):
        originals = {i for i in group if i < mol.GetNumAtoms()}
        if originals == selected:
            text = chem.MolFragmentToSmiles(cut, atomsToUse=list(group), canonical=True,
                                            isomericSmiles=True)
            fragment = chem.MolFromSmiles(text)
            if fragment is None:
                raise ValueError('Could not encode attachment-aware motif')
            return chem.MolToSmiles(chem.RemoveHs(fragment), isomericSmiles=True)
    raise ValueError('Fragment atoms do not form one connected motif')


def drawing(mol, atoms=(), bonds=(), contribution=0):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    picture = Chem.Mol(mol)
    # Map graph indices through hydrogen removal for uncluttered drawings.
    for atom in picture.GetAtoms():
        atom.SetIntProp('graph_id', atom.GetIdx())
    endpoints = [(mol.GetBondWithIdx(i).GetBeginAtomIdx(), mol.GetBondWithIdx(i).GetEndAtomIdx()) for i in bonds]
    picture = Chem.RemoveHs(picture)
    mapping = {a.GetIntProp('graph_id'): a.GetIdx() for a in picture.GetAtoms()}
    atoms = [mapping[i] for i in atoms if i in mapping]
    bonds = [picture.GetBondBetweenAtoms(mapping[a], mapping[b]).GetIdx()
             for a, b in endpoints if a in mapping and b in mapping]
    picture.RemoveAllConformers()
    rdDepictor.Compute2DCoords(picture)
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 250)
    color = ((0.75, 0.75, 0.75) if abs(contribution) < 1e-6 else
             (0.85, 0.45, 0.30) if contribution > 0 else (0.35, 0.60, 0.85))
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, picture, highlightAtoms=list(atoms), highlightBonds=list(bonds),
        highlightAtomColors={i: color for i in atoms}, highlightBondColors={i: color for i in bonds})
    drawer.FinishDrawing()
    return drawer.GetDrawingText().split('</svg>')[0] + '</svg>'


def explain(predictor, row):
    from rdkit import Chem
    capture = CaptureCreator(predictor.creator)
    graph = predictor.training.one_graph(SimpleNamespace(smiles=row['smiles'], y=0.), capture, predictor.seed)
    batch = predictor.collate([graph])
    full, embedding = evaluate_with_embedding(predictor, batch)
    if not math.isclose(full, row['saved_prediction'], rel_tol=1e-5, abs_tol=1e-4):
        raise ValueError(f"Prediction mismatch for {row['smiles']}: saved={row['saved_prediction']}, rebuilt={full}")
    mol = capture.mol
    ids = batch['atom_to_frag_ids'].tolist()
    if len(ids) != mol.GetNumAtoms():
        raise ValueError('Graph/drawing atom indices differ')
    groups = {f: [i for i, v in enumerate(ids) if v == f] for f in sorted(set(ids))}
    entries = []
    for f, atoms in groups.items():
        masked = fragment_prediction(predictor, batch, atoms)
        delta = full - masked
        entries.append(dict(kind='fragment', local_id=str(f), motif=motif(mol, atoms, Chem),
                            masked_prediction=masked, contribution=delta,
                            image=drawing(mol, atoms, contribution=delta)))
    for index, a, b in connection_pairs(batch):
        bonds = [bond.GetIdx() for bond in mol.GetBonds()
                 if {ids[bond.GetBeginAtomIdx()], ids[bond.GetEndAtomIdx()]} == {a, b}]
        if not bonds:
            raise ValueError('Fragment connection has no corresponding chemical bond')
        masked = connection_prediction(predictor, batch, index)
        delta = full - masked
        entries.append(dict(kind='connection', local_id=f'{a}:{b}',
                            motif=motif(mol, groups[a] + groups[b], Chem),
                            masked_prediction=masked, contribution=delta,
                            image=drawing(mol, bonds=bonds, contribution=delta)))
    restored = evaluate(predictor, batch)
    if not math.isclose(restored, full, rel_tol=1e-7, abs_tol=1e-6):
        raise RuntimeError('Masking changed the unmasked prediction')
    return dict(row, prediction=full, embedding=embedding, parity_difference=abs(full-row['saved_prediction']),
                image=drawing(mol), entries=entries)


def summarise(molecules, min_support):
    groups = defaultdict(lambda: defaultdict(list))
    cohorts = {}
    for molecule in molecules:
        cohorts[molecule['smiles']] = molecule['cohort']
        for entry in molecule['entries']:
            groups[(entry['kind'], entry['motif'])][molecule['smiles']].append(entry['contribution'])
    results = []
    for (kind, key), by_molecule in sorted(groups.items()):
        values = {s: statistics.mean(v) for s, v in by_molecule.items()}
        if len(values) < min_support:
            continue
        row = dict(kind=kind, motif=key, molecules=len(values),
                   occurrences=sum(map(len, by_molecule.values())),
                   mean=statistics.mean(values.values()), median=statistics.median(values.values()),
                   sd=statistics.stdev(values.values()) if len(values)>1 else 0.,
                   positive_fraction=sum(v>0 for v in values.values())/len(values))
        for cohort in ['high', 'low', 'middle']:
            tail = [v for s, v in values.items() if cohorts[s] == cohort]
            row[cohort+'_n'] = len(tail)
            row[cohort+'_mean'] = statistics.mean(tail) if tail else None
        results.append(row)
    return sorted(results, key=lambda r: abs(r['mean']), reverse=True)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def render_html(molecules, summary, metadata):
    escape = html.escape
    def fmt(value):
        return 'not sampled' if value is None else f'{value:.3f}'
    sections = []
    for kind in ['fragment', 'connection']:
        rows = []
        for entry in [r for r in summary if r['kind'] == kind]:
            example_m = next(m for m in molecules if any(e['kind']==kind and e['motif']==entry['motif'] for e in m['entries']))
            example = next(e for e in example_m['entries'] if e['kind']==kind and e['motif']==entry['motif'])
            rows.append(f"<tr><td>{example['image']}<details><summary>Motif SMILES</summary><code>{escape(entry['motif'])}</code></details></td>"
                        f"<td>{entry['molecules']}</td><td>{fmt(entry['mean'])}</td><td>{fmt(entry['sd'])}</td>"
                        f"<td>{entry['positive_fraction']:.0%}</td><td>{fmt(entry['high_mean'])} ({entry['high_n']})</td>"
                        f"<td>{fmt(entry['low_mean'])} ({entry['low_n']})</td></tr>")
        sections.append(f'<h2>Recurring {kind}s</h2><div class="scroll"><table><thead><tr>'
                        '<th>Example in molecular context</th><th>Molecules</th><th>Mean Δ</th><th>SD</th>'
                        '<th>Positive</th><th>High mean (n)</th><th>Low mean (n)</th></tr></thead><tbody>'+
                        ''.join(rows)+'</tbody></table></div>' if rows else
                        f'<h2>Recurring {kind}s</h2><p>No motifs meet the support threshold.</p>')
    details = []
    for i, m in enumerate(molecules, 1):
        entries = ''.join(f"<tr><td>{e['kind']} {escape(e['local_id'])}</td><td>{e['image']}</td>"
                          f"<td>{fmt(e['contribution'])}</td><td>{fmt(e['masked_prediction'])}</td></tr>"
                          for e in sorted(m['entries'], key=lambda e: abs(e['contribution']), reverse=True))
        details.append(f"<details><summary>Molecule {i} · {m['cohort']} · predicted log P {fmt(m['prediction'])}</summary>"
                       f"{m['image']}<p><code>{escape(m['smiles'])}</code></p><table><thead><tr>"
                       '<th>Component</th><th>Highlighted context</th><th>Δ</th><th>Masked prediction</th>'
                       f'</tr></thead><tbody>{entries}</tbody></table></details>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FragNet fragment analysis</title><style>
body{font:16px/1.5 Georgia,serif;color:#222;background:white;max-width:1150px;margin:32px auto;padding:0 20px}
h1,h2,summary,th{font-family:Arial,sans-serif}h1{font-size:26px}h2{font-size:20px;margin-top:32px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #ddd;padding:10px;text-align:left;vertical-align:middle}
th{background:#f4f4f4}svg{max-width:100%;height:auto;width:340px}code{overflow-wrap:anywhere;font-size:12px}
.scroll{overflow:auto}details{border-top:1px solid #ccc;padding:12px 0}summary{cursor:pointer}small{color:#555}
@media print{details{break-inside:avoid}body{max-width:none}}
</style><h1>FragNet fragment analysis</h1>'''+f"<p>{len(molecules)} unique molecules; target: log_P_upconversion. Selection: {escape(metadata['selection'])}.</p>"+'''
<p>Δ = prediction(full) − prediction(masked), in log-property units. Orange supports a higher prediction;
blue supports a lower prediction; grey is near zero. Summary drawings show one example occurrence,
whose sign may differ from the mean across molecules.</p>
<p>Fragment masking zeros final atom states before pooling; fragment states remain. Connection masking zeros
both directed connection states in every layer. Neither operation deletes atoms or breaks a chemical bond.
These are model sensitivities, not additive contributions, causal effects or confidence estimates.</p>
<p>Summary means give each distinct molecule equal weight after averaging repeated occurrences within it.
SD describes variation between molecules; Positive is the fraction with mean Δ &gt; 0.
High/low groups are the top/bottom 20% ranked by predicted log P, rounded down to whole molecules.
Equal predictions are ordered by canonical SMILES. The middle group contributes to overall statistics;
--top-n optionally samples the extremes within these 20% groups. Unlisted motifs may have insufficient support.
Connection motifs describe the joined fragment pair and its remaining attachment sites.</p>
<p>embeddings.csv contains the native pooled molecular vectors passed to the regression head, without PCA or normalisation.</p>'''+''.join(sections)+'<h2>Individual molecules</h2>'+''.join(details)+f'''
<h2>Run details</h2><small><pre style="white-space:pre-wrap;overflow-wrap:anywhere">{escape(json.dumps(metadata, indent=2))}</pre></small></html>'''



def collect_predictions(paths, destination, checkpoint):
    fields = ['source_file', 'source_line', 'checkpoint_sha256', 'oracle_call',
              'encoding', 'smiles', TARGET, 'prediction_ok', 'prediction_error']
    records = []
    for path in dict.fromkeys(p.resolve() for p in paths):
        summary = path.parent / 'run_summary.json'
        recorded = json.loads(summary.read_text()).get('checkpoint_sha256') if summary.exists() else None
        if recorded and recorded != checkpoint:
            raise ValueError(f'Checkpoint differs from generation run: {path}')
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'smiles', TARGET, 'prediction_ok'} <= set(reader.fieldnames or []):
                raise ValueError(f'Not a FragNet prediction history: {path}')
            for line, row in enumerate(reader, 2):
                source_hash = row.get('checkpoint_sha256') or recorded or ''
                if source_hash and source_hash != checkpoint:
                    raise ValueError(f'Mixed predictor checkpoints in {path}, line {line}')
                records.append({**row, 'source_file': row.get('source_file') or str(path),
                                'source_line': row.get('source_line') or line,
                                'checkpoint_sha256': source_hash})
    if not records:
        raise ValueError('Input histories are empty')
    write_csv(destination, records, fields)
    return len(records)


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new directory')
    sys.path.insert(0, str(ROOT))
    from generate_molecules import Predictor, digest
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        predictor = Predictor(args.root, args.fragnet_dir, args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    shared = args.output / 'full_history_generated.csv'
    collected = collect_predictions(args.input, shared, predictor.checkpoint_hash)
    if args.collect_only:
        print(f'Collected {collected} evaluations: {shared}')
        return
    rows, excluded = read_candidates(shared, predictor.training.molecule_identity)
    selected = select_candidates(rows, args.top_n, args.select)
    molecules = []
    for i, row in enumerate(selected, 1):
        print(f"Explaining {i}/{len(selected)} ({row['cohort']})", flush=True)
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            molecules.append(explain(predictor, row))
    dimensions = {len(m['embedding']) for m in molecules}
    if len(dimensions) != 1:
        raise ValueError('Inconsistent molecular embedding dimensions')
    embedding_dimension = dimensions.pop()
    summary = summarise(molecules, args.min_support)
    metadata = dict(inputs={str(p.resolve()): digest(p) for p in args.input},
                    collected_evaluations=collected, shared_history_sha256=digest(shared),
                    checkpoint_sha256=predictor.checkpoint_hash, config_sha256=digest(args.root/'fragnet_selected.yaml'),
                    script_sha256=digest(Path(__file__)), predictor_script_sha256=digest(ROOT/'generate_molecules.py'),
                    training_helper_sha256=digest(Path(predictor.training.__file__)), graph_seed=predictor.seed,
                    selection=args.select if args.top_n else 'all; top/bottom 20% comparison', requested_per_tail=args.top_n,
                    cohort_rule='rank by saved prediction, then canonical isomeric SMILES; floor(N/5) per tail',
                    tail_fraction=0.2, population_tail_count=len(rows)//5,
                    population_cohort_counts=dict(high=len(rows)//5, low=len(rows)//5,
                                                  middle=len(rows)-2*(len(rows)//5)),
                    embedding_dimension=embedding_dimension,
                    embedding_location='native pooled atom/fragment concatenation at fthead input',
                    embedding_transform=None,
                    accepted_unique=len(rows), explained=len(molecules), excluded_rows=len(excluded),
                    selected_ranges={c: [min(m['prediction'] for m in molecules if m['cohort']==c),
                                         max(m['prediction'] for m in molecules if m['cohort']==c)]
                                     for c in sorted({m['cohort'] for m in molecules})},
                    min_support=args.min_support, max_parity_difference=max(m['parity_difference'] for m in molecules),
                    parity_rtol=1e-5, parity_atol=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    flat = [dict(molecule_id=i, smiles=m['smiles'], cohort=m['cohort'], prediction=m['prediction'],
                 **{k:v for k,v in e.items() if k!='image'})
            for i,m in enumerate(molecules,1) for e in m['entries']]
    write_csv(args.output/'contributions.csv', flat,
              ['molecule_id','smiles','cohort','prediction','kind','local_id','motif','masked_prediction','contribution'])
    embedding_columns = [f'embedding_{i}' for i in range(embedding_dimension)]
    write_csv(args.output/'embeddings.csv',
              [dict(smiles=m['smiles'], cohort=m['cohort'], prediction=m['prediction'],
                    **dict(zip(embedding_columns, m['embedding']))) for m in molecules],
              ['smiles','cohort','prediction',*embedding_columns])
    fields = ['kind','motif','molecules','occurrences','mean','median','sd','positive_fraction',
              'high_n','high_mean','low_n','low_mean','middle_n','middle_mean']
    write_csv(args.output/'motif_summary.csv', summary, fields)
    if excluded:
        write_csv(args.output/'excluded.csv', excluded, ['line','smiles','reason'])
    (args.output/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (args.output/'report.html').write_text(render_html(molecules, summary, metadata))
    print(f"Report: {args.output/'report.html'}")


if __name__ == '__main__':
    main()
