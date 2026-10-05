#!/usr/bin/env python3
"""CLI: verify, fit-reference, predict, or project an existing prediction table."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from predictor import Predictor, ROOT


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig') as f: header = f.readline()
    sep = csv.Sniffer().sniff(header, delimiters=',;\t').delimiter
    frame = pd.read_csv(path, sep=sep, encoding='utf-8-sig', low_memory=False)
    frame.columns = frame.columns.str.strip()
    return frame


def reference_rows(inputs, out):
    from training_reference import molecule_identity
    records, errors = [], []
    paths = set()
    for value in inputs:
        p = Path(value)
        if not p.exists(): raise FileNotFoundError(p)
        paths.update([p.resolve()] if p.is_file() else p.rglob('full_history.csv'))
    if not paths: raise ValueError('No reference CSVs found')
    for path in sorted(paths):
        frame = read_csv(path)
        if not {'smiles','log_P_upconversion'}.issubset(frame):
            raise ValueError(f'{path}: needs smiles and log_P_upconversion columns')
        for index, (s, value) in enumerate(frame[['smiles','log_P_upconversion']].itertuples(index=False, name=None), 2):
            try:
                y = float(value)
                if not np.isfinite(y): raise ValueError('Non-finite target')
                smi, _, _ = molecule_identity(s)
                records.append(dict(smiles=smi, log_P_upconversion=y, source_file=str(path)))
            except (ValueError, TypeError) as exc:
                errors.append(dict(source_file=str(path), row=index, error=str(exc)))
    if not records: raise ValueError('No usable reference molecules')
    frame = pd.DataFrame(records).drop_duplicates(['smiles','log_P_upconversion'])
    result = frame.groupby('smiles', sort=False, as_index=False).agg(
        log_P_upconversion=('log_P_upconversion','mean'),
        n_distinct_targets=('log_P_upconversion','size'))
    return result, errors, sorted(map(str, paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('verify','predict'):
        p = sub.add_parser(command)
        p.add_argument('--run-dir', type=Path, default=ROOT/'fragnet_combined_selected_stereo')
        p.add_argument('--fragnet-root', type=Path, default=ROOT/'smiles_baseline2/FragNet')
        p.add_argument('--threads', type=int, default=1)
        if command == 'verify': p.add_argument('--count', type=int, default=5)
        else:
            p.add_argument('--input', type=Path, required=True)
            p.add_argument('--smiles-column', default='smiles')
            p.add_argument('--output', type=Path, required=True)
            p.add_argument('--reference', type=Path)
    p = sub.add_parser('fit-reference')
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--components', type=int, default=3)
    p.add_argument('--weight', type=float, default=1.)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--clusters', type=int, default=20)
    p = sub.add_parser('fit-existing')
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--embedding-csv', type=Path, required=True)
    p.add_argument('--clustered-csv', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--components', type=int, default=3)
    p.add_argument('--weight', type=float, default=1.)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--clusters', type=int, default=20)
    p = sub.add_parser('project')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--property-column', default='fragnet_log_P_upconversion')
    args = parser.parse_args()
    if args.command == 'verify':
        if args.count < 1: parser.error('--count must be positive')
        model = Predictor(args.run_dir, args.fragnet_root, args.threads)
        print(json.dumps(model.verify(args.count), indent=2))
    elif args.command == 'fit-reference':
        from space import fit_reference
        frame, errors, paths = reference_rows(args.input, args.output)
        print(json.dumps(fit_reference(frame, args.output, args.components, args.weight, args.threshold, args.clusters), indent=2))
        pd.DataFrame(errors, columns=['source_file','row','error']).to_csv(args.output/'input_rejected.csv', index=False)
        (args.output/'inputs.json').write_text(json.dumps(paths, indent=2)+'\n')
    elif args.command == 'predict':
        if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Choose a new prediction output directory')
        frame = read_csv(args.input)
        if args.smiles_column not in frame: raise ValueError(f'Missing {args.smiles_column}')
        model = Predictor(args.run_dir, args.fragnet_root, args.threads)
        result = model.predict(frame[args.smiles_column].tolist())
        result.insert(0, 'input_row', np.arange(len(result))+2)
        if 'log_P_upconversion' in frame:
            result['xtb_log_P_upconversion'] = pd.to_numeric(frame.log_P_upconversion, errors='coerce').to_numpy()
            result['prediction_minus_xtb'] = result.fragnet_log_P_upconversion-result.xtb_log_P_upconversion
        args.output.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output/'predictions.csv', index=False)
        (args.output/'prediction_manifest.json').write_text(json.dumps(dict(checkpoint=str(model.checkpoint), checkpoint_sha256=model.checkpoint_hash,
            input=str(args.input.resolve()), successful=int(result.prediction_ok.sum()), total=len(result), target='log_P_upconversion; no extra logarithm'), indent=2)+'\n')
        print(f'Predicted {result.prediction_ok.sum()}/{len(result)} molecules')
        if args.reference:
            from space import project
            good = result.loc[result.prediction_ok].copy()
            good['log_P_upconversion'] = good.fragnet_log_P_upconversion
            if good.empty: raise ValueError('No successful predictions')
            project(good, args.reference, args.output)
    elif args.command == 'fit-existing':
        from space import fit_existing
        print(json.dumps(fit_existing(args.cache_dir,args.embedding_csv,args.output,
            args.components,args.weight,args.threshold,args.clusters,args.clustered_csv),indent=2))
    elif args.command == 'project':
        from space import project
        frame = read_csv(args.input)
        frame['log_P_upconversion'] = pd.to_numeric(frame[args.property_column], errors='coerce')
        good = frame[np.isfinite(frame.log_P_upconversion)].copy()
        if good.empty: raise ValueError('No finite candidate properties')
        project(good, args.reference, args.output)

if __name__ == '__main__':
    main()
