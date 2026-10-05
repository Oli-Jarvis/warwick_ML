#!/usr/bin/env python3
"""Step 3: place scored molecules in the saved chemical space."""
import configparser
import json
import os

from generator import FITNESS, HERE, call, environment, paths_for
from settings import parse_args


def project(args, paths):
    import csv
    source = paths['run']/'fragnet_predictions.csv'
    if not source.is_file():
        raise FileNotFoundError(source)
    if paths['projection'].exists() and any(paths['projection'].iterdir()):
        raise FileExistsError(f'Projection already exists: {paths["projection"]}')

    accepted = {}
    total = failures = 0
    with source.open(newline='') as f:
        for row in csv.DictReader(f):
            total += 1
            if row['prediction_ok'].lower() not in ('true', '1'):
                failures += 1
                continue
            accepted.setdefault(row['smiles'], row)
    if not accepted:
        raise RuntimeError('No successfully scored molecules to project. Inspect fragnet_predictions.csv.')
    input_csv = paths['run']/'unique_fragnet_candidates.csv'
    with input_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(accepted.values()))))
        writer.writeheader(); writer.writerows(accepted.values())
    env = environment(args)
    python = args.fragnet_python.expanduser().resolve()
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    env['CONDA_PREFIX'] = str(python.parent.parent)
    env['PATH'] = str(python.parent)+os.pathsep+env.get('PATH', '')
    call([python, HERE/'workflow.py', 'project', '--input', input_csv,
          '--reference', args.reference, '--output', paths['projection']],
         paths['work']/(args.name+'_projection.log'), HERE, env)
    with (paths['projection']/'candidates_in_space.csv').open(newline='') as f:
        projected = sum(1 for _ in csv.DictReader(f))
    summary = dict(oracle_evaluations=total, prediction_failures=failures,
                   unique_successful_molecules=len(accepted), projected_molecules=projected,
                   requested_oracle_calls=args.calls, fitness=FITNESS,
                   length_area_penalties=False, chemical_space_reference=str(args.reference),
                   note='Generated candidates may include known molecules. Novelty relative to training/reference was not established.')
    (paths['run']/'generation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Open {paths["projection"]/"candidates_in_space.html"}', flush=True)


def main(argv=None):
    args = parse_args('project', argv)
    paths = paths_for(args)
    config = configparser.ConfigParser(interpolation=None)
    if config.read(paths['run'] / 'nmo_fragnet.ini'):
        args.calls = config.getint('Oracle', 'max_oracle_calls')
    project(args, paths)


if __name__ == '__main__':
    main()
