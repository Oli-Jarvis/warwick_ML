#!/usr/bin/env python3
"""Compare baseline and refactored predictions/projections in the FragNet environment.

Run on an allocated compute node. This reads models and the saved reference;
it does not train, refit clusters, or overwrite earlier results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASELINE = 'b240b788c8024c46a46bf365dc9be984fafef440'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from settings import load_defaults
    import numpy as np
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--input', type=Path, help='Optional CSV with a smiles column')
    p.add_argument('--output', type=Path, required=True, help='New directory for both runs and comparison')
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    if min(args.count, args.threads) < 1:
        p.error('count and threads must be positive')
    defaults = load_defaults(args.config)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    source = (args.input or defaults['run_dir'] / 'experiment/train_predictions.csv').expanduser().resolve()
    sample = pd.read_csv(source).head(args.count)
    if sample.empty or 'smiles' not in sample:
        raise ValueError('Input needs at least one row and a smiles column')
    checkpoint = defaults['run_dir'] / 'experiment/ft.pt'
    reference = defaults['reference'] / 'reference.pkl'
    before = {str(path): digest(path) for path in (checkpoint, reference)}
    output.mkdir(parents=True)
    baseline_code = output / 'baseline_code'
    baseline_code.mkdir()
    for name in ('workflow.py', 'predictor.py', 'training_reference.py', 'space.py', 'legacy_dim.py'):
        content = subprocess.check_output(['git', 'show', f'{BASELINE}:fragnet_workflow/{name}'], cwd=ROOT)
        (baseline_code / name).write_bytes(content)
    # Include a duplicate and a rejected input to check row alignment and caching.
    smiles = sample.smiles.tolist() + [sample.smiles.iloc[0], 'not-a-smiles']
    input_csv = output / 'input.csv'
    pd.DataFrame({'smiles': smiles}).to_csv(input_csv, index=False)
    for label, code in (('baseline', baseline_code), ('refactored', HERE)):
        command = [sys.executable, str(code / 'workflow.py'), 'predict',
                   '--input', str(input_csv), '--output', str(output / label),
                   '--run-dir', str(defaults['run_dir']), '--fragnet-root', str(defaults['fragnet_root']),
                   '--reference', str(defaults['reference']), '--threads', str(args.threads)]
        with (output / f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=code, stdout=log, stderr=subprocess.STDOUT, check=True)
    compared = []
    for name in ('predictions.csv', 'candidates_in_space.csv', 'projection_rejected.csv'):
        old = pd.read_csv(output / 'baseline' / name)
        new = pd.read_csv(output / 'refactored' / name)
        prediction = name == 'predictions.csv'
        pd.testing.assert_frame_equal(old, new, check_dtype=False, check_exact=False,
                                      rtol=1e-5 if prediction else 1e-6,
                                      atol=1e-4 if prediction else 1e-6)
        compared.append(name)
    predictions = pd.read_csv(output / 'refactored/predictions.csv')
    if not predictions.prediction_ok.iloc[:-1].all() or predictions.prediction_ok.iloc[-1]:
        raise AssertionError('Expected valid input predictions followed by one rejected SMILES')
    saved_comparison = args.input is None
    if saved_comparison:
        np.testing.assert_allclose(predictions.fragnet_log_P_upconversion.iloc[:len(sample)],
                                   sample.predicted, rtol=1e-5, atol=1e-4)
    after = {str(path): digest(path) for path in (checkpoint, reference)}
    if before != after:
        raise AssertionError('Checkpoint or reference changed during validation')
    report = dict(passed=True, baseline_commit=BASELINE, source=str(source),
                  sample_count=len(sample), compared=compared,
                  matched_saved_training_predictions=saved_comparison,
                  duplicate_and_invalid_input_checked=True, assets_sha256=after,
                  generation_run_tested=False)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
