"""Drop-in terahertz dispatcher; NMO still constructs its normal S-Au molecule."""
import csv
import hashlib
import os
from pathlib import Path
import numpy as np
from rdkit import Chem
from predictor import Predictor

_MODELS = {}

def dispatch(original, mols, encodings, config, metadata, rewards, indices, *args, **kwargs):
    if config.get('property_backend', 'xtb') != 'fragnet':
        return original(mols, encodings, config, metadata, rewards, indices, *args, **kwargs)
    key = (config['fragnet_run_dir'], config['fragnet_root'], int(config['fragnet_threads']))
    if key not in _MODELS:
        _MODELS[key] = Predictor(*key)
    model = _MODELS[key]
    n = len(encodings)
    indices = np.asarray(indices, dtype=int)
    smiles = [Chem.MolToSmiles(mols[i], canonical=True, isomericSmiles=True) for i in indices]
    result = model.predict(smiles)
    reasons = ['']*n
    logp = np.full(n, np.nan)
    ok = np.zeros(n, dtype=int)
    for index, row in zip(indices, result.itertuples(index=False)):
        reasons[index] = row.prediction_error
        if row.prediction_ok:
            logp[index] = row.fragnet_log_P_upconversion
            ok[index] = 1
    # A new ID for each evaluation avoids overwriting repeated candidates in HDF5.
    call_start = int(metadata.get('oracle_call_start', 0))
    rewards['hash_values'] = np.array([hashlib.md5(f'fragnet|{os.getpid()}|{call_start+i}|{e}'.encode()).hexdigest() for i,e in enumerate(encodings)])
    rewards['log_P_upconversion'] = logp
    rewards['fragnet_log_P_upconversion'] = logp.copy()
    rewards['fragnet_prediction_ok'] = ok
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        rewards['P_upconversion'] = np.power(10., logp)
    rewards['hl_gaps'] = np.full(n, np.nan)
    rewards['failure_reasons'] = np.asarray(reasons, dtype=object)
    logdir = Path(config['log_dir']); logdir.mkdir(parents=True, exist_ok=True)
    csv_path = logdir/f'fragnet_predictions_{os.getpid()}.csv'
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['oracle_call','encoding','smiles','fragnet_log_P_upconversion','prediction_ok','prediction_error','checkpoint_sha256'])
        if not exists: writer.writeheader()
        for index, row in zip(indices, result.itertuples(index=False)):
            writer.writerow(dict(oracle_call=call_start+index, encoding=encodings[index], smiles=row.smiles,
                fragnet_log_P_upconversion=row.fragnet_log_P_upconversion, prediction_ok=row.prediction_ok,
                prediction_error=row.prediction_error, checkpoint_sha256=model.checkpoint_hash))
    return rewards, indices[ok[indices].astype(bool)], reasons
