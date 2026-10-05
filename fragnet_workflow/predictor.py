"""Inference using the exact graph builder and model constructor supplied by Oli."""
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
import contextlib
import json
import os
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path('/storage/msszkb_grp/msshfg')

class Predictor:
    def __init__(self, run_dir=ROOT/'fragnet_combined_selected_stereo',
                 fragnet_root=ROOT/'smiles_baseline2/FragNet', threads=1):
        import torch
        self.torch = torch
        self.run_dir = Path(run_dir).resolve()
        self.fragnet_root = Path(fragnet_root).resolve()
        if not (self.fragnet_root/'fragnet').is_dir():
            raise FileNotFoundError(self.fragnet_root/'fragnet')
        sys.path.insert(0, str(self.fragnet_root))
        import training_reference as training
        from fragnet.dataset.data import CreateData, collate_fn
        self.training = training
        self.collate = collate_fn
        self.config = yaml.safe_load((self.run_dir/'fragnet_selected.yaml').read_text())
        self.seed = int(self.config['seed'])
        self.checkpoint = self.run_dir/'experiment/ft.pt'
        self.checkpoint_hash = training.digest(self.checkpoint)
        manifest_path = self.run_dir/'run_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            target = manifest['signature']['target_column']
            if target != 'log_P_upconversion':
                raise ValueError(f'Unexpected training target: {target}')
            # Match the feature/model sources to the recorded training environment.
            for old, expected in manifest['signature'].get('native_sources', {}).items():
                relative = Path(old).parts
                offset = relative.index('fragnet')
                path = self.fragnet_root.joinpath(*relative[offset:])
                if not path.exists() or training.digest(path) != expected:
                    raise RuntimeError(f'FragNet source differs from training: {path}')
        torch.set_num_threads(int(threads))
        training.native_feature_audit()
        self.model = training.build_model(self.config, self.config['finetune']['model'])
        state = training.unwrap_state_dict(training.torch_load(self.checkpoint, 'cpu', torch))
        self.model.load_state_dict(state, strict=True)
        self.model.to('cpu').eval()
        self.creator = CreateData(data_type='exp1s', create_bond_graph_data=True, add_dhangles=False)
        self.cache = OrderedDict()

    def predict(self, values, batch_size=128):
        rows = []
        pending = OrderedDict()
        for value in values:
            row = dict(input_smiles=value, smiles='', fragnet_log_P_upconversion=np.nan,
                       prediction_ok=False, prediction_error='')
            try:
                smi, _, _ = self.training.molecule_identity(value)
                row['smiles'] = smi
                if smi not in self.cache:
                    pending[smi] = None
            except (ValueError, TypeError) as exc:
                row['prediction_error'] = str(exc)
            rows.append(row)
        # Batches limit graph memory. Molecule failures remain aligned with inputs.
        names = list(pending)
        for start in range(0, len(names), batch_size):
            graphs, accepted = [], []
            for smi in names[start:start+batch_size]:
                try:
                    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                        graph = self.training.one_graph(SimpleNamespace(smiles=smi, y=0.), self.creator, self.seed)
                    graphs.append(graph)
                    accepted.append(smi)
                except Exception as exc:
                    self.cache[smi] = (np.nan, f'{type(exc).__name__}: {exc}')
            if graphs:
                with self.torch.inference_mode():
                    batch = {k:v.to('cpu') for k,v in self.collate(graphs).items()}
                    output = self.model(batch).reshape(-1).cpu().numpy()
                if len(output) != len(accepted) or not np.isfinite(output).all():
                    raise RuntimeError('Model returned invalid predictions; inference stopped')
                for smi, pred in zip(accepted, output):
                    self.cache[smi] = (float(pred), '')
        for row in rows:
            if row['smiles']:
                pred, error = self.cache[row['smiles']]
                row.update(fragnet_log_P_upconversion=pred, prediction_ok=not error,
                           prediction_error=error)
        # Keep the most recent 10,000 predictions between oracle batches.
        while len(self.cache) > 10000:
            self.cache.popitem(last=False)
        return pd.DataFrame(rows, columns=['input_smiles','smiles','fragnet_log_P_upconversion','prediction_ok','prediction_error'])

    def verify(self, count=5):
        """Check saved graphs AND regenerated graphs against existing train predictions."""
        import pickle
        path = self.run_dir/'experiment/train_predictions.csv'
        expected = pd.read_csv(path).head(count)
        with (self.run_dir/'graph_data/train.pkl').open('rb') as f:
            graphs = pickle.load(f)
        lookup = {g.smiles:g for g in graphs}
        subset = [lookup[s] for s in expected.smiles]
        with self.torch.inference_mode():
            actual = self.model(self.collate(subset)).reshape(-1).cpu().numpy()
        wanted = expected.predicted.to_numpy(float)
        if not np.allclose(actual, wanted, rtol=1e-5, atol=1e-4):
            raise RuntimeError(f'Saved graph prediction mismatch: max error {np.max(abs(actual-wanted))}')
        regenerated = self.predict(expected.smiles.tolist())
        if not regenerated.prediction_ok.all():
            raise RuntimeError(regenerated.to_string(index=False))
        fresh = regenerated.fragnet_log_P_upconversion.to_numpy(float)
        if not np.allclose(fresh, wanted, rtol=1e-5, atol=1e-4):
            raise RuntimeError(f'Regenerated graph prediction mismatch: max error {np.max(abs(fresh-wanted))}')
        return dict(n=len(wanted), saved_graph_max_difference=float(np.max(abs(actual-wanted))),
                    regenerated_graph_max_difference=float(np.max(abs(fresh-wanted))),
                    checkpoint_sha256=self.checkpoint_hash)
