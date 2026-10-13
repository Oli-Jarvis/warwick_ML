#!/usr/bin/env python3
"""Generate molecules with the saved NMO prior and FragNet fitness.

Run generation in the nmo environment; predict/verify in the fragnet environment.
The persistent FragNet worker uses this same file in its own environment.
"""
from __future__ import annotations
import argparse
import ast
import atexit
import configparser
import contextlib
import csv
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import selectors
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = 'b240b788c8024c46a46bf365dc9be984fafef440'
# Existing positive log-P reward, including SA and rotatable-bond penalties.
FITNESS = ('0.25*(np.logaddexp(0.0,np.log(10.0)*'
           'np.clip(log_P_upconversion+5.0,-50.0,80.0))/np.log(10.0))'
           '*((10.0-SA)/9.0)**2/(1.0+np.exp(2.0*(N_rot-3.5)))')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def molecule_identity(text):
    if not isinstance(text,str) or not text.strip(): raise ValueError('Missing/non-text SMILES')
    mol=Chem.MolFromSmiles(text.strip())
    if mol is None or mol.GetNumAtoms()==0: raise ValueError('Invalid/empty SMILES')
    # Atom map labels are bookkeeping, not molecular identity.
    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    if mol.GetStereoGroups(): raise ValueError('Enhanced stereo groups require a richer representation')
    allowed={Chem.ChiralType.CHI_UNSPECIFIED,Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    if any(x.GetChiralTag() not in allowed for x in mol.GetAtoms()):
        raise ValueError('Unsupported non-tetrahedral atom stereochemistry')
    stereo=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    # Remove only stereochemistry: retain isotopes, charges, and disconnected components.
    parent=Chem.Mol(mol); Chem.RemoveStereochemistry(parent)
    family=Chem.MolToSmiles(parent,canonical=True,isomericSmiles=True)
    unspecified=sum(str(info.specified)=='Unspecified' for info in Chem.FindPotentialStereo(mol))
    return stereo,family,int(unspecified)

def native_feature_audit():
    from fragnet.dataset.features import FeaturesEXP
    f=FeaturesEXP()
    if not f.use_bond_chirality: raise RuntimeError('Bond stereochemistry is disabled')
    def atom(s):
        m=Chem.MolFromSmiles(s)
        a=next(a for a in m.GetAtoms() if a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED)
        return f.atom_features_one_hot(a)
    def bond(s):
        m=Chem.MolFromSmiles(s)
        b=next(b for b in m.GetBonds() if b.GetBondType()==Chem.BondType.DOUBLE)
        return f.bond_features_one_hot(b,use_chirality=True)
    r,s=atom('N[C@H](C)C(=O)O'),atom('N[C@@H](C)C(=O)O')
    e,z=bond('F/C=C/F'),bond('F/C=C\\F')
    if len(r)!=167 or len(e)!=17: raise RuntimeError('Native feature dimensions changed')
    if np.array_equal(r,s) or np.array_equal(e,z): raise RuntimeError('Native atom/bond stereo features collapse stereoisomers')
    return dict(atom_stereo_probe=True,bond_stereo_probe=True,atom_features=167,bond_features=17)

def make_3d(smiles,seed):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    for random_coords in (False,True):
        settings=AllChem.ETKDGv3(); settings.randomSeed=seed
        settings.enforceChirality=True; settings.useRandomCoords=random_coords; settings.numThreads=1
        if AllChem.EmbedMolecule(m,settings)==0: break
    else: raise ValueError('3D embedding failed; no silent 2D fallback')
    if AllChem.MMFFHasAllMoleculeParams(m): AllChem.MMFFOptimizeMolecule(m,maxIters=200)
    # Retain labels, and independently verify specified stereo against coordinates.
    encoded=molecule_identity(Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True))[0]
    if encoded!=smiles: raise ValueError('Stereo/identity changed while adding coordinates')
    check=Chem.Mol(m); Chem.AssignStereochemistryFrom3D(check,confId=0,replaceExistingTags=True)
    Chem.AssignStereochemistry(m,cleanIt=True,force=True)
    Chem.AssignStereochemistry(check,cleanIt=True,force=True)
    for a,b in zip(m.GetAtoms(),check.GetAtoms()):
        if a.HasProp('_CIPCode') and (not b.HasProp('_CIPCode') or a.GetProp('_CIPCode')!=b.GetProp('_CIPCode')):
            raise ValueError('3D embedding inverted a specified stereocentre')
    for a,b in zip(m.GetBonds(),check.GetBonds()):
        if a.GetStereo() in (Chem.BondStereo.STEREOE,Chem.BondStereo.STEREOZ) and a.GetStereo()!=b.GetStereo():
            raise ValueError('3D embedding changed specified double-bond stereo')
    return m

def validate_graph(g,smiles,y):
    import torch
    if g is None: raise ValueError('FragNet returned no graph')
    if molecule_identity(g.smiles)[0]!=smiles: raise ValueError('Graph identity mismatch')
    if g.x_atoms.ndim!=2 or g.x_atoms.shape[1]!=167 or g.x_frags.shape[1]!=167 or g.edge_attr.shape[1]!=17:
        raise ValueError('Graph feature dimensions mismatch')
    if g.y.numel()!=1 or not np.isclose(float(g.y.item()),y,rtol=1e-6,atol=1e-6): raise ValueError('Graph label mismatch')
    for key,value in g:
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Non-finite graph tensor: {key}')

def one_graph(row,creator,seed):
    m=make_3d(row.smiles,seed)
    graph=creator.create_data_point([row.smiles,[float(row.y)],m,m.GetConformer(), 'brics'])
    validate_graph(graph,row.smiles,row.y)
    # FragmentedMol may add wedges. Compare against the SAME post-fragmentation
    # molecule, not an independently reordered SMILES parse.
    f=creator.feature_creator
    atoms,edges,bonds=f.get_atom_and_bond_features_atom_graph_one_hot(m,True)
    if not np.array_equal(graph.x_atoms.cpu().numpy(),np.asarray(atoms,dtype=np.float32)):
        raise ValueError('Graph lost or reordered atom features')
    if not np.array_equal(graph.edge_index.cpu().numpy(),np.asarray(edges)) or not np.array_equal(graph.edge_attr.cpu().numpy(),np.asarray(bonds,dtype=np.float32)):
        raise ValueError('Graph lost or reordered bond features')
    return graph

def torch_load(path: Path, device: Any, torch: Any) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def unwrap_state_dict(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested
    return value

def build_model(
    config: dict[str, Any],
    head_params: dict[str, Any],
) -> Any:
    from fragnet.model.gat.gat2 import FragNetFineTune

    model_config = config["finetune"]["model"]
    return FragNetFineTune(
        n_classes=int(model_config["n_classes"]),
        atom_features=int(config["atom_features"]),
        frag_features=int(config["frag_features"]),
        edge_features=int(config["edge_features"]),
        num_heads=int(model_config["num_heads"]),
        num_layer=int(model_config["num_layer"]),
        drop_ratio=float(model_config["drop_ratio"]),
        emb_dim=int(model_config.get("emb_dim", 128)),
        h1=int(head_params["h1"]),
        h2=int(head_params["h2"]),
        h3=int(head_params["h3"]),
        h4=int(head_params["h4"]),
        act=str(head_params["act"]),
        fthead="FTHead3",
    )

class Predictor:
    def __init__(self, run_dir=ROOT/'fragnet_config_details',
                 fragnet_root=ROOT/'fragnet_source_code', threads=1):
        global np, pd, Chem, AllChem
        import numpy as np
        import pandas as pd
        import yaml
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import torch
        self.torch = torch
        self.run_dir = Path(run_dir).resolve()
        self.fragnet_root = Path(fragnet_root).resolve()
        if not (self.fragnet_root/'fragnet').is_dir():
            raise FileNotFoundError(self.fragnet_root/'fragnet')
        sys.path.insert(0, str(self.fragnet_root))
        from fragnet.dataset.data import CreateData, collate_fn
        self.collate = collate_fn
        self.config = yaml.safe_load((self.run_dir/'fragnet_selected.yaml').read_text())
        self.seed = int(self.config['seed'])
        self.checkpoint = self.run_dir/'experiment/ft.pt'
        self.checkpoint_hash = digest(self.checkpoint)
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
                if not path.exists() or digest(path) != expected:
                    raise RuntimeError(f'FragNet source differs from training: {path}')
        torch.set_num_threads(int(threads))
        native_feature_audit()
        self.model = build_model(self.config, self.config['finetune']['model'])
        state = unwrap_state_dict(torch_load(self.checkpoint, 'cpu', torch))
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
                smi, _, _ = molecule_identity(value)
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
                        graph = one_graph(SimpleNamespace(smiles=smi, y=0.), self.creator, self.seed)
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

def worker(args):
    # Reserve the original stdout pipe for JSON; native/model prints go to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    model = Predictor(Path(args.run_dir), Path(args.fragnet_root), args.threads)
    protocol.write(json.dumps(dict(ready=True, pid=os.getpid(), checkpoint_sha256=model.checkpoint_hash))+'\n')
    for line in sys.stdin:
        request = json.loads(line)
        result = model.predict(request['smiles'])
        rows = []
        for row in result.to_dict(orient='records'):
            row['prediction_ok'] = bool(row['prediction_ok'])
            value = float(row['fragnet_log_P_upconversion'])
            row['fragnet_log_P_upconversion'] = value if math.isfinite(value) else None
            rows.append(row)
        protocol.write(json.dumps(dict(id=request['id'], rows=rows), allow_nan=False)+'\n')

class Client:
    def __init__(self, args, output):
        python = Path(args.fragnet_python).expanduser().resolve()
        if not python.is_file():
            raise FileNotFoundError(f'FragNet Python not found: {python}')
        self.timeout = args.timeout
        self.buffer = b''
        self.lock = threading.Lock()
        self.counter = 0
        self.log_path = output/'fragnet_worker.log'
        self.log = self.log_path.open('a', buffering=1)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env['CONDA_PREFIX'] = str(python.parent.parent)
        env['CONDA_DEFAULT_ENV'] = 'fragnet'
        env['PATH'] = str(python.parent)+os.pathsep+env.get('PATH', '')
        command = [str(python), '-u', str(Path(__file__).resolve()), '_worker',
                   '--run-dir', str(args.run_dir), '--fragnet-root', str(args.fragnet_root),
                   '--threads', str(args.threads)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, env=env, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        atexit.register(self.close)
        try:
            self.info = self.receive()
            if not self.info.get('ready'):
                raise RuntimeError('FragNet worker did not report ready')
        except BaseException:
            self.close()
            raise

    def receive(self):
        deadline = time.monotonic()+self.timeout
        while b'\n' not in self.buffer:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not self.selector.select(max(0, remaining)):
                raise TimeoutError(f'FragNet timed out; see {self.log_path}')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f'FragNet worker exited; see {self.log_path}')
            self.buffer += chunk
            if len(self.buffer) > 32*1024*1024:
                raise RuntimeError('Oversized FragNet worker response')
        line, self.buffer = self.buffer.split(b'\n', 1)
        return json.loads(line)

    def predict(self, smiles):
        with self.lock:
            self.counter += 1
            request = json.dumps(dict(id=self.counter, smiles=smiles), allow_nan=False).encode()+b'\n'
            try:
                view = memoryview(request)
                while view:
                    n = self.process.stdin.write(view)
                    if not n:
                        raise BrokenPipeError('FragNet request pipe closed')
                    view = view[n:]
                response = self.receive()
                if response.get('id') != self.counter or len(response.get('rows', [])) != len(smiles):
                    raise RuntimeError('FragNet response is not aligned with request')
                for row in response['rows']:
                    if row['prediction_ok'] and not math.isfinite(float(row['fragnet_log_P_upconversion'])):
                        raise RuntimeError('Non-finite successful prediction')
                return response['rows']
            except BaseException:
                self.close()
                raise

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for pipe in (process.stdin, process.stdout):
                if pipe:
                    pipe.close()
            self.process = None
        if getattr(self, 'selector', None):
            self.selector.close()
        if getattr(self, 'log', None):
            self.log.close()

def validate_fitness(expression):
    allowed = {'np', 'SA', 'N_rot', 'log_P_upconversion', 'fragnet_log_P_upconversion',
               'P_upconversion', 'fragnet_prediction_ok'}
    tree = ast.parse(expression, mode='eval')
    unknown = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}-allowed
    if unknown:
        raise ValueError('Unavailable fitness inputs: '+', '.join(sorted(unknown))+
                         '. Length, area and log_P_upconversion_scaled are not predicted.')

def install_runtime(client, output):
    import numpy as np
    from rdkit import Chem
    module = importlib.import_module('NMO.oracle_handler')
    if getattr(module, '_fragnet_process_bridge', False):
        raise RuntimeError('Bridge already installed in this process')
    if '# FRAGNET_WORKFLOW_DISPATCH' in Path(module.__file__).read_text():
        raise RuntimeError('Earlier install_nmo patch found. Restore its backup before using this launcher.')
    csv_path = output/'fragnet_predictions.csv'
    fields = ['oracle_call', 'encoding', 'smiles', 'fragnet_log_P_upconversion',
              'prediction_ok', 'prediction_error', 'checkpoint_sha256']
    with csv_path.open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def dispatch(mols, encodings, config, metadata, rewards, indices, *extra, **kwargs):
        indices = np.asarray(indices, dtype=int)
        n = len(encodings)
        values = np.full(n, np.nan)
        ok = np.zeros(n, dtype=int)
        reasons = ['']*n
        smiles = [Chem.MolToSmiles(mols[i], canonical=True, isomericSmiles=True) for i in indices]
        rows = client.predict(smiles) if smiles else []
        for index, row in zip(indices, rows):
            reasons[index] = row['prediction_error']
            if row['prediction_ok']:
                values[index] = row['fragnet_log_P_upconversion']
                ok[index] = 1
        rewards['hash_values'] = np.array([uuid.uuid4().hex for _ in encodings])
        rewards['log_P_upconversion'] = values
        rewards['fragnet_log_P_upconversion'] = values.copy()
        rewards['fragnet_prediction_ok'] = ok
        with np.errstate(over='ignore', invalid='ignore'):
            rewards['P_upconversion'] = np.power(10., values)
        rewards['hl_gaps'] = np.full(n, np.nan)
        rewards['failure_reasons'] = np.asarray(reasons, dtype=object)
        return rewards, indices[ok[indices].astype(bool)], reasons

    # Existing NMO anchoring, validation, SA and rotatable-bond routines are kept.
    module.terahertz_workflow_handler = dispatch
    original_init = module.Oracle_Handler.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        props = {s.strip() for s in self.calculated_props.split(',')}
        if props != {'SA', 'N_rot', 'P_upconversion'}:
            raise ValueError('This launcher requires calculated_props = SA,N_rot,P_upconversion')
        if self.n_process != 1:
            raise ValueError('Persistent FragNet bridge requires n_oracle_processes = 1')
        validate_fitness(self.fitness_func)
        self.fitness_func = f'np.where(fragnet_prediction_ok, ({self.fitness_func}), 0.0)'

    module.Oracle_Handler.__init__ = init

    def wrap_rewards(original):
        @functools.wraps(original)
        def wrapped(self, worker_id, encodings, meta_data=None):
            metadata = {} if meta_data is None else meta_data
            rewards = original(self, worker_id, encodings, metadata)
            n = len(encodings)
            ok = np.asarray(rewards.get('fragnet_prediction_ok', np.zeros(n)), dtype=bool)
            rewards['fragnet_prediction_ok'] = ok.astype(int)
            # NMO zero-fills failed entries after dispatch; restore missing properties.
            for key in ('log_P_upconversion', 'fragnet_log_P_upconversion', 'P_upconversion', 'hl_gaps'):
                values = np.asarray(rewards.get(key, np.full(n, np.nan)), dtype=float).copy()
                values[~ok] = np.nan
                rewards[key] = values
            with csv_path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                for i, encoding in enumerate(encodings):
                    writer.writerow(dict(oracle_call=int(metadata.get('oracle_call_start', self.oracle_calls))+i,
                        encoding=encoding, smiles=rewards['smiles'][i],
                        fragnet_log_P_upconversion=rewards['fragnet_log_P_upconversion'][i] if ok[i] else '',
                        prediction_ok=bool(ok[i]), prediction_error=rewards.get('failure_reasons', ['']*n)[i],
                        checkpoint_sha256=client.info['checkpoint_sha256']))
            return rewards
        return wrapped

    for cls in (module.Oracle_Handler_Smiles, module.Oracle_Handler_GGS):
        cls.get_rewards_subproc = wrap_rewards(cls.get_rewards_subproc)
    module._fragnet_process_bridge = True
    return module

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Generator source differs from reviewed version near {old[:90]!r}; no original file changed')
    return text.replace(old, new, 1)

def source_copy(path, framework):
    """Small guarded edits to working copies; original repository is untouched."""
    text = path.read_text()
    text = replace_once(text, 'os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)',
                        'os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))')
    for marker, indent in (
        ('            already_calculated = full_history.encodings_in_memory(encodings)', '            '),
        ('                    already_calculated = full_history.encodings_in_memory(child_encodings)', '                    '),
    ):
        variable = 'child_encodings' if 'child_encodings' in marker else 'encodings'
        text = replace_once(text, marker,
                            indent+'if len('+variable+') == 0:\n'+indent+'    continue\n'+marker)
    old = '            meta_data = {"debug" : self.debug, "step": int(step)}\n'
    new = '''            remaining = self.oracle_handler.max_oracle_calls - self.oracle_handler.oracle_calls
            if remaining <= 0:
                break
            encodings = encodings[:remaining]
            sequences_for_eval = sequences_for_eval[:remaining]
            if len(encodings) == 0:
                continue
'''+old
    text = replace_once(text, old, new)
    old = '                    meta_data = {"debug": self.debug, "step": int(step), "generation": int(g)}\n'
    new = '''                    remaining = self.oracle_handler.max_oracle_calls - self.oracle_handler.oracle_calls
                    if remaining <= 0:
                        oracle_calls_exceeded = True
                        break
                    child_encodings = child_encodings[:remaining]
                    child_mutation_stats = child_mutation_stats[:remaining]
                    child_crossover_stats = child_crossover_stats[:remaining]
'''+old
    text = replace_once(text, old, new)
    # Exactly reaching the budget should finish, without another oracle call.
    for marker in (
        '            fitness, rewards, oracle_calls_exceeded = self.oracle_handler.get_fitness(encodings, meta_data = meta_data)',
        '                    child_fitness, child_rewards, oracle_calls_exceeded = self.oracle_handler.get_fitness(child_encodings, meta_data = meta_data)',
    ):
        indent = marker[:len(marker)-len(marker.lstrip())]
        text = replace_once(text, marker, marker+'\n'+indent+
                            'oracle_calls_exceeded = self.oracle_handler.oracle_calls >= self.oracle_handler.max_oracle_calls')
    # Save the trained model before optional legacy plotting can fail.
    save = '        torch.save(Agent.net.state_dict(), f"{self.log_dir}/agent_final.pt")'
    text = replace_once(text, save, '')
    marker = '        full_history.write_memory(f"{self.log_dir}/full_history.csv")'
    # Exact indentation avoids similarly named writes in debug branches.
    text = replace_once(text, '\n'+marker+'\n', '\n'+save+'\n'+marker+'\n')
    prefix = ('# Generated working copy. Original source: '+str(path)+'\n'
              'import sys as _bridge_sys\n_bridge_sys.path.insert(0, '+repr(str(framework))+')\n')
    text = prefix+text
    ast.parse(text)
    return text


def build_config(args, paths):
    c = configparser.ConfigParser(interpolation=None)
    c.read_dict({
        'General': dict(encoding_type='Smiles', max_seq_length='140',
                        grammar_path=str(paths['vocabulary']), num_layers='3', d_model='512', model='rnn'),
        'Training': dict(prior_path=str(paths['prior']), log_dir=str(paths['run']), batch_size='64',
                         n_steps=str(args.steps), learning_rate='0.0005', learning_rate_z='0.1',
                         n_oracle_processes='1', use_masking='False', use_max_seq_length_padding='False',
                         temperature='1.0', gradient_clipping='100000000000', rank_coefficient='0.01',
                         exploration_patience='8', exploration_rank_coeff='0.015', exploration_temp='1.25',
                         dynamic_explor_exploit='False', dynamic_cooldown='False', debug='False',
                         use_SMARTS_filters='True'),
        'Replay Training': dict(memory_size='1024', experience_replay='64', n_experience_iterations='8',
                               beta='30', kl_coefficient='0.01', strict_memory_handling='False',
                               descriptor_weight='0.0'),
        'Genetic Search': dict(genetic_search='True', population_size='64', mutation_rate='0.5',
                               crossover_rate='0.5', ga_generations='2', offspring_size='8'),
        'Oracle': dict(calculated_props='SA,N_rot,P_upconversion', fitness_func=FITNESS,
                       max_oracle_calls=str(args.calls), n_cpus_total=str(args.threads),
                       w_len='0.0', w_area='0.0', debug='False'),
    })
    return c

def prediction_command(command, argv):
    parser = argparse.ArgumentParser(description=f'{command} FragNet predictions')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'fragnet_config_details')
    parser.add_argument('--fragnet-root', type=Path, default=ROOT/'fragnet_source_code')
    parser.add_argument('--threads', type=int, default=1)
    if command == 'predict':
        parser.add_argument('--input', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--smiles-column', default='smiles')
    else:
        parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args(argv)
    if args.threads < 1 or getattr(args, 'count', 1) < 1:
        parser.error('threads and count must be positive')
    if command == 'predict' and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    model = Predictor(args.run_dir, args.fragnet_root, args.threads)
    if command == 'verify':
        print(json.dumps(model.verify(args.count), indent=2))
        return
    with args.input.open(encoding='utf-8-sig') as handle:
        header = handle.readline()
    separator = csv.Sniffer().sniff(header, delimiters=',;\t').delimiter if any(c in header for c in ',;\t') else ','
    frame = pd.read_csv(args.input, sep=separator, encoding='utf-8-sig')
    result = model.predict(frame[args.smiles_column].tolist())
    result.insert(0, 'input_row', np.arange(len(result))+2)
    if 'log_P_upconversion' in frame:
        result['xtb_log_P_upconversion'] = pd.to_numeric(frame.log_P_upconversion, errors='coerce').to_numpy()
        result['prediction_minus_xtb'] = result.fragnet_log_P_upconversion-result.xtb_log_P_upconversion
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output/'predictions.csv', index=False)
    manifest = dict(checkpoint=str(model.checkpoint), checkpoint_sha256=model.checkpoint_hash,
                    input=str(args.input.resolve()), successful=int(result.prediction_ok.sum()),
                    total=len(result), target='log_P_upconversion; no extra logarithm')
    (args.output/'prediction_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Predicted {result.prediction_ok.sum()}/{len(result)} molecules')


def generate(args):
    framework = args.repo/'genetic_GFN_framework'
    prior = args.prior or args.work/'generator_prior/prior.pt'
    output = args.work/args.name
    vocabulary = framework/'data/experiments/Voc_adapted'
    for path in (prior, vocabulary, framework/'train.py', args.fragnet_python,
                 args.run_dir/'experiment/ft.pt', args.run_dir/'fragnet_selected.yaml'):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a new --name: {output}')
    record = prior.parent/'prior_manifest.json'
    if record.exists():
        manifest = json.loads(record.read_text())
        if manifest['checkpoint_sha256'] != digest(prior):
            raise ValueError('Prior checkpoint differs from its saved completion record')
        signature = manifest['signature']
        for key, wanted in dict(model='rnn', num_layers=3, d_model=512, max_seq_length=140,
                                vocabulary_sha256=digest(vocabulary)).items():
            if signature.get(key) != wanted:
                raise ValueError(f'Prior is incompatible with generation: {key}')
    elif args.prior is None:
        raise ValueError('Prior completion record missing; use --prior only for a validated compatible checkpoint')
    working_source = source_copy(framework/'train.py', framework)
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.chdir(framework)
    sys.path[:0] = [str(framework), str(args.repo/'NMO'), str(args.repo),
                    str(args.repo/'NMO/external_packages/group-selfies')]
    # Validate checkpoint dimensions before starting a long generation run.
    import torch
    from action_space import Action_Space_Smiles
    from model import get_model
    actions = Action_Space_Smiles(str(vocabulary))
    prior_model = get_model('rnn', actions, 64, 140, num_layers=3, d_model=512)
    prior_model.net.load_state_dict(torch.load(prior, map_location='cpu', weights_only=False), strict=True)
    del prior_model
    output.mkdir(parents=True, exist_ok=True)
    paths = dict(vocabulary=vocabulary, prior=prior, run=output)
    config = build_config(args, paths)
    config['Oracle']['property_backend'] = 'fragnet_process'
    config_path = output/'nmo_fragnet.ini'
    with config_path.open('w') as handle:
        config.write(handle)
    script = output/'train.py'
    script.write_text(working_source)
    # NMO stays in this environment; only the child process loads FragNet.
    client = Client(args, output)
    try:
        module = install_runtime(client, output)
        manifest = dict(source_commit=SOURCE_COMMIT, mode='run', nmo_python=sys.executable,
                        fragnet_python=str(args.fragnet_python), **client.info,
                        prior=str(prior), prior_sha256=digest(prior), vocabulary_sha256=digest(vocabulary),
                        fitness=FITNESS, original_length_area_scaling=False,
                        configuration=str(config_path), nmo_oracle=module.__file__,
                        oracle_source_sha256=digest(module.__file__),
                        generator_source_sha256=digest(framework/'train.py'),
                        working_source_sha256=digest(script), seed=args.seed)
        (output/'bridge_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        sys.argv = [str(script), str(config_path), '--seed', str(args.seed), '--gpu', '0']
        runpy.run_path(str(script), run_name='__main__')
    finally:
        client.close()
    for name in ('full_history.csv', 'agent_final.pt'):
        if not (output/name).is_file():
            raise RuntimeError(f'Generation did not finish: missing {output/name}')
    with (output/'fragnet_predictions.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    accepted = {}
    for row in rows:
        if row['prediction_ok'].lower() in ('true', '1'):
            accepted.setdefault(row['smiles'], row)
    with (output/'unique_fragnet_candidates.csv').open('w', newline='') as handle:
        fields = list(rows[0]) if rows else ['smiles', 'fragnet_log_P_upconversion', 'prediction_ok']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted.values())
    summary = dict(oracle_evaluations=len(rows), prediction_failures=sum(
        row['prediction_ok'].lower() not in ('true', '1') for row in rows),
        unique_successful_molecules=len(accepted), requested_oracle_calls=args.calls,
        fitness=FITNESS, length_area_penalties=False,
        note='Novelty relative to training/reference has not been established; mapping is a separate step.')
    (output/'generation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ('predict', 'verify'):
        prediction_command(argv[0], argv[1:])
        return
    if argv and argv[0] == '_worker':
        parser = argparse.ArgumentParser()
        parser.add_argument('--run-dir', type=Path, required=True)
        parser.add_argument('--fragnet-root', type=Path, required=True)
        parser.add_argument('--threads', type=int, required=True)
        worker(parser.parse_args(argv[1:]))
        return
    if argv and argv[0] == 'generate':
        argv = argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT/'NMO_source_code')
    parser.add_argument('--work', type=Path, default=ROOT/'nmo_fragnet_generation')
    parser.add_argument('--prior', type=Path, help='Existing compatible checkpoint; never retrains')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'fragnet_config_details')
    parser.add_argument('--fragnet-root', type=Path, default=ROOT/'fragnet_source_code')
    parser.add_argument('--fragnet-python', type=Path, default=Path.home()/'.conda/envs/fragnet/bin/python')
    parser.add_argument('--name', default='pilot_seed123_v2')
    parser.add_argument('--calls', type=int, default=256)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args(argv)
    if min(args.calls, args.steps, args.threads) < 1 or args.timeout <= 0 or args.seed < 0:
        parser.error('calls, steps, threads and timeout must be positive; seed must be nonnegative')
    if Path(args.name).name != args.name or args.name in ('', '.', '..', 'generator_prior', 'configs', 'prepared_code'):
        parser.error('--name must be a simple new run directory name')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    generate(args)


if __name__ == '__main__':
    main()
