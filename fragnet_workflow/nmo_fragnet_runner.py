#!/usr/bin/env python3
"""NMO -> persistent FragNet subprocess, with no installed-source edits.

Place beside predictor.py in fragnet_workflow. Run using the nmo Python.
smoke: exercise NMO anchoring, FragNet, fitness, HDF5 and prediction logging.
run: execute a real genetic_GFN_framework/train.py with an explicit objective.
The existing workflow.py project command accepts the resulting prediction CSV.
The original xTB length/area-scaled reward is deliberately unavailable.
"""
import argparse
import atexit
import ast
import configparser
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

ROOT = Path('/storage/msszkb_grp/msshfg')
HERE = Path(__file__).resolve().parent


def worker(args):
    # Reserve the original stdout pipe for JSON; native/model prints go to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from predictor import Predictor
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


def read_examples(path, count):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        first = f.readline()
        f.seek(0)
        dialect = csv.Sniffer().sniff(first, delimiters=',;\t')
        rows = list(csv.DictReader(f, dialect=dialect))
    if not rows or 'encoding' not in rows[0]:
        raise ValueError('Smoke input needs raw SMILES in its encoding column')
    result = [r['encoding'] for r in rows[:count]]
    if not result:
        raise ValueError('No smoke examples')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('smoke', 'run', '_worker'):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', type=Path, default=ROOT/'fragnet_combined_selected_stereo')
        p.add_argument('--fragnet-root', type=Path, default=ROOT/'smiles_baseline2/FragNet')
        p.add_argument('--threads', type=int, default=1)
        if name == '_worker':
            continue
        p.add_argument('--fragnet-python', type=Path, default=Path.home()/'.conda/envs/fragnet/bin/python')
        p.add_argument('--timeout', type=float, default=900, help='Worker startup/batch timeout in seconds')
        p.add_argument('--output', type=Path, required=True, help='New, empty output directory')
        if name == 'smoke':
            p.add_argument('--input', type=Path, default=HERE/'example_history.csv')
            p.add_argument('--count', type=int, default=5)
        else:
            p.add_argument('--config', type=Path, required=True, help='Complete generation config, not experiment template')
            p.add_argument('--train-script', type=Path, default=ROOT/'TheNanotechnologyMolecularOptimizationBenchmark/genetic_GFN_framework/train.py')
            p.add_argument('--fitness', required=True, help='Explicit objective using predicted log P, SA and/or N_rot')
            p.add_argument('--seed', type=int, default=123)
            p.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.command == '_worker':
        worker(args)
        return
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a new output directory: {output}')
    config = configparser.ConfigParser()
    examples = None
    if args.command == 'smoke':
        if args.count < 1:
            parser.error('--count must be positive')
        examples = read_examples(args.input, args.count)
        # Positive monotonic score for plumbing test ONLY, not a generation objective.
        fitness = '1.0/(1.0+np.exp(-np.clip(log_P_upconversion/5.0,-60,60)))'
        config.read_dict({'General': {'grammar_path': '', 'encoding_type': 'Smiles'},
                          'Training': {}, 'Oracle': {'max_oracle_calls': '-1', 'n_cpus_total': '1'}})
    else:
        if not config.read(args.config):
            raise FileNotFoundError(args.config)
        if not args.train_script.is_file():
            raise FileNotFoundError(args.train_script)
        for section in ('General', 'Training', 'Replay Training', 'Genetic Search', 'Oracle'):
            if not config.has_section(section):
                raise ValueError(f'Missing [{section}] in generation config')
        for section, key in (('Training', 'prior_path'), ('General', 'grammar_path')):
            value = config.get(section, key, fallback='').strip()
            if value in ('', '.', './', '..') or not Path(value).exists():
                raise ValueError(f'Provide an existing {section}.{key}; current value is {value!r}')
        fitness = args.fitness
    validate_fitness(fitness)
    config['Training']['n_oracle_processes'] = '1'
    config['Training']['log_dir'] = str(output)
    config['Oracle']['calculated_props'] = 'SA,N_rot,P_upconversion'
    config['Oracle']['fitness_func'] = fitness
    config['Oracle']['property_backend'] = 'fragnet_process'
    output.mkdir(parents=True, exist_ok=True)
    config_path = output/'nmo_fragnet.ini'
    with config_path.open('w') as f:
        config.write(f)
    # NMO is imported only in the nmo parent; Torch/predictor only in the worker.
    importlib.import_module('NMO.oracle_handler')
    client = Client(args, output)
    try:
        module = install_runtime(client, output)
        manifest = dict(mode=args.command, nmo_oracle=module.__file__, nmo_python=sys.executable,
                        fragnet_python=str(args.fragnet_python), **client.info,
                        fitness=fitness, original_length_area_scaling=False,
                        launch_cwd=str(Path.cwd()), configuration=str(config_path),
                        oracle_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        (output/'bridge_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'FragNet worker ready: PID {client.info["pid"]}', flush=True)
        if args.command == 'smoke':
            oracle = module.Oracle_Handler_Smiles(str(config_path))
            results = []
            # Two calls test persistence, including NMO metadata writing on each call.
            for repeat in range(2):
                scores, rewards, exceeded = oracle.get_fitness(examples, {'step': repeat})
                results.append(dict(batch=repeat+1, requested=len(examples),
                                    predicted=int(sum(rewards['fragnet_prediction_ok'])),
                                    fitness=[float(x) for x in scores]))
            report = dict(worker_pid=client.info['pid'], worker_requests=client.counter, batches=results,
                          prediction_csv=str(output/'fragnet_predictions.csv'),
                          note='Repeated known examples test integration; not new molecule generation or accuracy validation.')
            (output/'smoke_result.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report, indent=2))
            if not all(r['predicted'] == r['requested'] for r in results):
                raise RuntimeError('Some examples failed; inspect fragnet_predictions.csv and fragnet_worker.log')
        else:
            script = args.train_script.resolve()
            sys.path.insert(0, str(script.parent))
            sys.argv = [str(script), str(config_path), '--seed', str(args.seed), '--gpu', str(args.gpu)]
            runpy.run_path(str(script), run_name='__main__')
    finally:
        client.close()


if __name__ == '__main__':
    main()

