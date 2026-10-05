#!/usr/bin/env python3
"""Pretrain a generator, generate and score molecules, then project into a saved space.

Run generation in the nmo environment. Prediction and projection use the
FragNet interpreter configured in workflow.ini. Every generation needs a new name.
"""
import argparse
import ast
import configparser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from settings import ROOT, load_defaults
HERE = Path(__file__).resolve().parent
# Original positive log-P transformation with w_len=w_area=0, including offset 5.
# SA^2 and N_rot penalties, and the leading 1/4, match the user's original config.
# np.logaddexp is the numerically stable form of log10(1+10**exponent).
FITNESS = ('0.25*(np.logaddexp(0.0,np.log(10.0)*'
           'np.clip(log_P_upconversion+5.0,-50.0,80.0))/np.log(10.0))'
           '*((10.0-SA)/9.0)**2/(1.0+np.exp(2.0*(N_rot-3.5)))')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Generator source differs from reviewed version near {old[:90]!r}; no original file changed')
    return text.replace(old, new, 1)


def source_copy(path, framework, kind):
    """Small guarded edits to working copies; original repository is untouched."""
    text = path.read_text()
    text = replace_once(text, 'os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)',
                        'os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))')
    if kind == 'pretrain':
        # This unused constructor precedes the Smiles/GGS branch. The GGS branch
        # still constructs its real grammar with GroupGrammar.from_file.
        text = replace_once(text, '    grammar = GroupGrammar(grammar_path)\n', '')
        text = replace_once(text, 'for epoch in range(1, epochs):',
                            'for epoch in range(1, epochs + 1):')
    else:
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
        'Dataset': dict(dataset_path=str(paths['dataset']), include_descriptors='False'),
        'Pretrain': dict(dataset_path=str(paths['dataset']), epochs=str(args.epochs), batch_size='128',
                         output_path=str(paths['prior_dir']), learning_rate='0.0005', plot_loss='True',
                         save_logs='False', use_masking='False', use_max_seq_length_padding='False',
                         loss_weight='0.0'),
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


def paths_for(args):
    framework = args.repo.resolve()/'genetic_GFN_framework'
    work = args.work.resolve()
    prior_dir = work/'generator_prior'
    return dict(framework=framework, work=work, prior_dir=prior_dir,
                prior=args.prior.resolve() if args.prior else prior_dir/'prior.pt',
                vocabulary=framework/'data/experiments/Voc_adapted',
                dataset=framework/'data/experiments/translated_smiles.smi',
                run=work/args.name, projection=work/(args.name+'_space'),
                code=work/'prepared_code', config=work/'configs'/(args.name+'.ini'))


def prepare(args, paths):
    for name in ('dataset', 'vocabulary'):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    required = [HERE/'nmo_fragnet_runner.py', HERE/'predictor.py', HERE/'workflow.py',
                args.fragnet_python.expanduser(), args.reference/'reference.pkl',
                args.run_dir/'experiment/ft.pt', args.run_dir/'fragnet_selected.yaml']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    sources = {}
    for name in ('pretrain', 'train'):
        original = paths['framework']/(name+'.py')
        sources[name] = source_copy(original, paths['framework'], name)
    c = build_config(args, paths)
    paths['code'].mkdir(parents=True, exist_ok=True)
    paths['config'].parent.mkdir(parents=True, exist_ok=True)
    for name, text in sources.items():
        path = paths['code']/(name+'.py')
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
    with paths['config'].open('w') as f:
        c.write(f)
    report = dict(config=str(paths['config']), source_repo=str(args.repo.resolve()),
                  generated_sources={name: dict(original_sha256=digest(paths['framework']/(name+'.py')),
                                                working_sha256=digest(paths['code']/(name+'.py')))
                                     for name in sources},
                  prior=str(paths['prior']), prior_exists=paths['prior'].is_file(),
                  dataset=str(paths['dataset']), vocabulary=str(paths['vocabulary']),
                  pretrain_epochs=args.epochs, generation_seed=args.seed,
                  requested_oracle_calls=args.calls, max_steps=args.steps,
                  fitness=FITNESS, length_area_penalties=False,
                  original_positive_logp_transform=True,
                  note='Raw predictions are stored independently of fitness. Original xTB geometry penalties are not reproduced.')
    (paths['config'].with_suffix('.json')).write_text(json.dumps(report, indent=2)+'\n')
    (paths['work']/'README.txt').write_text(
        'FragNet-driven SMILES generation\n\n'
        'Generator: 3-layer RNN, d_model=512; vocabulary Voc_adapted.\n'
        'Pretraining: translated_smiles.smi, six actual epochs by default.\n'
        'FragNet checkpoint and historical reference are reused.\n\n'
        'Fitness = 1/4 * positive_logP * ((10-SA)/9)^2 / (1+exp(2*(N_rot-3.5)))\n'
        'positive_logP = log10(1+10^clip(predicted_logP+5,-50,80)).\n'
        'Length and area penalties have weights zero. This is NOT the complete\n'
        'original xTB geometry-scaled objective. Raw predictions remain in the CSV.\n\n'
        'Working source copies fix the pretraining epoch off-by-one, omit an unused\n'
        'GroupGrammar construction, respect scheduler CUDA_VISIBLE_DEVICES, skip\n'
        'empty evaluation batches and cap oracle batches at the remaining budget.\n'
        'The final generator checkpoint is saved before legacy plotting.\n'
        'Original source files and installed NMO are not edited.\n\n'
        'A pilot requests 256 oracle evaluations, including prediction failures.\n'
        'The step cap may stop a run before this budget; inspect generation_summary.json.\n'
        'Generated candidates are not guaranteed novel versus existing molecules.\n'
        'Candidate chemical-space descriptors are calculated; historical descriptors\n'
        'and PCA/cluster fits are reused.\n\n'
        'Pretraining is reused only when its recorded settings/hash match.\n'
        'Generation restart/resume is not implemented by this launcher.\n'
        'If a generation directory already exists, inspect it and choose a new --name.\n'
        'Existing prior configurations from outside this workflow require a compatible\n'
        'model AND exactly matching vocabulary/token order when using --prior.\n'
    )
    return c, report


def environment(args):
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(args.threads)
    env['MKL_NUM_THREADS'] = str(args.threads)
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONUNBUFFERED'] = '1'
    if args.cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def call(command, log_path, cwd, env):
    print('\nRunning: '+' '.join(map(str, command)), flush=True)
    print(f'Log: {log_path}', flush=True)
    with log_path.open('a', buffering=1) as log:
        log.write('\nCOMMAND '+json.dumps(list(map(str, command)))+'\n')
        proc = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            raise
        finally:
            proc.stdout.close()
    if code:
        raise RuntimeError(f'Command exited with code {code}; see {log_path}')


def runtime_check(args, paths):
    # Separate interpreter keeps parent imports/device initialization out of launch.
    script = '''import sys, json, importlib, os
sys.path.insert(0, sys.argv[1])
modules = ['NMO', 'action_space', 'model', 'utils', 'dataset', 'genetic_search', 'train_utils', 'analysis']
errors = {}
for name in modules:
    try: importlib.import_module(name)
    except Exception as exc: errors[name] = type(exc).__name__+': '+str(exc)
if errors:
    raise RuntimeError('Generator import errors: '+json.dumps(errors))
import torch
from action_space import Action_Space_Smiles
from utils import Smiles_MolData, padding_and_valid_mask
from model import get_model
vocabulary, dataset, prior = sys.argv[2:5]
actions = Action_Space_Smiles(vocabulary)
data = Smiles_MolData(dataset, actions)
if len(data) < 128:
    raise ValueError('Fewer than 128 training rows; configured drop_last pretraining would be unsuitable')
# Use the actual tokenizer/collator before starting a long pretraining job.
sample = [data[i] for i in range(min(8, len(data)))]
batch = Smiles_MolData.collate_fn(sample, fill_value=actions.reversed_action_space['End'])
padding_and_valid_mask(batch.long(), actions, 140)
# A supplied/reused prior must match this model and vocabulary in size.
if os.path.isfile(prior):
    model = get_model('rnn', actions, 64, 140, num_layers=3, d_model=512)
    state = torch.load(prior, map_location='cpu', weights_only=False)
    model.net.load_state_dict(state, strict=True)
print(json.dumps(dict(python=sys.executable, torch=torch.__version__, cuda_available=torch.cuda.is_available(),
    training_rows=len(data), vocabulary=vocabulary, prior_shape_check=os.path.isfile(prior)), indent=2))
'''
    call([sys.executable, '-c', script, paths['framework'], paths['vocabulary'], paths['dataset'], paths['prior']],
         paths['work']/'preflight.log', paths['framework'], environment(args))


def prior_signature(args, paths):
    return dict(dataset_sha256=digest(paths['dataset']), vocabulary_sha256=digest(paths['vocabulary']),
                pretrain_source_sha256=digest(paths['code']/'pretrain.py'), epochs=args.epochs,
                model='rnn', num_layers=3, d_model=512, max_seq_length=140, seed=args.seed)


def pretrain(args, paths):
    signature = prior_signature(args, paths)
    record = paths['prior_dir']/'prior_manifest.json'
    if args.prior:
        if not paths['prior'].is_file():
            raise FileNotFoundError(paths['prior'])
        print(f'Using supplied prior: {paths["prior"]}', flush=True)
        return
    if paths['prior'].is_file():
        if not record.is_file():
            raise ValueError('Prior exists without completion record. Check pretrain.log before reusing it; use --prior explicitly if validated.')
        manifest = json.loads(record.read_text())
        if manifest['signature'] != signature or manifest['checkpoint_sha256'] != digest(paths['prior']):
            raise ValueError('Prior settings/hash differ from this request. Use a new --work directory or explicitly supply a compatible --prior.')
        print(f'Reusing completed generator prior: {paths["prior"]}', flush=True)
        return
    paths['prior_dir'].mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    call([sys.executable, paths['code']/'pretrain.py', paths['config'], '--seed', args.seed],
         paths['work']/'pretrain.log', paths['framework'], environment(args))
    if not paths['prior'].is_file():
        raise RuntimeError('Pretraining ended without prior.pt')
    record.write_text(json.dumps(dict(signature=signature, checkpoint_sha256=digest(paths['prior']),
                                    elapsed_seconds=time.monotonic()-start), indent=2)+'\n')


def generate(args, paths):
    if not paths['prior'].is_file():
        raise FileNotFoundError(f'Generator prior missing: {paths["prior"]}. Run pretrain or all first.')
    if paths['run'].exists() and any(paths['run'].iterdir()):
        raise FileExistsError(f'Generation output exists: {paths["run"]}. Choose a new --name.')
    command = [sys.executable, HERE/'nmo_fragnet_runner.py', 'run',
               '--config', paths['config'], '--train-script', paths['code']/'train.py',
               '--fitness', FITNESS, '--seed', args.seed, '--output', paths['run'],
               '--fragnet-python', args.fragnet_python, '--run-dir', args.run_dir,
               '--fragnet-root', args.fragnet_root, '--threads', args.threads,
               '--timeout', args.timeout]
    call(command, paths['work']/(args.name+'.log'), paths['framework'], environment(args))
    if not (paths['run']/'full_history.csv').is_file():
        raise RuntimeError('Generator exited before final history was written (possibly a scheduler time-limit checkpoint). See generation log.')
    if not (paths['run']/'agent_final.pt').is_file():
        raise RuntimeError('Generator ended without agent_final.pt; inspect generation log.')


def project(args, paths):
    import csv
    source = paths['run']/'fragnet_predictions.csv'
    if not source.is_file():
        raise FileNotFoundError(source)
    if paths['projection'].exists() and any(paths['projection'].iterdir()):
        raise FileExistsError(f'Projection already exists: {paths["projection"]}')
    # Deduplicate the bridge log by its canonical anchored SMILES; retain raw log.
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


def prediction_stage(args):
    """Run prediction or CSV projection in the existing FragNet environment."""
    command = [args.fragnet_python, HERE/'workflow.py', args.stage,
               '--input', args.input, '--output', args.output]
    if args.stage == 'predict':
        command += ['--run-dir', args.run_dir, '--fragnet-root', args.fragnet_root,
                    '--threads', args.threads, '--smiles-column', args.smiles_column]
        if args.with_space:
            command += ['--reference', args.reference]
    else:
        command += ['--reference', args.reference, '--property-column', args.property_column]
    env = environment(args)
    python = args.fragnet_python.expanduser().resolve()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['CONDA_PREFIX'] = str(python.parent.parent)
    env['PATH'] = str(python.parent)+os.pathsep+env.get('PATH', '')
    subprocess.run(list(map(str, command)), cwd=HERE, env=env, check=True)


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    defaults = load_defaults(config_args.config)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('prepare', 'pretrain', 'generate', 'predict', 'project', 'all'))
    p.add_argument('--config', type=Path, help='Path/run settings; defaults to workflow.ini')
    p.add_argument('--repo', type=Path, default=defaults['repo'])
    p.add_argument('--work', type=Path, default=defaults['work'])
    p.add_argument('--reference', type=Path, default=defaults['reference'])
    p.add_argument('--run-dir', type=Path, default=defaults['run_dir'])
    p.add_argument('--fragnet-root', type=Path, default=defaults['fragnet_root'])
    p.add_argument('--fragnet-python', type=Path, default=defaults['fragnet_python'])
    p.add_argument('--name', default=defaults['name'])
    p.add_argument('--prior', type=Path, help='Optional compatible RNN prior; skips pretraining')
    p.add_argument('--epochs', type=int, default=defaults['epochs'], help='Actual pretraining epochs, default 6')
    p.add_argument('--calls', type=int, default=defaults['calls'], help='Oracle budget; default is a pilot, not a full search')
    p.add_argument('--steps', type=int, default=defaults['steps'], help='Maximum generator steps')
    p.add_argument('--seed', type=int, default=defaults['seed'])
    p.add_argument('--threads', type=int, default=defaults['threads'], help='Set to no more than your allocated CPU count')
    p.add_argument('--timeout', type=float, default=defaults['timeout'], help='FragNet startup/batch timeout in seconds')
    p.add_argument('--cpu', action=argparse.BooleanOptionalAction, default=defaults['cpu'],
                   help='Force CPU instead of any visible GPU')
    p.add_argument('--input', type=Path, help='CSV for predict or standalone project')
    p.add_argument('--output', type=Path, help='New output directory for CSV commands')
    p.add_argument('--smiles-column', default='smiles')
    p.add_argument('--property-column', default='fragnet_log_P_upconversion')
    p.add_argument('--with-space', action='store_true', help='Project predictions into the saved reference')
    args = p.parse_args(argv)
    if Path(args.name).name != args.name or args.name in ('', '.', '..', 'configs', 'prepared_code', 'generator_prior'):
        p.error('--name must be a simple run directory name')
    if min(args.epochs, args.calls, args.steps, args.threads) < 1 or args.timeout <= 0 or args.seed < 0:
        p.error('epochs/calls/steps/threads/timeout must be positive; seed must be nonnegative')
    for name in ('repo', 'work', 'reference', 'run_dir', 'fragnet_root', 'fragnet_python'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.input is not None or args.stage == 'predict':
        if args.stage not in ('predict', 'project') or args.input is None or args.output is None:
            p.error('CSV commands require predict/project, --input and --output')
        args.input = args.input.expanduser().resolve()
        args.output = args.output.expanduser().resolve()
    elif args.output is not None:
        p.error('--output requires --input')
    if args.with_space and args.stage != 'predict':
        p.error('--with-space is only used with predict')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.input is not None:
        prediction_stage(args)
        return
    paths = paths_for(args)
    if args.stage == 'project':
        project(args, paths)
        return
    if args.stage in ('generate', 'all') and paths['run'].exists() and any(paths['run'].iterdir()):
        raise FileExistsError(f'Output exists: {paths["run"]}; choose a new --name')
    c, report = prepare(args, paths)
    print(json.dumps(report, indent=2), flush=True)
    runtime_check(args, paths)
    if args.stage in ('pretrain', 'all'):
        pretrain(args, paths)
    if args.stage in ('generate', 'all'):
        # Check newly trained weights match the generation architecture exactly.
        if args.stage == 'all': runtime_check(args, paths)
        generate(args, paths)
        project(args, paths)


if __name__ == '__main__':
    main()
