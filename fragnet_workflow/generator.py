"""Shared generator configuration, source preparation and process helpers."""
import ast
import configparser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
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

        for marker in (
            '            fitness, rewards, oracle_calls_exceeded = self.oracle_handler.get_fitness(encodings, meta_data = meta_data)',
            '                    child_fitness, child_rewards, oracle_calls_exceeded = self.oracle_handler.get_fitness(child_encodings, meta_data = meta_data)',
        ):
            indent = marker[:len(marker)-len(marker.lstrip())]
            text = replace_once(text, marker, marker+'\n'+indent+
                                'oracle_calls_exceeded = self.oracle_handler.oracle_calls >= self.oracle_handler.max_oracle_calls')

        save = '        torch.save(Agent.net.state_dict(), f"{self.log_dir}/agent_final.pt")'
        text = replace_once(text, save, '')
        marker = '        full_history.write_memory(f"{self.log_dir}/full_history.csv")'

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


def prepare(args, paths, *, prediction=False):
    for name in ('dataset', 'vocabulary'):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    if prediction:
        for path in (HERE/'nmo_fragnet_runner.py', HERE/'predictor.py',
                     args.fragnet_python, args.run_dir/'experiment/ft.pt',
                     args.run_dir/'fragnet_selected.yaml'):
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
