#!/usr/bin/env python3
"""Step 2: generate molecules, using FragNet predictions to calculate fitness."""
import sys

from generator import FITNESS, HERE, call, environment, paths_for, prepare, runtime_check
from settings import parse_args


def generate(args, paths):
    if not paths['prior'].is_file():
        raise FileNotFoundError(f'Generator prior missing: {paths["prior"]}. Run pretrain_generator.py first.')
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


def main(argv=None):
    args = parse_args('generate', argv)
    paths = paths_for(args)
    if paths['run'].exists() and any(paths['run'].iterdir()):
        raise FileExistsError(f"Choose a new run name: {paths['run']}")
    prepare(args, paths, prediction=True)
    runtime_check(args, paths)
    generate(args, paths)


if __name__ == '__main__':
    main()
