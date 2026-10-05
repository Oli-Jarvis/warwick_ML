"""Paths and run defaults, resolved independently of the working directory."""
import argparse
import configparser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name('workflow.ini')


def load_defaults(path=None):
    path = DEFAULT_CONFIG if path is None else Path(path).expanduser().resolve()
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path):
        raise FileNotFoundError(path)
    path_keys = {'root', 'repo', 'work', 'reference', 'run_dir', 'fragnet_root', 'fragnet_python'}
    run_types = dict(name=str, epochs=int, calls=int, steps=int, seed=int,
                     threads=int, timeout=float, cpu=bool)
    if set(config.sections()) != {'paths', 'run'}:
        raise ValueError('Settings must have [paths] and [run] sections')
    if set(config['paths']) != path_keys or set(config['run']) != set(run_types):
        raise ValueError('Settings keys must match the supplied workflow.ini')
    root = Path(config['paths']['root']).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    values = {}
    for key in sorted(path_keys - {'root'}):
        value = Path(config['paths'][key]).expanduser()
        values[key] = (value if value.is_absolute() else root / value).resolve()
    for key, convert in run_types.items():
        values[key] = (config.getboolean('run', key) if convert is bool
                       else convert(config['run'][key]))
    return values


def parse_args(step, argv=None):
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument('--config', type=Path)
    config_args, _ = initial.parse_known_args(argv)
    defaults = load_defaults(config_args.config)
    p = argparse.ArgumentParser(description={
        'pretrain': 'Pretrain or reuse the RNN generator.',
        'generate': 'Generate molecules and score them with FragNet.',
        'project': 'Project a scored run into the saved chemical space.',
    }[step])
    p.set_defaults(**defaults, prior=None)
    p.add_argument('--config', type=Path)
    p.add_argument('--work', type=Path)
    p.add_argument('--name')
    p.add_argument('--threads', type=int)
    if step in ('pretrain', 'generate'):
        p.add_argument('--repo', type=Path)
        p.add_argument('--prior', type=Path)
        p.add_argument('--seed', type=int)
        p.add_argument('--cpu', action=argparse.BooleanOptionalAction)
    if step == 'pretrain':
        p.add_argument('--epochs', type=int)
    if step == 'generate':
        p.add_argument('--calls', type=int)
        p.add_argument('--steps', type=int)
        p.add_argument('--run-dir', type=Path)
        p.add_argument('--fragnet-root', type=Path)
        p.add_argument('--timeout', type=float)
    if step in ('generate', 'project'):
        p.add_argument('--fragnet-python', type=Path)
    if step == 'project':
        p.add_argument('--reference', type=Path)
        p.add_argument('--calls', type=int, help='Original requested budget for the summary')
    args = p.parse_args(argv)
    if Path(args.name).name != args.name or args.name in ('', '.', '..', 'configs', 'prepared_code', 'generator_prior'):
        p.error('name must be a simple run directory name')
    if min(args.epochs, args.calls, args.steps, args.threads) < 1 or args.timeout <= 0 or args.seed < 0:
        p.error('counts and timeout must be positive; seed must be nonnegative')
    for key in ('repo', 'work', 'reference', 'run_dir', 'fragnet_root', 'fragnet_python', 'prior'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    return args
