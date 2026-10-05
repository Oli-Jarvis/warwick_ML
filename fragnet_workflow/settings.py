"""Paths and run defaults, resolved independently of the working directory."""
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
