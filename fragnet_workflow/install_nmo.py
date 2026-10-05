#!/usr/bin/env python3
"""Patch the supplied oracle shape; backups and a separate config are mandatory."""
import argparse
import ast
import configparser
from pathlib import Path


def patch_source(source):
    if '# FRAGNET_WORKFLOW_DISPATCH' in source: return source
    old = 'from .terahertz_work_flow_handler import terahertz_workflow_handler'
    if source.count(old) != 1: raise ValueError('Unexpected terahertz import; stopping without changes')
    new = '''# FRAGNET_WORKFLOW_DISPATCH
from .terahertz_work_flow_handler import terahertz_workflow_handler as _xtb_terahertz_workflow_handler
from nmo_bridge import dispatch as _fragnet_dispatch

def terahertz_workflow_handler(*args, **kwargs):
    return _fragnet_dispatch(_xtb_terahertz_workflow_handler, *args, **kwargs)
'''
    source = source.replace(old, new)
    marker = '            config_dict["calculated_props"] = self.calculated_props'
    if source.count(marker) != 2: raise ValueError('Expected both GGS and SMILES config builders')
    extra = '''
            config_dict["property_backend"] = self.config.get("Oracle", "property_backend", fallback="xtb")
            if config_dict["property_backend"] == "fragnet":
                for _key in ("fragnet_run_dir", "fragnet_root", "fragnet_threads"):
                    config_dict[_key] = self.config.get("Oracle", _key)
'''
    source = source.replace(marker, marker+extra)
    marker = '        fitness = np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)'
    if source.count(marker) != 1: raise ValueError('Unexpected fitness handling')
    source = source.replace(marker, marker+'''
        if self.config.get("Oracle", "property_backend", fallback="xtb") == "fragnet":
            _prediction_ok = np.asarray(calculated_rewards.get("fragnet_prediction_ok", np.zeros(len(encoding_batch))), dtype=bool)
            fitness = np.where(_prediction_ok, fitness, 0.0)
''')
    ast.parse(source)
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--new-config', type=Path, required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--apply', action='store_true', help='Write patch and separate config; default is dry run')
    a = p.parse_args()
    c = configparser.ConfigParser(); c.read(a.config)
    if not c.has_section('Oracle') or not c.has_section('Training'): raise ValueError('Missing Oracle/Training sections')
    if a.config.resolve() == a.new_config.resolve(): raise ValueError('New config must have a different path')
    if a.new_config.exists(): raise FileExistsError(a.new_config)
    patched = patch_source(a.oracle.read_text())
    backup = a.oracle.with_suffix('.py.pre_fragnet')
    already_patched = '# FRAGNET_WORKFLOW_DISPATCH' in a.oracle.read_text()
    if backup.exists() and not already_patched: raise FileExistsError(backup)
    c['Oracle'].update(property_backend='fragnet', calculated_props='SA,N_rot,P_upconversion',
        fragnet_run_dir='/storage/msszkb_grp/msshfg/fragnet_combined_selected_stereo',
        fragnet_root='/storage/msszkb_grp/msshfg/smiles_baseline2/FragNet', fragnet_threads='1',
        fitness_func='1.0 / (1.0 + np.exp(-np.clip(log_P_upconversion / 5.0, -60.0, 60.0)))')
    c['Training']['n_oracle_processes'] = '1'
    c['Training']['log_dir'] = str(a.log_dir.resolve())
    if a.log_dir.exists() and any(a.log_dir.iterdir()): raise FileExistsError('Use a new NMO log directory')
    print('Validated oracle patch. New fitness: sigmoid(log_P_upconversion / 5); failed predictions get zero.')
    print('SA and N_rot are recorded; the configured objective rewards log P only.')
    print('The original xTB route remains selected for configurations without property_backend=fragnet.')
    if not a.apply:
        print('Dry run only. Repeat with --apply to write.'); return
    if not already_patched:
        backup.write_text(a.oracle.read_text())
        tmp = a.oracle.with_suffix('.py.tmp'); tmp.write_text(patched); tmp.replace(a.oracle)
    a.new_config.parent.mkdir(parents=True, exist_ok=True)
    with a.new_config.open('w') as f: c.write(f)
    print(f'Patched {a.oracle}; original backup {backup}; new configuration {a.new_config}')

if __name__ == '__main__': main()
