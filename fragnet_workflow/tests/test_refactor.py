"""Compare the refactor with the uploaded baseline using only the standard library.

Run: python -m unittest discover -s fragnet_workflow/tests -v
The baseline commit must be available in local Git history.
"""
import ast
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / 'fragnet_workflow'
BASELINE = os.environ.get('WORKFLOW_BASELINE', 'b240b788c8024c46a46bf365dc9be984fafef440')
sys.path.insert(0, str(WORKFLOW))
import run
from settings import load_defaults


def original(path):
    return subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT, text=True)


def definitions(source):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


class RefactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = types.ModuleType('baseline_generator')
        cls.old.__file__ = str(WORKFLOW / 'start_fragnet_generation.py')
        exec(compile(original('fragnet_workflow/start_fragnet_generation.py'), cls.old.__file__, 'exec'), cls.old.__dict__)

    def test_generation_functions_and_fitness_unchanged(self):
        before = definitions(original('fragnet_workflow/start_fragnet_generation.py'))
        after = definitions((WORKFLOW / 'run.py').read_text())
        for name, node in before.items():
            if name != 'main':
                with self.subTest(function=name):
                    self.assertEqual(node, after[name])
        self.assertEqual(self.old.FITNESS, run.FITNESS)

    def test_extracted_helpers_match_training_and_prediction_baseline(self):
        helpers = definitions((WORKFLOW / 'model_utils.py').read_text())
        self.assertEqual(len(helpers), 9)
        for path in ('run_fragnet.py', 'fragnet_workflow/training_reference.py'):
            before = definitions(original(path))
            for name, node in helpers.items():
                with self.subTest(path=path, function=name):
                    self.assertEqual(node, before[name])

    def test_predictor_only_changes_helper_import(self):
        before = original('fragnet_workflow/predictor.py').replace(
            'import training_reference as training', 'import model_utils as training')
        self.assertEqual(definitions(before), definitions((WORKFLOW / 'predictor.py').read_text()))

    def test_bridge_and_space_calculations_unchanged(self):
        for path in ('nmo_fragnet_runner.py', 'space.py', 'legacy_dim.py'):
            with self.subTest(path=path):
                self.assertEqual(definitions(original('fragnet_workflow/' + path)),
                                 definitions((WORKFLOW / path).read_text()))
        self.assertEqual(original('run_fragnet.py').rstrip(), (ROOT / 'run_fragnet.py').read_text().rstrip())

    def test_original_defaults_and_full_nmo_configuration(self):
        args = run.parse_args(['all'])
        for key, expected in dict(epochs=6, calls=256, steps=10000, seed=123,
                                  threads=4, timeout=1800, cpu=False, name='pilot_seed123').items():
            self.assertEqual(getattr(args, key), expected)
        paths = run.paths_for(args)
        self.assertEqual(dict(self.old.build_config(args, paths)), dict(run.build_config(args, paths)))

    def test_settings_paths_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'settings.ini'
            config.write_text((WORKFLOW / 'workflow.ini').read_text().replace('root = ..', 'root = data')
                              .replace('calls = 256', 'calls = 40').replace('cpu = false', 'cpu = true'))
            defaults = load_defaults(config)
            self.assertEqual(defaults['work'], Path(directory) / 'data/nmo_fragnet_generation')
            args = run.parse_args(['generate', '--config', str(config), '--calls', '32', '--no-cpu'])
            self.assertEqual(args.calls, 32)
            self.assertFalse(args.cpu)
            self.assertEqual(args.work, defaults['work'])

    def test_unknown_settings_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'settings.ini'
            config.write_text((WORKFLOW / 'workflow.ini').read_text().replace('calls =', 'calss ='))
            with self.assertRaises(ValueError):
                load_defaults(config)

    def test_stage_order_matches_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            for stage in ('prepare', 'pretrain', 'generate', 'project', 'all'):
                traces = []
                for module in (self.old, run):
                    trace = []
                    with contextlib.ExitStack() as stack:
                        for name in ('prepare', 'runtime_check', 'pretrain', 'generate', 'project'):
                            def record(*args, _name=name):
                                trace.append(_name)
                                return ({}, {}) if _name == 'prepare' else None
                            stack.enter_context(patch.object(module, name, record))
                        stack.enter_context(patch.object(sys, 'argv', ['run.py', stage, '--work', directory]))
                        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        module.main()
                    traces.append(trace)
                with self.subTest(stage=stage):
                    self.assertEqual(*traces)

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / 'pilot_seed123'
            existing.mkdir()
            (existing / 'keep.txt').write_text('existing result')
            with self.assertRaises(FileExistsError):
                run.main(['generate', '--work', directory])
            self.assertEqual((existing / 'keep.txt').read_text(), 'existing result')

    def test_prediction_routes_to_fragnet_with_absolute_paths(self):
        args = run.parse_args(['predict', '--input', 'molecules.csv', '--output', 'new_results', '--with-space'])
        with patch.dict(os.environ, {'PYTHONPATH': 'parent-only', 'PYTHONHOME': 'parent-only'}):
            with patch.object(run.subprocess, 'run') as launch:
                run.prediction_stage(args)
        command = launch.call_args.args[0]
        self.assertEqual(command[0], str(args.fragnet_python))
        self.assertIn(str(args.input), command)
        self.assertIn(str(args.reference), command)
        self.assertTrue(launch.call_args.kwargs['check'])
        self.assertNotIn('PYTHONPATH', launch.call_args.kwargs['env'])
        self.assertNotIn('PYTHONHOME', launch.call_args.kwargs['env'])

    def test_incomplete_csv_commands_rejected(self):
        for args in (['predict'], ['project', '--input', 'molecules.csv'],
                     ['all', '--input', 'molecules.csv', '--output', 'out']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run.parse_args(args)


if __name__ == '__main__':
    unittest.main()
