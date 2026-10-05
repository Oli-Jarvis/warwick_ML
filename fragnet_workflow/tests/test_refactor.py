"""Check the stepwise workflow against the uploaded baseline."""
import ast
import contextlib
import io
import json
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
import generator
import pretrain_generator
import generate_molecules
import project_molecules
from settings import load_defaults, parse_args


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

    def test_relocated_generation_and_projection_functions(self):
        before = original('fragnet_workflow/start_fragnet_generation.py').replace(
            'Run pretrain or all first.', 'Run pretrain_generator.py first.')
        expected = definitions(before)
        for filename in ('generator.py', 'pretrain_generator.py', 'generate_molecules.py', 'project_molecules.py'):
            for name, node in definitions((WORKFLOW / filename).read_text()).items():
                if name not in ('main', 'prepare'):
                    with self.subTest(file=filename, function=name):
                        self.assertEqual(node, expected[name])
        self.assertEqual(self.old.FITNESS, generator.FITNESS)

    def test_model_and_geometry_helpers_match_baseline(self):
        for new, old, count in (('model_utils.py', 'training_reference.py', 9), ('conformers.py', 'legacy_dim.py', 5)):
            helpers = definitions((WORKFLOW / new).read_text())
            expected = definitions(original('fragnet_workflow/' + old))
            self.assertEqual(len(helpers), count)
            for name, node in helpers.items():
                with self.subTest(file=new, function=name):
                    self.assertEqual(node, expected[name])

    def test_predictor_and_space_only_change_imports(self):
        for name in ('predictor.py', 'space.py', 'workflow.py', 'nmo_fragnet_runner.py'):
            before = original('fragnet_workflow/' + name)
            before = before.replace('import training_reference as training', 'import model_utils as training')
            before = before.replace('from training_reference import molecule_identity', 'from model_utils import molecule_identity')
            before = before.replace('import legacy_dim as legacy', 'import conformers as geometry')
            before = before.replace('legacy.make_3d_molecules', 'geometry.make_3d_molecules')
            with self.subTest(file=name):
                self.assertEqual(definitions(before), definitions((WORKFLOW / name).read_text()))
        self.assertEqual(original('run_fragnet.py').rstrip(), (ROOT / 'run_fragnet.py').read_text().rstrip())

    def test_defaults_and_full_nmo_configuration(self):
        args = parse_args('generate', [])
        for key, expected in dict(epochs=6, calls=256, steps=10000, seed=123,
                                  threads=4, timeout=1800, cpu=False, name='pilot_seed123').items():
            self.assertEqual(getattr(args, key), expected)
        paths = generator.paths_for(args)
        self.assertEqual(dict(self.old.build_config(args, paths)), dict(generator.build_config(args, paths)))

    def test_settings_paths_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'settings.ini'
            config.write_text((WORKFLOW / 'workflow.ini').read_text().replace('root = ..', 'root = data')
                              .replace('calls = 256', 'calls = 40').replace('cpu = false', 'cpu = true'))
            defaults = load_defaults(config)
            self.assertEqual(defaults['work'], Path(directory) / 'data/nmo_fragnet_generation')
            args = parse_args('generate', ['--config', str(config), '--calls', '32', '--no-cpu'])
            self.assertEqual(args.calls, 32)
            self.assertFalse(args.cpu)
            self.assertEqual(args.work, defaults['work'])

    def test_unknown_settings_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'settings.ini'
            config.write_text((WORKFLOW / 'workflow.ini').read_text().replace('calls =', 'calss ='))
            with self.assertRaises(ValueError):
                load_defaults(config)

    def test_each_script_runs_only_its_own_step(self):
        with tempfile.TemporaryDirectory() as directory:
            for module, expected in ((pretrain_generator, ['prepare', 'runtime_check', 'pretrain']),
                                     (generate_molecules, ['prepare', 'runtime_check', 'generate']),
                                     (project_molecules, ['project'])):
                trace = []
                with contextlib.ExitStack() as stack:
                    for name in expected:
                        stack.enter_context(patch.object(module, name, lambda *a, _name=name, **kw: trace.append(_name)))
                    module.main(['--work', directory])
                self.assertEqual(trace, expected)
                self.assertNotIn('project', trace if module != project_molecules else [])

    def test_existing_generation_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / 'pilot_seed123'
            existing.mkdir()
            (existing / 'keep.txt').write_text('existing result')
            with self.assertRaises(FileExistsError):
                generate_molecules.main(['--work', directory])
            self.assertEqual((existing / 'keep.txt').read_text(), 'existing result')

    def test_pretraining_needs_no_fragnet_or_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args('pretrain', ['--repo', directory, '--work', str(Path(directory) / 'work')])
            args.fragnet_python = Path(directory) / 'missing-python'
            args.run_dir = Path(directory) / 'missing-model'
            args.reference = Path(directory) / 'missing-reference'
            paths = generator.paths_for(args)
            for path in (paths['dataset'], paths['vocabulary'], paths['framework']/'pretrain.py', paths['framework']/'train.py'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('input')
            with patch.object(generator, 'source_copy', return_value='pass\n'):
                generator.prepare(args, paths)
                with self.assertRaises(FileNotFoundError):
                    generator.prepare(args, paths, prediction=True)
                args.fragnet_python.touch()
                args.run_dir.mkdir()
                (args.run_dir / 'experiment').mkdir()
                (args.run_dir / 'experiment/ft.pt').touch()
                (args.run_dir / 'fragnet_selected.yaml').touch()
                generator.prepare(args, paths, prediction=True)
            self.assertTrue(paths['config'].is_file())
            self.assertFalse(args.reference.exists())

    def test_projection_deduplication_and_failures_match_baseline(self):
        results = []
        for module in (self.old, project_molecules):
            with tempfile.TemporaryDirectory() as directory:
                args = parse_args('project', ['--work', directory])
                paths = generator.paths_for(args)
                paths['run'].mkdir(parents=True)
                (paths['run']/'fragnet_predictions.csv').write_text(
                    'smiles,prediction_ok,fragnet_log_P_upconversion\nCC,True,2.5\nCC,True,2.5\nCO,True,-2\nbad,False,\n')
                def fake_projection(*unused):
                    paths['projection'].mkdir()
                    (paths['projection']/'candidates_in_space.csv').write_text('smiles,cluster\nCC,0\nCO,1\n')
                with patch.object(module, 'call', fake_projection), contextlib.redirect_stdout(io.StringIO()):
                    module.project(args, paths)
                summary = json.loads((paths['run']/'generation_summary.json').read_text())
                results.append(((paths['run']/'unique_fragnet_candidates.csv').read_text(), summary))
        self.assertEqual(*results)
        self.assertEqual(results[0][1]['unique_successful_molecules'], 2)
        self.assertEqual(results[0][1]['prediction_failures'], 1)

    def test_projection_reads_recorded_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'pilot_seed123'
            run_dir.mkdir()
            (run_dir / 'nmo_fragnet.ini').write_text('[Oracle]\nmax_oracle_calls = 512\n')
            with patch.object(project_molecules, 'project') as project:
                project_molecules.main(['--work', directory])
            self.assertEqual(project.call_args.args[0].calls, 512)

    def test_step_arguments_are_separate(self):
        for step, args in (('pretrain', ['--calls', '10']), ('generate', ['--reference', 'space']),
                           ('project', ['--epochs', '6'])):
            with self.subTest(step=step), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(step, args)


if __name__ == '__main__':
    unittest.main()
