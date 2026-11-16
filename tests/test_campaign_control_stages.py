"""CPU activation contract tests; no host data, GPU or production queue access."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

from mediflow_medvlm import campaign_supervisor as engine

local = Path(__file__).resolve().parent
controller_path = local / 'campaign_control_stages.py'
if not controller_path.exists():
    controller_path = local.parent / 'scripts/campaign_control_stages.py'
spec = importlib.util.spec_from_file_location('tested_stage_control', controller_path)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class StageContractTests(unittest.TestCase):
    def setUp(self):
        self.before = {'campaign_id': 'cpu-fixture', 'jobs': []}
        for cell in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'V1', 'V2', 'V3', 'V4']:
            job_id = 'v6-first-seed-' + cell
            self.before['jobs'].append({'job_id': job_id, 'run_id': job_id, 'phase': 'native_main_train',
                'cell_id': cell, 'implementation_pending': True, 'dependencies': []})
        self.before['jobs'].append({'job_id': stage.STAGE_ID, 'run_id': stage.STAGE_ID,
            'phase': 'first_seed_final', 'cell_id': 'ALL_V6', 'implementation_pending': True,
            'dependencies': [j['job_id'] for j in self.before['jobs']], 'dependency_policy': 'terminal',
            'reason': 'Original immutable placeholder reason'})
        self.after = copy.deepcopy(self.before)
        self.job = self.after['jobs'][-1]
        files = {'src/mediflow_medvlm/campaign_supervisor.py': 'a' * 64,
                 'src/mediflow_medvlm/campaign_recorded_first_seed.py': 'b' * 64}
        receipt = '/tmp/cpu-fixture/receipts/aggregate.json'
        self.job.update(implementation_pending=False, uses_gpu=False, gpu_lock_mode='wrapper',
            cwd='/tmp/cpu-fixture/snapshot', success_receipt=receipt,
            argv=[sys.executable, '-m', stage.STAGE_MODULE, '--config', '/tmp/cpu-fixture/config.json', '--receipt', receipt],
            source_files=files, source_hash=engine.canonical_hash(files), config_hash='c' * 64,
            data_hash='d' * 64, model_hash='e' * 64)

    def test_exact_cpu_aggregate_activation_preserves_ten_jobs(self):
        self.assertEqual(stage.validate_pair(engine, self.before, self.after), self.job)
        self.assertEqual(self.before['jobs'][:10], self.after['jobs'][:10])

    def test_completed_training_job_identity_may_not_change(self):
        self.after['jobs'][4]['run_id'] = 'replacement-t5'
        with self.assertRaisesRegex(ValueError, 'only unstarted CPU'):
            stage.validate_pair(engine, self.before, self.after)

    def test_metadata_and_scheduled_scope_may_not_change(self):
        self.after['seed'] = 123
        with self.assertRaisesRegex(ValueError, 'metadata'):
            stage.validate_pair(engine, self.before, self.after)

    def test_job_order_may_not_change(self):
        self.after['jobs'][0], self.after['jobs'][1] = self.after['jobs'][1], self.after['jobs'][0]
        with self.assertRaisesRegex(ValueError, 'membership or order'):
            stage.validate_pair(engine, self.before, self.after)

    def test_extra_gpu_model_job_is_rejected(self):
        self.after['jobs'].insert(0, {'job_id': 'unexpected', 'run_id': 'unexpected', 'cell_id': 'T1',
            'phase': 'calibration', 'implementation_pending': True, 'dependencies': []})
        with self.assertRaisesRegex(ValueError, 'membership or order'):
            stage.validate_pair(engine, self.before, self.after)

    def test_dependency_and_declared_reason_must_remain_immutable(self):
        for key, value in [('dependencies', []), ('dependency_policy', 'success'), ('reason', 'replacement')]:
            with self.subTest(key=key):
                altered = copy.deepcopy(self.after)
                altered['jobs'][-1][key] = value
                with self.assertRaisesRegex(ValueError, 'scientific identity or dependency'):
                    stage.validate_pair(engine, self.before, altered)

    def test_gpu_requirement_is_rejected(self):
        for key in ('uses_gpu', 'requires_gpu'):
            with self.subTest(key=key):
                altered = copy.deepcopy(self.after)
                altered['jobs'][-1][key] = True
                with self.assertRaises(ValueError):
                    stage.validate_pair(engine, self.before, altered)

    def test_child_gpu_lease_mode_is_rejected(self):
        self.job['gpu_lock_mode'] = 'child'
        with self.assertRaisesRegex(ValueError, 'CPU only'):
            stage.validate_pair(engine, self.before, self.after)

    def test_shell_or_different_module_cannot_replace_aggregate(self):
        self.job['argv'] = ['/bin/sh', '-c', 'arbitrary-command']
        with self.assertRaises(ValueError):
            stage.validate_pair(engine, self.before, self.after)

    def test_noop_is_not_new_stage_activation(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            stage.validate_pair(engine, self.before, copy.deepcopy(self.before))

    def test_previously_implemented_aggregate_may_not_be_replaced(self):
        self.before = copy.deepcopy(self.after)
        self.job['config_hash'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'unimplemented placeholder'):
            stage.validate_pair(engine, self.before, self.after)

    def test_uncaptured_executable_module_is_rejected_by_native_validator(self):
        self.job['source_files'].pop('src/mediflow_medvlm/campaign_recorded_first_seed.py')
        self.job['source_hash'] = engine.canonical_hash(self.job['source_files'])
        with self.assertRaisesRegex(ValueError, 'captured'):
            stage.validate_pair(engine, self.before, self.after)


if __name__ == '__main__':
    unittest.main()
