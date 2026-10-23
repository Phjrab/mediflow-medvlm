"""CPU-only evidence checks for campaign resumes; no model or GPU is loaded."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from mediflow_medvlm.runtime import atomic_json, canonical_hash, directory_digest, file_hash
from mediflow_medvlm.training.config import load_frozen_data


SPEC = importlib.util.spec_from_file_location('medvlm_campaign', Path(__file__).resolve().parents[1] / 'scripts/complete_v1.py')
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class CampaignResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        out = self.root / 'runs' / 'fixture'
        checkpoint = out / 'checkpoints' / 'step-000010'
        checkpoint.mkdir(parents=True)
        self.config = {'run_id': 'fixture', 'pair_id': 'pair', 'model_id': 'official/model',
                       'revision': 'a' * 40, 'quantization': 'none', 'output_dir': str(out)}
        splits = {}
        for split in ('train', 'validation'):
            path = self.root / (split + '.jsonl')
            path.write_text(json.dumps({'sample_id': split, 'split': split, 'source_id': 'public', 'source_revision': 'fixed'}) + '\n')
            self.config[split + '_file'] = str(path)
            splits['text/' + split] = {'sha256': file_hash(path), 'count': 1}
        freeze = self.root / 'freeze.json'
        atomic_json(freeze, {'splits': splits})
        self.config['freeze_manifest'] = str(freeze)
        self.config_path = self.root / 'config.json'
        atomic_json(self.config_path, self.config)
        atomic_json(out / 'config.json', self.config)
        _, data = load_frozen_data(self.config)
        self.native = {'files': {'config.json': {'sha256': 'cache-file', 'bytes': 10}}, 'sha256': 'native'}
        self.identity = {'config_sha256': canonical_hash(self.config), 'data': data,
                         'model_id': self.config['model_id'], 'revision': self.config['revision'],
                         'quantization': 'none', 'template_sha256': 'template'}
        (checkpoint / 'adapter_model.safetensors').write_bytes(b'fixture-weights')
        atomic_json(checkpoint / 'checkpoint.json', {'complete': True, 'identity': self.identity,
                                                    'artifacts': directory_digest(checkpoint)})
        model = {'model_id': self.config['model_id'], 'revision': self.config['revision'],
                 'quantization': 'none', 'dtype': 'float32', 'chat_template_sha256': 'template',
                 'native_files': self.native}
        adapter = {'path': str(checkpoint), 'digest': directory_digest(checkpoint)['sha256']}
        self.manifest = {**{k: self.config[k] for k in ('run_id', 'pair_id', 'model_id', 'revision')},
                         'status': 'SUCCEEDED', 'config_hash': canonical_hash(self.config),
                         'data': data, 'identity': self.identity, 'model': model, 'adapter': adapter,
                         'training_pid': 1, 'reload_verified': True}
        atomic_json(out / 'manifest.json', self.manifest)
        self.selection = {'run_id': 'fixture', 'pair_id': 'pair', 'identity': self.identity,
                          'checkpoint': str(checkpoint), 'adapter': adapter, 'reload_verified': True}
        atomic_json(out / 'selection.json', self.selection)
        atomic_json(out / 'reload.json', {'run_id': 'fixture', 'checkpoint': str(checkpoint),
                                         'pid': 2, 'status': 'SUCCEEDED', 'fresh_process_required': True,
                                         'generated_tokens': 2, 'model': model})
        self.native_patch = patch('mediflow_medvlm.training.models.model_file_digests', return_value=self.native)
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)

    def test_verified_successes_do_not_train_or_reload_again(self):
        with patch.object(campaign, 'invoke') as invoke:
            self.assertTrue(campaign.train_stage(self.config_path, self.root))
            self.assertTrue(campaign.reload_stage(self.config_path, self.root))
            invoke.assert_not_called()

    def test_success_rejects_changed_config(self):
        changed = {**self.config, 'max_length': 123}
        with self.assertRaisesRegex(ValueError, 'config changed'):
            campaign.verify_training_cache(changed)

    def test_success_rejects_changed_frozen_file(self):
        Path(self.config['train_file']).write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'frozen split digest'):
            campaign.verify_training_cache(self.config)

    def test_success_rejects_changed_native_cache(self):
        with patch('mediflow_medvlm.training.models.model_file_digests', return_value={'sha256': 'different'}):
            with self.assertRaisesRegex(ValueError, 'native model files changed'):
                campaign.verify_training_cache(self.config)

    def test_success_rechecks_image_bytes_in_frozen_training_rows(self):
        image = self.root / 'image.png'
        image.write_bytes(b'original-image')
        row = {'task_type': 'medical_vqa', 'input': {'images': [str(image)]},
               'provenance': {'image_sha256': file_hash(image)}}
        with patch('mediflow_medvlm.training.config.load_frozen_data', return_value=({'train': [row], 'validation': []}, self.identity['data'])):
            campaign.verify_training_cache(self.config)
            image.write_bytes(b'changed-image')
            with self.assertRaisesRegex(ValueError, 'image hash changed'):
                campaign.verify_training_cache(self.config)

    def test_success_rejects_changed_adapter_bytes(self):
        (Path(self.selection['checkpoint']) / 'adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint artifact (changed|hash mismatch)'):
            campaign.verify_training_cache(self.config)

    def test_reload_success_rejects_another_checkpoint(self):
        path = Path(self.config['output_dir']) / 'reload.json'
        previous = campaign.read(path)
        previous['checkpoint'] = str(self.root / 'another-checkpoint')
        atomic_json(path, previous)
        with self.assertRaisesRegex(ValueError, 'another checkpoint'):
            campaign.reload_stage(self.config_path, self.root)

    def test_evaluation_always_checks_success_and_propagates_gate_failure(self):
        out = self.root / 'runs' / 'key-test-base-20261023'
        atomic_json(out / 'run_manifest.json', {'status': 'SUCCEEDED'})
        with patch.object(campaign, 'invoke', return_value=0) as invoke:
            self.assertEqual(campaign.evaluate_stage(self.config_path, 'protocol', self.root, 'key', 'test', 'base', self.selection), out)
            invoke.assert_called_once()
        with patch.object(campaign, 'invoke', return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'cache verification failed'):
                campaign.evaluate_stage(self.config_path, 'protocol', self.root, 'key', 'test', 'base', self.selection)

    def run_main(self, keys):
        (self.root / 'configs').mkdir(exist_ok=True)
        atomic_json(self.root / 'configs/evaluation_policy.json', {})
        previous_cwd = Path.cwd()
        previous_umask = os.umask(0o077)
        try:
            os.chdir(self.root)
            with patch.dict(sys.modules, {'huggingface_hub': types.SimpleNamespace(get_token=lambda: None)}), \
                    patch.dict(os.environ), patch.object(sys, 'argv', ['campaign', '--data-root', str(self.root), '--skip-download', '--keys', *keys]), \
                    contextlib.redirect_stdout(io.StringIO()):
                campaign.main()
        finally:
            os.chdir(previous_cwd)
            os.umask(previous_umask)
        return campaign.read(self.root / 'runtime/medvlm_state.json')

    def test_missing_cell_continues_and_prior_diagnostics_are_preserved(self):
        (self.root / 'configs').mkdir()
        atomic_json(self.root / 'configs/good-v1-20261023.json', self.config)
        atomic_json(self.root / 'configs/good-smoke-20261023.json', self.config)
        atomic_json(self.root / 'manifests/models/good.json', {})
        atomic_json(self.root / 'runtime/medvlm_state.json', {'models': {'good': {'diagnostics': {'prior': 'kept'}}, 'other': {'status': 'SUCCEEDED'}}})
        visited = []
        def stop_at_smoke(path, root):
            visited.append(Path(path).name)
            return False
        with patch.object(campaign, 'train_stage', side_effect=stop_at_smoke):
            result = self.run_main(['missing', 'good'])
        self.assertEqual(result['models']['missing']['reason_code'], 'FileNotFoundError')
        self.assertEqual(visited, ['good-smoke-20261023.json'])
        self.assertEqual(result['models']['good']['diagnostics'], {'prior': 'kept'})
        self.assertEqual(result['models']['other']['status'], 'SUCCEEDED')
        self.assertIsNone(result['active_process'])

    def test_successful_primary_skips_both_fallbacks_without_erasing_prior_evidence(self):
        atomic_json(self.root / 'runtime/medvlm_state.json', {'models': {'medgemma15_4b': {'status': 'SUCCEEDED'}, 'qwen35_2b': {'status': 'SUCCEEDED'},
                    'smolvlm_500m': {'status': 'SUCCEEDED', 'diagnostics': {'prior': 'kept'}}}})
        with patch.object(campaign, 'train_stage') as train:
            result = self.run_main(['gemma3_4b', 'smolvlm_500m'])
            train.assert_not_called()
        self.assertEqual(result['models']['gemma3_4b']['reason_code'], 'FALLBACK_NOT_NEEDED')
        self.assertEqual(result['models']['smolvlm_500m']['diagnostics'], {'prior': 'kept'})
        self.assertEqual(result['models']['smolvlm_500m']['status'], 'SUCCEEDED')
        self.assertEqual(result['models']['smolvlm_500m']['campaign_disposition'], 'FALLBACK_NOT_NEEDED')

    def test_medical_success_does_not_hide_general_vlm_failure(self):
        models={'medgemma15_4b':{'status':'SUCCEEDED'},'qwen35_2b':{'status':'BLOCKED_HARDWARE'}}
        self.assertFalse(campaign.fallback_not_needed(models,'qwen3vl_2b'))
        self.assertFalse(campaign.fallback_not_needed(models,'smolvlm_500m'))
        self.assertTrue(campaign.fallback_not_needed(models,'gemma3_4b'))
        self.assertEqual(campaign.expand_general_fallbacks(['qwen35_2b','medgemma15_4b','qwen3vl_2b']),
                         ['qwen35_2b','qwen3vl_2b','qwen25vl_3b','medgemma15_4b'])


class CampaignDownloadBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'model_id': 'official/model', 'revision': 'a' * 40, 'cache_dir': str(self.root / 'models')}
        self.gib = 1024 ** 3

    def api(self, assets):
        info = types.SimpleNamespace(sha=self.config['revision'], siblings=[types.SimpleNamespace(rfilename=n, size=s) for n, s in assets])
        client = types.SimpleNamespace(model_info=Mock(return_value=info))
        return types.SimpleNamespace(HfApi=Mock(return_value=client), snapshot_download=Mock(return_value='snapshot'),
                                     try_to_load_from_cache=Mock(return_value=None))

    def test_expected_growth_respects_reserve_and_model_budget(self):
        cases = [(15*self.gib, 30*self.gib, 100*self.gib, 'DISK_RESERVE'),
                 (2*self.gib, 31*self.gib, 200*self.gib, 'DISK_RESERVE'),
                 (41*self.gib, 100*self.gib, 200*self.gib, 'MODEL_DOWNLOAD_BUDGET')]
        for size, free, total, reason in cases:
            with self.subTest(reason=reason, total=total):
                api = self.api([('model.safetensors', size)])
                with patch.dict(sys.modules, {'huggingface_hub': api}), patch.object(campaign.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=free, total=total)):
                    with self.assertRaisesRegex(RuntimeError, reason):
                        campaign.download(self.config, {}, self.root, token='fixture-token')
                api.snapshot_download.assert_not_called()

    def test_reservation_ignores_complete_cached_and_unselected_files(self):
        cached = self.root / 'cached-config.json'
        cached.write_bytes(b'{}')
        api = self.api([('config.json', 2), ('model.safetensors', self.gib), ('ignored.bin', 100*self.gib)])
        api.try_to_load_from_cache.side_effect = lambda repo, name, **kw: str(cached) if name == 'config.json' else None
        with patch.dict(sys.modules, {'huggingface_hub': api}), patch.object(campaign.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=21*self.gib, total=100*self.gib)):
            self.assertEqual(campaign.download(self.config, {}, self.root, token='fixture-token'), 'snapshot')
        api.HfApi.assert_called_once_with(token='fixture-token')
        api.HfApi.return_value.model_info.assert_called_once_with('official/model', revision=self.config['revision'], files_metadata=True, timeout=30)
        self.assertEqual(api.snapshot_download.call_args.kwargs['max_workers'], 2)
        self.assertEqual(api.snapshot_download.call_args.kwargs['token'], 'fixture-token')

    def test_unknown_file_size_fails_before_download(self):
        api = self.api([('model.safetensors', None)])
        with patch.dict(sys.modules, {'huggingface_hub': api}):
            with self.assertRaisesRegex(ValueError, 'size metadata missing'):
                campaign.download(self.config, {}, self.root)
        api.snapshot_download.assert_not_called()
