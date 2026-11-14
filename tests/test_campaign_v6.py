"""CPU fixtures for finite planning, family barriers and immutable Hub pins."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from mediflow_medvlm.campaign_v6 import (
    CELLS, build_plan, preparation_files, resolve_pin, select_families,
)


def test_finite_matrix_and_shared_unique_preparation(tmp_path):
    plan = build_plan(tmp_path / 'data', tmp_path / 'v6')
    assert len(plan['cells']) == 10
    assert len({row['model_id'] for row in plan['cells']}) == 8
    prep = [j for j in plan['jobs'] if not j['implementation_pending']]
    assert len(prep) == 8
    assert {j['cell_id'] for j in prep} == {'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'V1', 'V4'}
    assert plan['counts']['v6_total_conditions_max'] == 18
    assert plan['counts']['v7_conditions_max'] == 4
    assert plan['seeds'] == [20261023, 20261114, 20261115]
    assert plan['pending_implementation_is_terminal'] is False
    assert all(v is None for k, v in plan['resource_policy'].items() if k != 'budget_mode')
    assert all(j['dependencies'] == ['v6-report'] for j in plan['jobs'] if j['cell_id'] in {'P0', 'P1', 'R0', 'R1'})
    assert all('argv' not in j for j in plan['jobs'] if j['implementation_pending'])


def dev_fixture():
    return {c[0]: {'status': 'SUCCEEDED', 'split': 'validation', 'full_common_dev': True,
                   'score': 0.5, 'tie_score': 0.5} for c in CELLS}


def test_selection_uses_two_different_families_and_id_tie(tmp_path):
    rows = dev_fixture()
    rows['T1']['score'] = 0.99
    rows['T2']['score'] = 0.98
    rows['T3']['score'] = 0.90
    rows['T4']['score'] = 0.95
    selected = select_families(rows, caps_frozen=True)
    assert selected['text'] == ['T1', 'T4']
    # The Gemma family has one representative, with deterministic model ID tie.
    assert len(selected['vqa']) == 2
    families = {next(c[3] for c in CELLS if c[0] == key) for key in selected['vqa']}
    assert len(families) == 2


def test_seed_and_test_barrier_requires_every_first_seed_full_dev():
    rows = dev_fixture()
    with pytest.raises(ValueError, match='frozen caps'):
        select_families(rows)
    rows.pop('T6')
    with pytest.raises(ValueError, match='all first-seed'):
        select_families(rows, caps_frozen=True)
    rows = dev_fixture()
    rows['T6']['status'] = 'PENDING_IMPLEMENTATION'
    with pytest.raises(ValueError, match='barrier'):
        select_families(rows, caps_frozen=True)
    rows = dev_fixture()
    rows['T1']['split'] = 'test'
    with pytest.raises(ValueError, match='validation'):
        select_families(rows, caps_frozen=True)


def test_revision_is_persisted_before_download_and_never_refloats(tmp_path):
    directory = tmp_path / 'model'
    directory.mkdir()
    cache = tmp_path / 'cache'
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha='a' * 40)
    pin, _ = resolve_pin('official/model', directory, cache, api)
    assert json.loads((directory / 'pin.json').read_text())['revision'] == 'a' * 40
    api.model_info.assert_called_once_with('official/model', revision=None, files_metadata=True)
    api.model_info.reset_mock()
    resolve_pin('official/model', directory, cache, api)
    api.model_info.assert_called_once_with('official/model', revision='a' * 40, files_metadata=True)
    api.model_info.return_value = SimpleNamespace(sha='b' * 40)
    with pytest.raises(ValueError, match='immutable revision'):
        resolve_pin('official/model', directory, cache, api)
    assert json.loads((directory / 'pin.json').read_text()) == pin


def test_existing_cache_revision_verified_upstream_before_reuse(tmp_path):
    directory = tmp_path / 'pin'
    directory.mkdir()
    cache = tmp_path / 'cache'
    snapshot = cache / 'models--official--model' / 'snapshots' / ('c' * 40)
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    api = Mock(model_info=Mock(return_value=SimpleNamespace(sha='c' * 40)))
    pin, _ = resolve_pin('official/model', directory, cache, api)
    assert pin['selection'] == 'existing_cached_revision_verified_upstream'
    assert api.model_info.call_args.kwargs['revision'] == 'c' * 40


def test_weight_allowlist_excludes_duplicate_formats_and_remote_code():
    names = ['config.json', 'README.md', 'LICENSE', 'tokenizer.json', 'custom.py',
             'pytorch_model.bin', 'original/model.safetensors', 'model.safetensors',
             'model.safetensors.index.json', 'model-00001-of-00001.safetensors',
             'onnx/model.onnx', 'training_args.json']
    small, weights = preparation_files(names)
    assert weights == ['model-00001-of-00001.safetensors']
    assert small == ['LICENSE', 'README.md', 'config.json', 'model.safetensors.index.json', 'tokenizer.json']
    with pytest.raises(ValueError, match='unsafe'):
        preparation_files(names + ['../escape'])


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """No network, GPU, real model imports or weight transfer in these fixtures."""
    import hashlib
    import sys
    import types
    payloads = {'config.json': b'{}', 'tokenizer_config.json': b'{}',
                'README.md': b'official model card fixture', 'LICENSE': b'fixture license',
                'model.safetensors': b'fixture weights'}
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    calls = []
    def download(model_id, *, revision, cache_dir, allow_patterns, max_workers):
        assert revision == 'a' * 40
        calls.append(list(allow_patterns))
        for name in allow_patterns:
            (snapshot / name).write_bytes(payloads[name])
        return str(snapshot)
    info = SimpleNamespace(sha='a' * 40, gated=False, card_data={'license': 'apache-2.0'},
                           siblings=[SimpleNamespace(rfilename=n, size=len(p),
                                     lfs={'sha256': hashlib.sha256(p).hexdigest()}) for n, p in payloads.items()])
    hub = types.ModuleType('huggingface_hub')
    hub.HfApi = Mock(return_value=Mock(model_info=Mock(return_value=info)))
    hub.snapshot_download = download
    hub.try_to_load_from_cache = lambda model_id, name, **kwargs: str(snapshot / name) if (snapshot / name).is_file() else None
    config = SimpleNamespace(architectures=['FixtureNative'], to_dict=lambda: {'model_type': 'fixture'})
    native = types.ModuleType('transformers')
    native.AutoConfig = Mock(from_pretrained=Mock(return_value=config))
    native.AutoTokenizer = Mock(from_pretrained=Mock(return_value=SimpleNamespace(chat_template='fixture template')))
    native.AutoProcessor = Mock(from_pretrained=Mock(return_value=SimpleNamespace(chat_template='fixture template')))
    native.FixtureNative = object()
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    monkeypatch.setitem(sys.modules, 'transformers', native)
    monkeypatch.setattr('mediflow_medvlm.campaign_v6.shutil.disk_usage', lambda p: SimpleNamespace(free=10**12))
    return calls, native


def test_preparation_receipt_binds_supervisor_and_hashes_verified_artifacts(fake_hub, monkeypatch, tmp_path):
    from mediflow_medvlm.campaign_v6 import prepare_model
    from mediflow_medvlm.runtime import file_hash
    for key in ('RUN_ID', 'SOURCE_HASH', 'CONFIG_HASH', 'DATA_HASH', 'MODEL_HASH'):
        monkeypatch.setenv('MEDVLM_SUPERVISOR_' + key, 'supervisor-' + key.lower())
    receipt = tmp_path / 'receipt.json'
    result = prepare_model('T1', tmp_path / 'data', tmp_path / 'v6', receipt)
    assert result['status'] == 'SUCCEEDED'
    assert result['run_id'] == 'supervisor-run_id'
    assert result['source_hash'] == 'supervisor-source_hash'
    assert result['model_revision'] == 'a' * 40
    assert result['training_performed'] is False
    assert result['campaign_terminal'] is False
    assert all(file_hash(a['path']) == a['sha256'] for a in result['artifacts'])
    calls, native = fake_hub
    assert calls[-1] == ['model.safetensors']
    native.AutoProcessor.from_pretrained.assert_not_called()
    assert json.loads(receipt.with_suffix('.progress.json').read_text())['optimizer_step'] == 0


def test_free_disk_guard_precedes_any_download_and_preserves_pin(fake_hub, monkeypatch, tmp_path):
    from mediflow_medvlm.campaign_v6 import prepare_model
    monkeypatch.setattr('mediflow_medvlm.campaign_v6.shutil.disk_usage', lambda p: SimpleNamespace(free=0))
    receipt = tmp_path / 'receipt.json'
    with pytest.raises(OSError, match='INSUFFICIENT_DISK'):
        prepare_model('T1', tmp_path / 'data', tmp_path / 'v6', receipt)
    assert fake_hub[0] == []
    assert list((tmp_path / 'v6' / 'models').rglob('pin.json'))
    failure = json.loads(receipt.with_suffix('.failure.json').read_text())
    assert failure['reason_code'] == 'INSUFFICIENT_DISK_BEFORE_WEIGHT_DOWNLOAD'
    assert not receipt.exists()


def test_gemma_text_and_vqa_share_same_prepared_base(fake_hub, tmp_path):
    from mediflow_medvlm.campaign_v6 import prepare_model
    text = prepare_model('T3', tmp_path / 'data', tmp_path / 'v6', tmp_path / 't3.json')
    vqa = prepare_model('V2', tmp_path / 'data', tmp_path / 'v6', tmp_path / 'v2.json')
    assert text['model_revision'] == vqa['model_revision']
    assert text['model_files_hash'] == vqa['model_files_hash']
    assert len(list((tmp_path / 'v6' / 'models').rglob('pin.json'))) == 1
    fake_hub[1].AutoProcessor.from_pretrained.assert_called()


def test_one_blocked_model_keeps_other_families_eligible_and_barriers_terminal(tmp_path):
    plan = build_plan(tmp_path / 'data', tmp_path / 'v6')
    gates = [j for j in plan['jobs'] if j['implementation_pending']]
    assert all(j['dependency_policy'] == 'terminal' for j in gates)
    rows = dev_fixture()
    rows['T6'] = {'status': 'BLOCKED_ACCESS', 'reason': 'actual HTTP 403'}
    selected = select_families(rows, caps_frozen=True)
    assert len(selected['text']) == 2
    assert 'T6' not in selected['text']
    assert len(selected['vqa']) == 2


def test_source_identity_hashes_imported_bytes_with_stable_checkout_keys():
    from mediflow_medvlm import campaign_v6
    from mediflow_medvlm.runtime import canonical_hash, file_hash
    files, digest = campaign_v6.source_identity()
    package = Path(campaign_v6.__file__).resolve().parent
    assert files
    assert all(name.startswith('src/mediflow_medvlm/') for name in files)
    assert files == {name: file_hash(package / Path(name).name) for name in files}
    assert digest == canonical_hash(files)


def test_plan_binds_explicit_snapshot_even_when_other_package_is_imported(tmp_path):
    import shutil
    from mediflow_medvlm import campaign_v6
    from mediflow_medvlm.runtime import file_hash
    package = Path(campaign_v6.__file__).resolve().parent
    snapshot = tmp_path / 'snapshot'
    files, _ = campaign_v6.source_identity()
    for name in files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / Path(name).name, target)
    changed = snapshot / 'src/mediflow_medvlm/domain.py'
    changed.write_text(changed.read_text() + '\n# snapshot fixture\n')
    plan = build_plan(tmp_path / 'data', tmp_path / 'campaign', repo=snapshot)
    job = plan['jobs'][0]
    assert job['cwd'] == str(snapshot)
    assert job['source_files']['src/mediflow_medvlm/domain.py'] == file_hash(changed)
    assert job['source_hash'] != campaign_v6.source_identity()[1]
    changed.unlink()
    with pytest.raises(FileNotFoundError):
        build_plan(tmp_path / 'data', tmp_path / 'campaign', repo=snapshot)


def test_supervisor_source_identity_does_not_compute_standalone_fallback(monkeypatch, tmp_path):
    from mediflow_medvlm import campaign_v6
    monkeypatch.setenv('MEDVLM_SUPERVISOR_SOURCE_HASH', 'bound-snapshot')
    monkeypatch.setattr(campaign_v6, 'source_identity', Mock(side_effect=AssertionError('eager fallback')))
    assert campaign_v6._identity(campaign_v6.cell_spec('T1'), tmp_path)['source_hash'] == 'bound-snapshot'
