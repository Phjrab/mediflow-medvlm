"""Exercise existing native20-update hook using its established CPU model seam."""
import json
import os
from pathlib import Path

import pytest

from mediflow_medvlm import campaign_gpu_smoke as worker
from mediflow_medvlm import telemetry as t
from mediflow_medvlm import telemetry_probe as probe
from test_campaign_gpu_smoke import candidate, test_train_loop_twenty_updates_320_examples_and_each_adapter_changes as _exercise


def test_existing_twenty_update_smoke_emits_every_already_computed_scalar(candidate, monkeypatch, tmp_path):
    pytest.importorskip('torch')
    directory=tmp_path/'telemetry'
    manifest=t.manifest_for(candidate.config,'cpu-hook-attempt','cpu-fixture')
    lifecycle=t.TelemetryWriter(directory,manifest)
    writer=t.TelemetryWriter(directory,manifest,role='training')
    original=worker._train
    captured={}
    def observed(config,rows,staging):
        captured.update(original(config,rows,staging,telemetry=writer))
        return captured
    monkeypatch.setattr(worker,'_train',observed)
    _exercise(candidate,monkeypatch,tmp_path)
    writer.close()
    steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==20 and all(x['batch_examples']==16 for x in steps)
    assert [x['lr_used'] for x in steps]==captured['learning_rates']
    assert steps[-1]['lr_next']==0.
    assert t.reconcile(steps,captured)['status']=='passed'
    lifecycle.close()


def test_preparation_is_cpu_only_exact_source_and_same_physical_lock_paths(candidate,tmp_path,monkeypatch):
    path=tmp_path/'candidate-config.json'
    prepared=probe.prepare(candidate.prereceipt,tmp_path/'new-output','T1',path,run_id='forge-T1-candidate')
    config=json.loads(path.read_text())
    assert prepared['optimizer_steps']==20 and prepared['samples_total']==320
    assert config['supervisor_lock_paths']==candidate.config['supervisor_lock_paths']
    assert config['source_files']['src/mediflow_medvlm/telemetry.py']==t.file_hash(Path(t.__file__))
    assert config['source_files']['src/mediflow_medvlm/telemetry_probe.py']==t.file_hash(Path(probe.__file__))
    assert not (tmp_path/'new-output').exists()
    assert probe.prepare(candidate.prereceipt,tmp_path/'new-output','T1',path,run_id='forge-T1-candidate')==prepared
    with pytest.raises(ValueError,match='BOUNDED_T1_V1_PROBE_ONLY'):
        probe.prepare(candidate.prereceipt,tmp_path/'new-output','T2',path,run_id='invalid')


def test_owner_run_refuses_hash_mismatch_before_loading_config_or_locks(tmp_path):
    path=tmp_path/'config.json';path.write_text('{}')
    with pytest.raises(ValueError,match='PROBE_CONFIG_HASH_CHANGED'):
        probe.owner_run(path,tmp_path/'receipt.json',expected_sha256='0'*64)
