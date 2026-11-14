"""CPU worker contracts; real native fixtures use tiny local assets only."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mediflow_medvlm import campaign_preflight as worker
from mediflow_medvlm.campaign_supervisor import verify_success


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def candidate(tmp_path):
    model_id, revision = "Qwen/Qwen3-1.7B", "a" * 40
    cache = tmp_path / "models" / "hub"
    snapshot = cache / "models--Qwen--Qwen3-1.7B" / "snapshots" / revision
    asset = write(snapshot / "config.json", {"model_type": "qwen3"})
    provenance = {"config.json": {"sha256": worker.file_hash(asset), "bytes": asset.stat().st_size}}
    directory = tmp_path / "preparation"
    paths = [write(directory / "pin.json", {"model_id": model_id, "revision": revision}),
             write(directory / "native_assets.json", {"model_id": model_id, "revision": revision,
                   "snapshot_path": str(snapshot), "cache_dir": str(cache)}),
             write(directory / "provenance.json", {"model_id": model_id, "revision": revision, "files": provenance})]
    preparation = write(tmp_path / "prepared.json", {"status": "SUCCEEDED", "scope": "MODEL_PREPARATION_ONLY",
        "model_id": model_id, "model_revision": revision, "model_files_hash": worker.canonical_hash(provenance),
        "artifacts": [{"path": str(p), "sha256": worker.file_hash(p)} for p in paths]})
    rows = [{"sample_id": f"train-{i}", "split": "train", "source_id": "engineering_fixture",
             "language": "en", "task_type": "medical_qa", "input": {"question": "private question", "images": []},
             "supervision": {"answer": "private answer"}} for i in range(3)]
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = worker.build_config("T1", train, preparation, tmp_path / "private" / "T1", expected_count=3)
    config_path = write(tmp_path / "config.json", config)
    return SimpleNamespace(root=tmp_path, config=config, config_path=config_path, rows=rows,
                           preparation=preparation, train=train, asset=asset, receipt=tmp_path / "receipt.json")


@pytest.fixture
def native_seams(monkeypatch):
    @contextmanager
    def cpu():
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        yield SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    monkeypatch.setattr(worker, "_cpu_only", cpu)
    monkeypatch.setattr(worker, "_load_native", lambda c: (None, None, {"architecture": {"exact_target_modules": ["model.layers.0.self_attn.q_proj"]}}))
    monkeypatch.setattr(worker, "_fixtures", lambda *args: {"engineering": "VERIFIED"})
    def encode(row, *args, **kwargs):
        if row["sample_id"] == "train-1":
            raise ValueError("CONTEXT_OVERFLOW")
        return {}, {"sequence_tokens": 17, "supervised_tokens_after_shift": 3,
                    "tensor_shapes": {"input_ids": [1, 17]}, "full_target_preserved": True}
    monkeypatch.setattr(worker, "_checked_encode", encode)


def test_complete_scan_exclusion_identity_artifacts_and_exact_resume(candidate, native_seams, monkeypatch):
    c = candidate
    for key in worker.IDENTITY_KEYS:
        monkeypatch.setenv("MEDVLM_SUPERVISOR_" + key.upper(), c.config[key])
    result = worker.run_preflight(c.config_path, c.receipt)
    assert result["status"] == "SUCCEEDED"
    assert result["eligible_count"] == 2 and result["excluded_count"] == 1
    assert result["gpu_smoke_verified"] is False and result["common_train_freeze_decided"] is False
    job = {**c.config, "success_receipt": str(c.receipt)}
    assert verify_success(job)["artifact_count"] == 6
    output = Path(c.config["output_dir"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["eligible_ids"] == ["train-0", "train-2"]
    assert summary["excluded_ids"] == ["train-1"]
    assert summary["excluded_reasons"] == {"CONTEXT_OVERFLOW": 1}
    records = [json.loads(line) for line in (output / "eligibility.jsonl").read_text().splitlines()]
    assert [r["position"] for r in records] == [0, 1, 2]
    assert "private question" not in (output / "eligibility.jsonl").read_text()
    assert "private answer" not in (output / "eligibility.jsonl").read_text()
    monkeypatch.setattr(worker, "_load_native", lambda c: pytest.fail("must reuse verified success"))
    assert worker.run_preflight(c.config_path, c.receipt) == result
    assert not list(c.root.glob("**/state.json"))
    assert not list(c.root.glob("**/*failure_count*"))


def test_rejects_changed_train_before_native_import(candidate, monkeypatch):
    candidate.train.write_text(candidate.train.read_text() + "\n")
    monkeypatch.setattr(worker, "_load_native", lambda c: pytest.fail("must validate inputs first"))
    with pytest.raises(worker.PreflightError, match="TRAIN_BYTES_CHANGED"):
        worker.run_preflight(candidate.config_path, candidate.receipt)


def test_rejects_changed_native_asset(candidate):
    candidate.asset.write_text("changed")
    with pytest.raises(worker.PreflightError, match="PREPARED_MODEL_FILE_MISSING_OR_RESIZED|NATIVE_ASSET_BYTES_CHANGED"):
        worker.run_preflight(candidate.config_path, candidate.receipt)


@pytest.mark.parametrize("kind", ["config", "source", "supervisor", "incomplete", "bad_receipt", "runtime"])
def test_rejects_incompatible_or_incomplete_identity(candidate, monkeypatch, kind):
    c = candidate
    if kind == "config":
        c.config["fixed_template_date"] = "2026-11-15"
        write(c.config_path, c.config)
    elif kind == "source":
        monkeypatch.setattr(worker, "source_identity", lambda repo=None: ({}, "wrong"))
    elif kind == "supervisor":
        monkeypatch.setenv("MEDVLM_SUPERVISOR_MODEL_HASH", "wrong")
    elif kind == "incomplete":
        Path(c.config["output_dir"]).mkdir(parents=True)
    elif kind == "runtime":
        monkeypatch.setattr(worker, "runtime_versions", lambda: {"python": "wrong"})
    else:
        write(c.receipt, {"status": "FAILED"})
    with pytest.raises(worker.PreflightError):
        worker.run_preflight(c.config_path, c.receipt)


def test_rejects_tampered_success_and_preserves_candidate(candidate, native_seams):
    worker.run_preflight(candidate.config_path, candidate.receipt)
    artifact = Path(candidate.config["output_dir"]) / "summary.json"
    artifact.write_text("tampered")
    with pytest.raises(worker.PreflightError, match="ARTIFACT_CHANGED"):
        worker.run_preflight(candidate.config_path, candidate.receipt)
    assert artifact.read_text() == "tampered"


def test_atomic_directory_completion_recovers_receipt_crash_gap(candidate, native_seams, monkeypatch):
    original = worker.atomic_json
    def crash(path, value):
        if Path(path) == candidate.receipt:
            raise OSError("synthetic crash after directory rename")
        return original(path, value)
    monkeypatch.setattr(worker, "atomic_json", crash)
    with pytest.raises(OSError):
        worker.run_preflight(candidate.config_path, candidate.receipt)
    assert not candidate.receipt.exists()
    marker = Path(candidate.config["output_dir"]) / "completion.json"
    before = worker.file_hash(marker)
    monkeypatch.setattr(worker, "atomic_json", original)
    monkeypatch.setattr(worker, "_load_native", lambda c: pytest.fail("do not rescan completed output"))
    recovered = worker.run_preflight(candidate.config_path, candidate.receipt)
    assert recovered["status"] == "SUCCEEDED" and worker.file_hash(marker) == before
    assert verify_success({**candidate.config, "success_receipt": str(candidate.receipt)})["artifact_count"] == 6


@pytest.mark.parametrize("error", [worker.PreflightError("NON_CPU_PROCESSOR_TENSOR"), ValueError("image hash changed"),
                                  ValueError("unexpected native integration error"), OSError("missing private image path")])
def test_integrity_or_cpu_error_aborts_entire_scan(candidate, native_seams, monkeypatch, error):
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(worker, "_checked_encode", fail)
    with pytest.raises(worker.PreflightError):
        worker.run_preflight(candidate.config_path, candidate.receipt)
    assert not candidate.receipt.exists() and not Path(candidate.config["output_dir"]).exists()


@pytest.mark.parametrize("change", ["split", "duplicate", "axis", "count", "forbidden"])
def test_train_only_builder_rejects_invalid_partition(candidate, change):
    rows = candidate.rows
    if change == "split": rows[1]["split"] = "validation"
    elif change == "duplicate": rows[1]["sample_id"] = rows[0]["sample_id"]
    elif change == "axis": rows[1]["task_type"] = "medical_vqa"
    elif change == "forbidden": rows[1]["source_id"] = "aihub_71762"
    else: rows.pop()
    candidate.train.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(worker.PreflightError):
        worker.build_config("T1", candidate.train, candidate.preparation, candidate.root / "new", expected_count=3)


def test_native_failure_does_not_publish_partial_output(candidate, native_seams, monkeypatch):
    monkeypatch.setattr(worker, "_fixtures", lambda *args: (_ for _ in ()).throw(ValueError("private data in native error")))
    assert worker.main(["--config", str(candidate.config_path), "--receipt", str(candidate.receipt)]) == 1
    assert not candidate.receipt.exists()
    assert not Path(candidate.config["output_dir"]).exists()
    assert not list(Path(candidate.config["output_dir"]).parent.glob("*.candidate-*"))


def test_main_never_emits_native_private_errors(candidate, monkeypatch, capsys):
    def fail(*args): raise ValueError("private target and secret")
    monkeypatch.setattr(worker, "run_preflight", fail)
    assert worker.main(["--config", str(candidate.config_path), "--receipt", str(candidate.receipt)]) == 1
    message = capsys.readouterr().err
    assert "private target" not in message and "secret" not in message
    assert json.loads(message)["exception_type"] == "ValueError"


def test_classification_strata_allow_bounded_labels_but_no_freeform_answers():
    assert worker._stratum({"task_type": "context_qa", "supervision": {"answer": "Maybe"}})["classification_label"] == "maybe"
    assert worker._stratum({"input": {"options": {"A": "private option"}}, "supervision": {"option_id": "A"}})["classification_label"] == "A"
    for row in ({"task_type": "medical_qa", "supervision": {"answer": "private freeform answer"}},
                {"task_type": "context_qa", "supervision": {"answer": "private freeform answer"}},
                {"input": {"options": ["private option"]}, "supervision": {"option_id": "private freeform answer"}}):
        assert "private" not in json.dumps(worker._stratum(row))


def tiny_processor():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    vocab = {word: i for i, word in enumerate(["[UNK]", "<|user|>", "<|assistant|>", "<|end|>", "ready", "start"])}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = WhitespaceSplit()
    processor = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="<|end|>", pad_token="<|end|>")
    processor.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|> ' + message['content'] + ' <|end|> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
    return processor


def test_actual_native_encoding_multiturn_empty_overflow_and_eos_pad(tmp_path):
    processor = tiny_processor()
    with worker._cpu_only():
        result = worker._fixtures(processor, {}, "text", tmp_path)
    assert result["single_turn"]["prompt_image_pad_mask_verified"]
    assert result["multiturn_last_assistant_only"]["assistant_start"] > result["single_turn"]["assistant_start"]
    assert result["padding"]["eos_equals_pad"]
    assert result["empty_target"] == result["overflow"] == "EXPECTED_REJECTION_VERIFIED"


def test_target_containing_masked_reserved_image_token_is_not_claimed_preserved():
    processor = tiny_processor()
    processor.add_special_tokens({"additional_special_tokens": ["<image>"]})
    row = {"sample_id": "reserved-target", "task_type": "medical_qa",
           "input": {"question": "Return ready.", "images": []},
           "supervision": {"answer": "<image> ready"}}
    with worker._cpu_only(), pytest.raises(worker.PreflightError, match="TARGET_CONTAINS_MASKED"):
        worker._checked_encode(row, processor, {}, "text", None)


def test_overflow_keeps_intact_native_length_and_target_measurement():
    processor = tiny_processor()
    row = {"sample_id": "long-target", "task_type": "medical_qa",
           "input": {"question": "Return ready.", "images": []},
           "supervision": {"answer": "ready " * 3000}}
    with worker._cpu_only(), pytest.raises(worker.IneligibleRow) as error:
        worker._checked_encode(row, processor, {}, "text", None)
    assert error.value.native["sequence_tokens"] > 3000
    assert error.value.native["supervised_tokens_after_shift"] == 3001
    assert error.value.native["full_target_preserved"]


def test_cuda_initialized_runtime_is_rejected(monkeypatch):
    fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    with pytest.raises(worker.PreflightError, match="CUDA_ALREADY_INITIALIZED"):
        with worker._cpu_only():
            pytest.fail("must not reach native work")


def test_worker_cli_requires_absolute_paths():
    with pytest.raises(worker.PreflightError, match="ABSOLUTE_PATH_REQUIRED"):
        worker.run_preflight("relative.json", "/tmp/receipt.json")
