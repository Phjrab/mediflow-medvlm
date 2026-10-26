"""No-image diagnostics cannot silently change inputs or include vision/gold."""
import base64
import copy
import json
from types import SimpleNamespace

import pytest

from mediflow_medvlm.evaluation.image_diagnostics import (
    UnsupportedImageDiagnostic, compare_image_conditions, derive_no_image_protocol,
    encode_no_image_sample, unsupported_no_image_prediction,
)
from mediflow_medvlm.evaluation.runner import freeze_protocol, load_protocol
from mediflow_medvlm.runtime import canonical_hash, file_hash
from mediflow_medvlm.training.rendering import user_text


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path, count=4):
    image = tmp_path / "fixture.png"
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV7cAAAAASUVORK5CYII="))
    rows = [{"sample_id": str(i), "source_id": "fixture", "task_type": "medical_vqa",
             "language": "en", "split": "test", "input": {"question": "Image fixture question " + str(i), "images": [str(image)]},
             "supervision": {"answer": "GOLD_DO_NOT_RENDER"}, "group_ids": {"image": "same-image"},
             "metadata": {"answer_type": "CLOSED", "modality": "CT", "rationale": "DO_NOT_RENDER_EITHER"}}
            for i in range(count)]
    data = tmp_path / "test.jsonl"; write_rows(data, rows)
    parent_path = tmp_path / "protocol.json"
    parent = freeze_protocol([data], parent_path, cap=0)
    return parent, parent_path, rows


class Processor:
    def __init__(self, extra=None):
        self.extra = extra or {}

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.template_kwargs = messages, kwargs
        return json.dumps(messages)

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": SimpleNamespace(shape=(1, 8)), "attention_mask": SimpleNamespace(shape=(1, 8)), **self.extra}


def run_files(root, protocol, *, no_image=False, all_rows=True):
    root.mkdir()
    manifest = {"run_id": root.name, "pair_id": "pair", "identity_hash": root.name + "-identity",
                "protocol_hash": protocol["protocol_hash"], "model_id": "fixture/model", "model_revision": "fixed",
                "config_hash": "config", "precision": "none", "adapter_path": None,
                "model": {"native_files": {"sha256": "native-base"},
                          "chat_template_sha256": "native-template", "dtype": "bfloat16"},
                "generation": {"do_sample": False, "budgets": protocol["max_new_tokens"]}}
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    predictions = []
    for sid in protocol["sample_ids"] if all_rows else []:
        predictions.append({"sample_id": sid, "input_hash": protocol["input_digests"][sid],
                            "run_id": manifest["run_id"], "pair_id": "pair", "model_identity": manifest["identity_hash"],
                            "attempted": True, "status": "answered", "parsed_answer": "GOLD_DO_NOT_RENDER",
                            "image_condition": "no_image" if no_image else "image", "vision_ingested": not no_image,
                            "pixel_tensor_keys": [] if no_image else ["pixel_values"]})
    path = root / "predictions.jsonl"; write_rows(path, predictions)
    return path, manifest, predictions


def test_derived_protocol_retains_original_asset_identity_and_fixed_ids(tmp_path):
    parent, path, rows = fixture(tmp_path)
    before = file_hash(path)
    derived_path = tmp_path / "no-image.json"
    derived = derive_no_image_protocol(path, derived_path, cap=2)
    expected = sorted((r["sample_id"] for r in rows), key=lambda sid: canonical_hash([20261023, sid]))[:2]
    assert derived["sample_ids"] == expected
    assert derived["files"] == parent["files"]
    assert derived["parent_reference_digest"] == parent["reference_digest"]
    assert derived["protocol_hash"] != parent["protocol_hash"]
    assert derived["image_condition"] == "no_image"
    for sid in expected:
        assert derived["image_files"][sid] == parent["image_files"][sid]
        assert derived["input_digests"][sid] == parent["input_digests"][sid]
    assert file_hash(path) == before
    assert derive_no_image_protocol(path, derived_path, cap=2) == derived
    assert [r["sample_id"] for r in load_protocol(derived_path)[1]] == expected
    with pytest.raises(ValueError, match="overwrite"):
        derive_no_image_protocol(path, derived_path, cap=1)
    asset = tmp_path / "fixture.png"; asset.write_bytes(asset.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="frozen image bytes changed"):
        load_protocol(derived_path)


@pytest.mark.parametrize("cap", [0, 33, -1, 1.5, True])
def test_unbounded_no_image_protocol_rejected(tmp_path, cap):
    with pytest.raises(ValueError, match="cap"):
        derive_no_image_protocol(tmp_path / "absent.json", tmp_path / "output.json", cap=cap)


def test_native_no_image_encoding_sends_same_text_without_gold_or_image(tmp_path):
    _, _, rows = fixture(tmp_path)
    sample = copy.deepcopy(rows[0]); sample["input"]["images"] = ["/does/not/exist.png"]
    processor = Processor()
    encoded, meta = encode_no_image_sample(sample, processor)
    assert set(encoded) == {"input_ids", "attention_mask"}
    assert processor.messages == [{"role": "user", "content": [{"type": "text", "text": user_text(sample)}]}]
    assert processor.template_kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert set(processor.kwargs) == {"text", "return_tensors", "padding", "add_special_tokens"}
    assert "GOLD_DO_NOT_RENDER" not in processor.kwargs["text"][0]
    assert "DO_NOT_RENDER_EITHER" not in processor.kwargs["text"][0]
    assert meta["images"] == [] and not meta["vision_ingested"]
    assert meta["native_generation_supported"] == "not_yet_tested"
    assert meta["input_hash"] == canonical_hash(sample["input"])


@pytest.mark.parametrize("key", ["pixel_values", "image_grid_thw", "pixel_attention_mask", "unrecognized_tensor"])
def test_native_no_image_rejects_unexpected_tensor_even_when_empty(tmp_path, key):
    _, _, rows = fixture(tmp_path)
    with pytest.raises(UnsupportedImageDiagnostic, match="UNEXPECTED_NO_IMAGE_TENSORS"):
        encode_no_image_sample(rows[0], Processor({key: None}))


@pytest.mark.parametrize("all_zero,shape,accepted", [(True, (1, 8), True), (False, (1, 8), False), (True, (1, 7), False)])
def test_native_modality_mask_must_be_zero_and_token_aligned(tmp_path, all_zero, shape, accepted):
    class Mask:
        def __eq__(self, other):
            assert other == 0
            return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: all_zero))
    mask = Mask(); mask.shape = shape
    _, _, rows = fixture(tmp_path)
    processor = Processor({"mm_token_type_ids": mask})
    if accepted:
        encoded, _ = encode_no_image_sample(rows[0], processor)
        assert encoded["mm_token_type_ids"] is mask
    else:
        with pytest.raises(UnsupportedImageDiagnostic, match="NON_TEXT_NO_IMAGE_MODALITY_MASK"):
            encode_no_image_sample(rows[0], processor)


def test_no_image_never_falls_back_on_processor_failure(tmp_path):
    class Broken(Processor):
        def __call__(self, **kwargs):
            raise RuntimeError("sensitive source text")
    _, _, rows = fixture(tmp_path)
    with pytest.raises(UnsupportedImageDiagnostic, match="^NATIVE_NO_IMAGE_PROCESSOR_UNSUPPORTED:RuntimeError$"):
        encode_no_image_sample(rows[0], Broken())
    with pytest.raises(ValueError, match="cannot train"):
        encode_no_image_sample(rows[0], Processor(), include_labels=True)
    with pytest.raises(UnsupportedImageDiagnostic, match="CONTEXT_OVERFLOW"):
        encode_no_image_sample(rows[0], Processor(), max_length=7)


def test_matched_report_retains_failures_and_uses_image_groups(tmp_path):
    parent, path, _ = fixture(tmp_path, count=3)
    derived_path = tmp_path / "no-image.json"; derived = derive_no_image_protocol(path, derived_path, cap=3)
    image_path, _, _ = run_files(tmp_path / "image", parent)
    no_image_path, identity, predictions = run_files(tmp_path / "noimage", derived, no_image=True)
    unsupported = unsupported_no_image_prediction(derived["sample_ids"][1], derived)
    unsupported.update(run_id=identity["run_id"], pair_id="pair", model_identity=identity["identity_hash"])
    write_rows(no_image_path, [predictions[0], unsupported])  # Third slot remains not_attempted.
    result = compare_image_conditions(path, derived_path, image_path, no_image_path, tmp_path / "result.json", resamples=20)
    cohort = result["no_image"][0]
    assert cohort["scheduled"] == 3 and cohort["valid_answers"] == 1
    assert cohort["statuses"] == {"answered": 1, "unsupported_input": 1, "not_attempted": 1}
    assert result["unsupported_ablation_count"] == 1
    pair = result["paired_image_minus_no_image"][0]
    assert pair["delta"] == pytest.approx(2 / 3) and pair["group_count"] == 1
    assert result["human_review"] == "not_performed" and result["clinical_accuracy"] is None


@pytest.mark.parametrize("mutation", ["adapter", "model", "generation", "condition", "pixels",
                                      "native_files", "chat_template_sha256", "dtype", "missing_loaded_model"])
def test_report_rejects_unmatched_conditions(tmp_path, mutation):
    parent, path, _ = fixture(tmp_path, count=1)
    derived_path = tmp_path / "no-image.json"; derived = derive_no_image_protocol(path, derived_path, cap=1)
    image_path, _, _ = run_files(tmp_path / "image", parent)
    no_image_path, manifest, predictions = run_files(tmp_path / "noimage", derived, no_image=True)
    if mutation == "adapter": manifest["adapter_digest"] = "different-adapter"
    elif mutation == "model": manifest["model_revision"] = "different-revision"
    elif mutation == "generation": manifest["generation"]["do_sample"] = True
    elif mutation == "condition": predictions[0].pop("image_condition")
    elif mutation == "pixels": predictions[0]["pixel_tensor_keys"] = ["pixel_values"]
    elif mutation == "native_files": manifest["model"]["native_files"]["sha256"] = "other-base"
    elif mutation in {"chat_template_sha256", "dtype"}: manifest["model"][mutation] = "different"
    elif mutation == "missing_loaded_model": manifest.pop("model")
    (no_image_path.parent / "run_manifest.json").write_text(json.dumps(manifest))
    write_rows(no_image_path, predictions)
    with pytest.raises(ValueError):
        compare_image_conditions(path, derived_path, image_path, no_image_path, tmp_path / "result.json", resamples=20)
