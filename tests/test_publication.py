import hashlib
import json
import tarfile

import pytest

from mediflow_medvlm.publication import archive, metadata, public_prediction, safe_member, sha


def test_private_source_content_and_credentials_removed_recursively():
    private = {"config": {"model_id": "Qwen/Qwen3-4B", "target_tokens": 42,
        "api_key": "DO_NOT_PUBLISH", "input": {"question": "PRIVATE_QUESTION"},
        "supervision": {"answer": "PRIVATE_GOLD"}, "messages": ["PRIVATE_PROMPT"],
        "sample_ids": ["PRIVATE_CASE_ID"], "group_ids": ["PRIVATE_GROUP_ID"],
        "cache_dir": "/home/person/private/hub", "nested": [{"password": "PRIVATE_PASSWORD"}]} }
    public = metadata(private)
    encoded = json.dumps(public)
    assert "PRIVATE" not in encoded and "DO_NOT_PUBLISH" not in encoded
    assert "/home/" not in encoded
    assert public["config"]["target_tokens"] == 42
    assert public["config"]["model_id"] == "Qwen/Qwen3-4B"
    assert public["config"]["case_keys"] == [hashlib.sha256(b"PRIVATE_CASE_ID").hexdigest()]


def test_generated_outputs_kept_but_freeform_gold_never_exported():
    pred = {"sample_id": "dataset/image-name", "raw_output": "model response", "parsed_answer": "model response",
            "status": "answered", "question": "PRIVATE_QUESTION", "output_tokens": 3}
    score = {"sample_id": pred["sample_id"], "group_id": "private/image.jpg", "token_f1": .25,
             "reference_label": "PRIVATE_GOLD", "answer": "PRIVATE_GOLD"}
    out = public_prediction(pred, score)
    assert out["raw_output"] == "model response"
    assert out["raw_output_sha256"] == hashlib.sha256(b"model response").hexdigest()
    assert out["score"]["token_f1"] == .25
    encoded = json.dumps(out)
    assert "PRIVATE" not in encoded and "image-name" not in encoded and "image.jpg" not in encoded
    score["reference_label"] = "A"
    assert public_prediction(pred, score)["score"]["reference_label"] == "A"


def test_generated_secret_redacted_and_original_hash_retained():
    text = "hf_" + "X" * 30
    out = public_prediction({"sample_id": "x", "raw_output": text, "parsed_answer": text}, {})
    assert text not in json.dumps(out)
    assert out["output_redacted"] is True
    assert out["raw_output_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_archive_bytes_paths_and_determinism(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    (root / "adapter_model.safetensors").write_bytes(b"UNCHANGED_ADAPTER")
    (root / "LICENSE.txt").write_text("license")
    first, second = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    a = archive(root, first, {"adapter_model.safetensors"})
    b = archive(root, second, {"adapter_model.safetensors"})
    assert a["sha256"] == b["sha256"] == sha(first)
    assert a["verified_members"] == 2
    with tarfile.open(first) as tar:
        info = tar.getmember("adapter_model.safetensors")
        assert "Mediflow" in info.pax_headers["comment"]
        assert tar.extractfile(info).read() == b"UNCHANGED_ADAPTER"
    assert not safe_member("../secret") and not safe_member("/absolute")
    assert not safe_member("x\\secret") and safe_member("run/checkpoint/config.json")
    (root / "link").symlink_to(root / "LICENSE.txt")
    with pytest.raises(ValueError, match="symlink"):
        archive(root, tmp_path / "bad.tar.gz")
