import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image

from mediflow_medvlm.data.slake import build_slake, extract_images, fetch_file, image_info


def write_rows(root, split, rows):
    (root / f"{split}.json").write_text(json.dumps(rows))


def qa(qid, name, question="What is shown?", answer="lung"):
    return {"qid": qid, "img_id": name, "img_name": name, "question": question,
            "answer": answer, "q_lang": "en", "answer_type": "OPEN", "content_type": "Organ"}


def test_decode_identity_and_heldout_preservation(tmp_path):
    # PNG metadata/compression differs but the decoded pixels are identical.
    image = Image.new("RGB", (18, 16))
    image.putdata([((i * 31) % 256, (i * 71) % 256, (i * 119) % 256) for i in range(288)])
    image.save(tmp_path / "train.png", compress_level=0)
    image.save(tmp_path / "test.png", compress_level=9)
    a, b = image_info(tmp_path / "train.png"), image_info(tmp_path / "test.png")
    assert a["sha256"] != b["sha256"]
    assert a["decoded_pixel_hash"] == b["decoded_pixel_hash"]
    write_rows(tmp_path, "train", [qa(1, "train.png", "A training question")])
    write_rows(tmp_path, "validation", [])
    write_rows(tmp_path, "test", [qa(2, "test.png", "A held-out question")])
    records, audit = build_slake(tmp_path, tmp_path, "resolved-revision", "2026-10-23")
    assert [r["original_id"] for r in records] == ["2"]
    assert records[0]["split"] == "test"
    assert audit["excluded_counts"]["heldout_image_component_overlap"] == 1
    assert audit["exact_pixel_duplicate_pairs"]


def test_missing_and_corrupt_images_never_become_ready(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    write_rows(tmp_path, "train", [qa(1, "missing.png"), qa(2, "bad.png")])
    write_rows(tmp_path, "validation", [])
    write_rows(tmp_path, "test", [])
    rows, audit = build_slake(tmp_path, tmp_path, "revision", "2026-10-23")
    assert not rows
    assert len(audit["invalid_images"]) == 2
    assert audit["status"] == "FAILED_REPRODUCIBLE"


def test_image_conditioned_input_conflicts_are_quarantined(tmp_path):
    Image.new("RGB", (10, 10), "red").save(tmp_path / "image.png")
    write_rows(tmp_path, "train", [qa(1, "image.png", answer="yes"), qa(2, "image.png", answer="no")])
    write_rows(tmp_path, "validation", [])
    write_rows(tmp_path, "test", [])
    rows, audit = build_slake(tmp_path, tmp_path, "revision", "2026-10-23")
    assert not rows
    assert audit["excluded_counts"]["conflicting_input_label"] == 2


def test_zip_traversal_rejected_before_any_extraction(tmp_path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("okay.png", b"example")
        z.writestr("../outside", b"bad")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="unsafe archive"):
        extract_images(archive, output)
    assert not (output / "okay.png").exists()


def test_existing_extraction_symlink_cannot_escape_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "images"
    output.mkdir()
    (output / "linked").symlink_to(outside, target_is_directory=True)
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("okay.jpg", b"safe")
        z.writestr("linked/escape.jpg", b"must not write")
    with pytest.raises(ValueError, match="unsafe destination"):
        extract_images(archive, output)
    assert not (output / "okay.jpg").exists()
    assert not (outside / "escape.jpg").exists()


def test_extraction_checks_remaining_budget_before_writes(tmp_path, monkeypatch):
    from mediflow_medvlm.data import slake
    output = tmp_path / "images"
    output.mkdir()
    (output / "existing.jpg").write_bytes(b"old")
    (output / "new.jpg.part").write_bytes(b"n")
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("existing.jpg", b"old")
        z.writestr("new.jpg", b"new-image")
    calls = []
    def denied(root, expected_bytes):
        calls.append((root, expected_bytes))
        raise RuntimeError("SKIPPED_BUDGET: test reserve")
    monkeypatch.setattr(slake, "disk_guard", denied)
    with pytest.raises(RuntimeError, match="SKIPPED_BUDGET"):
        extract_images(archive, output)
    assert calls == [(output, len(b"new-image") - len(b"n"))]
    assert not (output / "new.jpg").exists()
    assert (output / "new.jpg.part").read_bytes() == b"n"
    assert (output / "existing.jpg").read_bytes() == b"old"


@pytest.mark.parametrize("filename", ["train.json", "train.json.part"])
def test_slake_download_rejects_existing_symlink_before_transfer(tmp_path, monkeypatch, filename):
    from mediflow_medvlm.data import slake
    outside = tmp_path / "outside"
    outside.write_bytes(b"retain")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / filename).symlink_to(outside)
    def unexpected(*args, **kwargs):
        pytest.fail("download must not begin")
    monkeypatch.setattr(slake, "download", unexpected)
    with pytest.raises(ValueError, match="unsafe destination"):
        fetch_file("https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/revision/train.json", raw / "train.json", 6)
    assert outside.read_bytes() == b"retain"


def test_source_metadata_is_bounded_and_uses_fixed_pinned_url(monkeypatch):
    from types import SimpleNamespace
    from mediflow_medvlm.data import slake
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self,n):return b'[]'
    calls=[]
    def open_(url,timeout):calls.append((url,timeout));return Response()
    monkeypatch.setattr(slake.urllib.request,'build_opener',lambda *_:SimpleNamespace(open=open_))
    assert slake.fetch_source_tree()==[]
    assert calls==[(f'https://huggingface.co/api/datasets/BoKelvin/SLAKE/tree/{slake.REVISION}?recursive=true',30)]
    monkeypatch.setattr(slake,'MAX_TREE_BYTES',1)
    with pytest.raises(ValueError,match='bounded response'):slake.fetch_source_tree()


def test_source_metadata_does_not_retry_auth_denial(monkeypatch):
    from types import SimpleNamespace
    import urllib.error
    from mediflow_medvlm.data import slake
    calls=[]
    def denied(*a,**k):calls.append(1);raise urllib.error.HTTPError('https://huggingface.co',403,'denied',{},None)
    monkeypatch.setattr(slake.urllib.request,'build_opener',lambda *_:SimpleNamespace(open=denied))
    with pytest.raises(urllib.error.HTTPError):slake.fetch_source_tree()
    assert len(calls)==1
