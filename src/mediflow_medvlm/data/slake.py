"""Official SLAKE acquisition and image-aware English conversion.

Only the author-linked Hugging Face distribution is used. Raw files are immutable;
decoded RGB pixel identity (including dimensions) is separate from byte identity.
Perceptual neighbours are conservative exclusion candidates, not asserted duplicates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from .io import _OfficialRedirectHandler, disk_guard, download

SOURCE = "slake"
REPO = "BoKelvin/SLAKE"
REVISION = "a9083ce6c34ac3ffb17671a605962924d8a8f9e9"
TRANSFORM = "slake-en-image-disjoint-v1"
OFFICIAL_URL = "https://www.med-vqa.com/slake/"
LICENSE_URL = f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md"
MAX_TREE_BYTES = 1024**2


def fetch_source_tree():
    """Pinned author metadata with the same bounded HTTPS redirect policy."""
    url=f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true"
    opener=urllib.request.build_opener(_OfficialRedirectHandler())
    for attempt in range(3):
        try:
            with opener.open(url,timeout=30) as response:
                content=response.read(MAX_TREE_BYTES+1)
            if len(content)>MAX_TREE_BYTES:
                raise ValueError("SLAKE source metadata exceeds bounded response budget")
            return json.loads(content)
        except urllib.error.HTTPError as exc:
            if exc.code in (401,403,404) or attempt==2:raise
        except (OSError,TimeoutError,http.client.IncompleteRead):
            if attempt==2:raise
        time.sleep(2**attempt)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def _safe_target(root: Path, relative: str) -> Path:
    """Reject existing symlinks which can escape a trusted destination directory."""
    relative_path = PurePosixPath(relative)
    if root.is_symlink() or relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
        raise ValueError("unsafe destination path")
    target = root / relative
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe destination symlink or path escape")
    return target


def fetch_file(url: str, destination: Path, size: int, expected_sha: str | None = None) -> dict:
    """Source-specific checks plus the common budget/redirect/resume contract."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or not parsed.path.startswith(f"/datasets/{REPO}/resolve/"):
        raise ValueError("URL outside SLAKE source allowlist")
    if size <= 0 or size > 256 * 1024**2:
        raise ValueError("source file exceeds SLAKE download budget")
    _safe_target(destination.parent, destination.name)
    _safe_target(destination.parent, destination.name + ".part")
    if destination.exists():
        digest = sha256(destination)
        if destination.stat().st_size != size or expected_sha and digest != expected_sha:
            raise ValueError(f"existing immutable file differs: {destination.name}")
        return {"path": str(destination), "url": url, "bytes": size, "sha256": digest}
    download(url, destination, max_bytes=size, expected_sha256=expected_sha)
    if destination.stat().st_size != size:
        raise ValueError("download differs from declared size")
    digest = sha256(destination)
    return {"path": str(destination), "url": url, "bytes": size, "sha256": digest}


def extract_images(archive: Path, destination: Path) -> dict:
    """Reject traversal/symlinks and expansion beyond 2 GiB before writing."""
    if destination.is_symlink():
        raise ValueError("unsafe destination symlink")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        expanded = sum(m.file_size for m in members)
        if expanded > 2 * 1024**3:
            raise ValueError("archive expansion exceeds budget")
        missing_bytes = 0
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("unsafe archive member")
            target = _safe_target(destination, member.filename)
            if not member.is_dir():
                temp = _safe_target(destination, member.filename + ".part")
                if not target.exists():
                    missing_bytes += max(0, member.file_size - (temp.stat().st_size if temp.exists() else 0))
        disk_guard(destination, missing_bytes)
        # Extract immutable members only when absent; CRC detects changed existing files.
        import zlib
        for member in members:
            target = _safe_target(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                if target.stat().st_size != member.file_size or zlib.crc32(target.read_bytes()) != member.CRC:
                    raise ValueError(f"existing extracted file differs: {member.filename}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = _safe_target(destination, member.filename + ".part")
            with z.open(member) as source, temp.open("wb") as output:
                shutil.copyfileobj(source, output)
            temp.replace(target)
    return {"archive_members": len(members), "expanded_bytes": expanded}


def image_info(path: Path) -> dict:
    from PIL import Image, ImageOps
    with Image.open(path) as raw:
        raw.load()
        image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        digest = hashlib.sha256(struct.pack(">II", width, height) + image.tobytes()).hexdigest()
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
        dhash = 0
        for y in range(8):
            for x in range(8):
                dhash = (dhash << 1) | (pixels[y*9+x] > pixels[y*9+x+1])
    return {"path": str(path.resolve()), "sha256": sha256(path), "decoded_pixel_hash": digest,
            "width": width, "height": height, "dhash64": f"{dhash:016x}", "decode_verified": True}


def build_slake(raw_root: Path, image_root: Path, revision: str, fetched_at: str) -> tuple[list[dict], dict]:
    if not revision:
        raise ValueError("resolved revision required")
    original = []
    raw_counts = {}
    for split in ("train", "validation", "test"):
        rows = json.loads((raw_root / f"{split}.json").read_text())
        raw_counts[split] = len(rows)
        for row in rows:
            original.append((split, row))
    images, invalid = {}, {}
    for _, row in original:
        name = row["img_name"]
        if name in images or name in invalid:
            continue
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            invalid[name] = "unsafe_image_path"
            continue
        path = image_root / name
        if not path.exists():
            path = image_root / "imgs" / name
        try:
            images[name] = image_info(path)
        except (OSError, ValueError) as exc:
            invalid[name] = type(exc).__name__

    parent = {name: name for name in images}
    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    exact, near = [], []
    names = sorted(images)
    for i, name in enumerate(names):
        left = images[name]
        for other in names[i+1:]:
            right = images[other]
            if left["decoded_pixel_hash"] == right["decoded_pixel_hash"]:
                exact.append([name, other])
                union(name, other)
            else:
                distance = (int(left["dhash64"], 16) ^ int(right["dhash64"], 16)).bit_count()
                # Candidate-only similarity; conservative grouping protects held-out images.
                if distance <= 4:
                    near.append({"left": name, "right": other, "distance": distance})
                    union(name, other)
    group_splits = defaultdict(set)
    direct_splits = defaultdict(set)
    image_modalities = defaultdict(set)
    for split, row in original:
        if row["img_name"] in images:
            group_splits[find(row["img_name"])].add(split)
            direct_splits[row["img_name"]].add(split)
            image_modalities[row["img_name"]].add(str(row.get("modality")))
    priority = {"train": 0, "validation": 1, "test": 2}
    owner = {group: max(splits, key=priority.get) for group, splits in group_splits.items()}
    direct_owner = {name: max(splits, key=priority.get) for name, splits in direct_splits.items()}
    excluded = Counter()
    overlap_causes = Counter()
    excluded_ids = []
    records = []
    for split, row in original:
        reason = None
        name = row["img_name"]
        if row.get("q_lang") != "en":
            reason = "not_english"
        elif name in invalid:
            reason = "image_missing_or_invalid"
        elif owner[find(name)] != split:
            reason = "heldout_image_component_overlap"
            overlap_causes["same_original_image" if direct_owner[name] != split else "additional_perceptual_candidate_component"] += 1
        elif not str(row.get("question", "")).strip() or not str(row.get("answer", "")).strip():
            reason = "empty_question_or_answer"
        if reason:
            excluded[reason] += 1
            excluded_ids.append({"original_id": str(row.get("qid")), "original_split": split, "reason": reason})
            continue
        info = images[name]
        oid = str(row["qid"])
        identity = hashlib.sha256(f"{SOURCE}/{revision}/{oid}".encode()).hexdigest()
        metadata = {key: row.get(key) for key in ("qid", "img_id", "img_name", "answer_type", "content_type", "modality", "q_lang", "location", "base_type", "triple")}
        metadata.update({"image_sha256": info["sha256"], "decoded_pixel_hash": info["decoded_pixel_hash"], "image_width": info["width"], "image_height": info["height"], "image_decode_verified": True, "image_required": True})
        records.append({"schema_version": "medvlm.dataset.v1", "sample_id": f"slake:{identity}", "source_id": SOURCE,
                        "source_revision": revision, "original_id": oid, "task_type": "medical_vqa", "language": "en",
                        "original_split": split, "split": split,
                        "input": {"question": row["question"], "context": None, "options": {}, "images": [info["path"]]},
                        "supervision": {"answer": str(row["answer"]), "option_id": None, "rationale": None, "annotation_provenance": "author_provided_SLAKE_annotation"},
                        "group_ids": {"image": f"slake:{find(name)}", "duplicate_cluster": f"slake:{find(name)}", "patient": None, "study": None, "document": None, "parent": None},
                        "grouping_quality": {"level": "image", "status": "verified", "patient_group_verified": False, "near_candidate_policy": "conservative_component_exclusion"},
                        "provenance": {"source_url": OFFICIAL_URL, "local_asset_sha256": info["sha256"], "image_sha256": info["sha256"], "decoded_pixel_hash": info["decoded_pixel_hash"], "transformation_version": TRANSFORM, "fetched_at": fetched_at, "license_ref": LICENSE_URL},
                        "metadata": metadata, "derivation": "original", "review_status": "machine_checked",
                        "restrictions": {"dataset_license": "CC-BY-4.0", "attribution_required": True, "raw_redistribution": False, "research_only": True, "sft_allowed": True}})
    # Conflicting labels for identical image+question are quarantined, never majority-voted.
    questions = defaultdict(list)
    import unicodedata
    for record in records:
        key = (record["metadata"]["decoded_pixel_hash"], " ".join(unicodedata.normalize("NFKC", record["input"]["question"]).casefold().split()))
        questions[key].append(record)
    conflicts = {r["sample_id"] for group in questions.values() if len({" ".join(r["supervision"]["answer"].casefold().split()) for r in group}) > 1 for r in group}
    for r in records:
        if r["sample_id"] in conflicts:
            excluded["conflicting_input_label"] += 1
            excluded_ids.append({"original_id": r["original_id"], "original_split": r["original_split"], "reason": "conflicting_input_label"})
    records = [r for r in records if r["sample_id"] not in conflicts]
    # Retain whole image groups when imposing the declared 5,000 train-row cap.
    groups = defaultdict(list)
    for r in records:
        if r["split"] == "train":
            groups[r["group_ids"]["image"]].append(r)
    keep, count = set(), 0
    for group in sorted(groups, key=lambda x: hashlib.sha256(f"20261023:{x}".encode()).hexdigest()):
        if count + len(groups[group]) <= 5000:
            keep.add(group)
            count += len(groups[group])
    capped = []
    for r in records:
        if r["split"] == "train" and r["group_ids"]["image"] not in keep:
            excluded["train_cap"] += 1
            excluded_ids.append({"original_id": r["original_id"], "original_split": r["original_split"], "reason": "train_cap"})
        else:
            capped.append(r)
    records = capped
    components = defaultdict(list)
    for name in images:
        components[find(name)].append(name)
    component_stats = [{"component_id": key, "image_count": len(names), "splits": sorted(group_splits[key]),
                        "modality_counts": dict(Counter(mod for name in names for mod in image_modalities[name]))}
                       for key, names in components.items()]
    component_stats.sort(key=lambda x: (-x["image_count"], x["component_id"]))
    return records, {"source_id": SOURCE, "source_revision": revision, "raw_counts": raw_counts,
                     "english_original_counts": dict(Counter(split for split, r in original if r.get("q_lang") == "en")),
                     "used_counts": dict(Counter(r["split"] for r in records)), "excluded_counts": dict(excluded), "excluded_ids": excluded_ids,
                     "image_inventory": images, "invalid_images": invalid, "verified_image_count": len(images),
                     "used_image_count": len({r["metadata"]["decoded_pixel_hash"] for r in records}),
                     "exact_pixel_duplicate_pairs": exact, "near_duplicate_candidates": near,
                     "near_duplicate_policy": {"method": "64-bit grayscale dHash", "preprocessing": "EXIF transpose; RGB; Pillow grayscale L; resize9x8 LANCZOS; adjacent-column left>right bits", "hamming_threshold": 4, "clinical_equivalence_verified": False, "action": "conservatively group all candidates; exclude lower-priority split overlap; no claim all candidates true duplicates"},
                     "overlap_exclusions_by_cause": dict(overlap_causes), "image_component_statistics": component_stats,
                     "largest_component_image_count": component_stats[0]["image_count"] if component_stats else 0,
                     "direct_original_image_cross_split_count": sum(len(splits) > 1 for splits in direct_splits.values()),
                     "cross_split_image_components": {k: sorted(v) for k, v in group_splits.items() if len(v) > 1},
                     "patient_group_verified": False, "protocol": "official_split_preserved_with_lower_priority_overlap_and_conflict_exclusion",
                     "status": "SUCCEEDED" if records and not invalid else "FAILED_REPRODUCIBLE", "evidence_level": "CPU_TESTED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    root = args.data_root.resolve()
    raw = root / "raw" / "slake" / REVISION
    image_root = root / "images" / "slake" / REVISION
    fetched_at = datetime.now(timezone.utc).isoformat()
    fetch_manifest = raw / "fetch_manifest.json"
    if args.fetch:
        tree = fetch_source_tree()
        selected = {"README.md", "train.json", "validation.json", "test.json", "imgs.zip"}
        files = []
        for entry in tree:
            if entry["path"] in selected:
                files.append(fetch_file(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry['path']}", raw / entry["path"], entry["size"], entry.get("lfs", {}).get("oid")))
        manifest = {"source_id": SOURCE, "resolved_revision": REVISION, "retrieved_at": fetched_at, "official_url": OFFICIAL_URL, "files": files,
                    "license": {"declared_dataset_license": "CC-BY-4.0", "resolved_applicable_license": "CC-BY-4.0 author-linked dataset distribution", "code_license": None, "individual_original_image_rights": "not separately enumerated in author distribution", "evidence_url": LICENSE_URL, "use": "offline noncommercial research with attribution; no raw redistribution in Git"}, "access_status": "public_author_linked_download"}
        manifest_bytes=len((json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode())
        disk_guard(raw,max(0,manifest_bytes-(fetch_manifest.stat().st_size if fetch_manifest.exists() else 0)))
        _json_write(fetch_manifest, manifest)
    else:
        manifest = json.loads(fetch_manifest.read_text())
        fetched_at = manifest["retrieved_at"]
    extraction = extract_images(raw / "imgs.zip", image_root)
    records, audit = build_slake(raw, image_root, REVISION, fetched_at)
    from .pipeline import write_source
    result = write_source(root, SOURCE, records, audit)
    manifest.update({"schema_version": "medvlm.source.v1", "acquisition_status": "SUCCEEDED", "raw_path": str(raw),
                     "total_bytes": sum(f["bytes"] for f in manifest["files"]), "declared_license": "CC-BY-4.0",
                     "resolved_applicable_license": "CC-BY-4.0 author-linked dataset distribution", "code_license": None,
                     "original_content_license": "individual underlying image rights not separately enumerated by author",
                     "license_evidence": [LICENSE_URL, OFFICIAL_URL], "raw_redistribution": False,
                     "status": audit["status"], "evidence_level": "CPU_TESTED", "actual_downloaded_rows": sum(audit["raw_counts"].values()), "raw_counts": audit["raw_counts"], "used_counts": audit["used_counts"], "excluded_counts": audit["excluded_counts"], "extraction": extraction, "canonical_files": result["canonical_files"], "canonical_counts": result["canonical_counts"]})
    _json_write(root / "manifests" / "sources" / "slake.json", manifest)
    _json_write(root / "manifests" / "slake_image_audit.json", audit)
    print(json.dumps({"source": SOURCE, "counts": audit["used_counts"], "images": audit["verified_image_count"], "invalid_images": audit["invalid_images"], "near_pairs": len(audit["near_duplicate_candidates"]), "exact_pairs": len(audit["exact_pixel_duplicate_pairs"]), "canonical_files": result["canonical_files"]}, default=str))


if __name__ == "__main__":
    main()
