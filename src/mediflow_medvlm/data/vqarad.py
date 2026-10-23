"""Small official OSF VQA-RAD acquisition; frozen external evaluation only."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from .io import atomic_json, digest, disk_guard, now, sha256
from .schema import sample
from .slake import _safe_target, image_info

NODE = "https://api.osf.io/v2/nodes/89kps/"
FILES = NODE + "files/osfstorage/"
LICENSE = "https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96c/"
OSF_OBJECT_HOST = "idvyorynw6xy.compat.objectstorage.us-ashburn-1.oraclecloud.com"
OSF_OBJECT_PREFIX = "/cos-prod-osf-1-files-ash/"
MAX_METADATA_BYTES = 4 * 1024**2


def _approved_osf_url(url, metadata_only=False):
    """Only the source node, its provider and its observed official storage."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None,443):
        return False
    if parsed.hostname == "api.osf.io":
        return parsed.path.startswith("/v2/nodes/89kps/") or parsed.path == urllib.parse.urlsplit(LICENSE).path
    if metadata_only:
        return False
    if parsed.hostname == "osf.io":
        return parsed.path.startswith("/download/")
    if parsed.hostname == "files.osf.io":
        return parsed.path.startswith("/v1/resources/89kps/providers/osfstorage/")
    return parsed.hostname == OSF_OBJECT_HOST and parsed.path.startswith(OSF_OBJECT_PREFIX)


class _OSFRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, metadata_only=False):
        super().__init__()
        self.metadata_only=metadata_only

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _approved_osf_url(newurl,self.metadata_only):
            raise ValueError("OSF redirect outside HTTPS source/provider allowlist")
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def _osf_open(url,metadata_only=False):
    if not _approved_osf_url(url,metadata_only):
        raise ValueError("OSF URL outside HTTPS source/provider allowlist")
    opener=urllib.request.build_opener(_OSFRedirectHandler(metadata_only))
    return opener.open(url,timeout=45)


def get_json(url):
    if not _approved_osf_url(url,metadata_only=True):
        raise ValueError("OSF metadata outside HTTPS source allowlist")
    for attempt in range(3):
        try:
            with _osf_open(url,metadata_only=True) as response:
                content=response.read(MAX_METADATA_BYTES+1)
            if len(content)>MAX_METADATA_BYTES:
                raise ValueError("OSF metadata exceeds bounded response budget")
            return json.loads(content)
        except urllib.error.HTTPError as exc:
            if exc.code in (401,403,404) or attempt==2:raise
        except (OSError,TimeoutError,http.client.IncompleteRead):
            if attempt==2:raise
        time.sleep(2**attempt)


def fetch_asset(entry, directory):
    attrs = entry["attributes"]
    name = attrs["name"]
    if Path(name).name != name or not 0 < attrs["size"] <= 4 * 1024**2:
        raise ValueError("invalid OSF asset name or size budget")
    url = entry["links"]["download"]
    if urllib.parse.urlparse(url).netloc != "osf.io" or not url.startswith("https://osf.io/download/"):
        raise ValueError("OSF file outside source allowlist")
    path = _safe_target(directory, name)
    part = _safe_target(directory, name + ".part")
    expected = attrs["extra"]["hashes"]["sha256"]
    if not path.exists():
        disk_guard(directory, max(0, attrs["size"] - (part.stat().st_size if part.exists() else 0)))
        for attempt in range(3):
            try:
                with _osf_open(url) as response:
                    data = response.read(attrs["size"] + 1)
                if len(data) != attrs["size"]:
                    raise OSError("OSF incomplete transfer or size mismatch")
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("OSF checksum mismatch")
                _safe_target(directory, name)
                part = _safe_target(directory, name + ".part")
                part.write_bytes(data)
                part.replace(path)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403, 404) or attempt == 2:
                    raise
                time.sleep(2**attempt)
            except (OSError, TimeoutError, http.client.IncompleteRead):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    if path.stat().st_size != attrs["size"] or sha256(path) != expected:
        raise ValueError("immutable OSF asset modified")
    return {"path": str(path), "url": url, "sha256": expected, "bytes": attrs["size"], "osf_file_id": entry["id"], "version": attrs["current_version"]}


def run(root, fetch=False):
    root = Path(root).resolve()
    raw = root / "raw" / "vqa_rad"
    source_manifest = root / "manifests" / "sources" / "vqa_rad.json"
    if fetch:
        node = get_json(NODE)
        license_record = get_json(LICENSE)
        if license_record["data"]["attributes"]["name"] != "CC0 1.0 Universal":
            raise ValueError("OSF license changed; review current source terms")
        listing = get_json(FILES)
        question_entry = next(x for x in listing["data"] if x["attributes"]["name"] == "VQA_RAD Dataset Public.json")
        folder = next(x for x in listing["data"] if x["attributes"]["kind"] == "folder")
        url = folder["relationships"]["files"]["links"]["related"]["href"] + "?page[size]=100"
        images = []
        visited_pages=set()
        while url:
            if url in visited_pages or len(visited_pages)>=4:
                raise ValueError("VQA-RAD metadata pagination exceeds 400-file scope or cycles")
            visited_pages.add(url)
            page = get_json(url)
            images.extend(page["data"])
            if len(images)>400:
                raise ValueError("VQA-RAD scope exceeds 400-file budget")
            url = page["links"]["next"]
        image_bytes = sum(e["attributes"]["size"] for e in images)
        if len(images) > 400 or image_bytes > 128 * 1024**2:
            raise ValueError("VQA-RAD scope exceeds declared small-source budget")
        # Reserve the entire pending batch before parallel requests so separate
        # workers cannot each pass a budget that the complete batch would exceed.
        missing_bytes = 0
        for entry, directory in [(question_entry, raw)] + [(entry, root / "images" / "vqa_rad") for entry in images]:
            target = _safe_target(directory, entry["attributes"]["name"])
            part = _safe_target(directory, entry["attributes"]["name"] + ".part")
            if not target.exists():
                missing_bytes += max(0, entry["attributes"]["size"] - (part.stat().st_size if part.exists() else 0))
        source_metadata = {"node": node, "license": license_record, "files": listing, "images": images}
        metadata_target = _safe_target(raw, "source_metadata.json")
        _safe_target(raw, "source_metadata.json.tmp")
        metadata_bytes = len((json.dumps(source_metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
        missing_bytes += max(0, metadata_bytes - (metadata_target.stat().st_size if metadata_target.exists() else 0))
        disk_guard(raw, missing_bytes)
        revision = "osf-89kps-" + digest([(x["id"], x["attributes"]["current_version"], x["attributes"]["extra"]["hashes"]["sha256"]) for x in [question_entry] + images])
        fetched_at = now()
        files = [fetch_asset(question_entry, raw)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            files.extend(pool.map(lambda entry: fetch_asset(entry, root / "images" / "vqa_rad"), images))
        manifest = {"schema_version": "medvlm.source.v1", "source_id": "vqa_rad", "resolved_revision": revision,
                    "official_url": "https://osf.io/89kps/", "raw_path": str(raw), "retrieved_at": fetched_at,
                    "files": files, "total_bytes": sum(x["bytes"] for x in files), "acquisition_status": "SUCCEEDED",
                    "access_status": "public_official_OSF_API_and_download", "declared_license": "CC0-1.0",
                    "resolved_applicable_license": "OSF project CC0-1.0", "code_license": None,
                    "original_content_license": "OSF project terms; source MedPix case URLs retained",
                    "license_evidence": [LICENSE, NODE], "raw_redistribution": False,
                    "roles": ["external_frozen_evaluation_candidate"], "sft_allowed": False}
        atomic_json(raw / "source_metadata.json", source_metadata)
        atomic_json(source_manifest, manifest)
    else:
        manifest = json.loads(source_manifest.read_text())
    rows = json.loads((raw / "VQA_RAD Dataset Public.json").read_text())
    images = {p.name: image_info(p) for p in (root / "images" / "vqa_rad").glob("*.jpg")}
    missing = sorted({r["image_name"] for r in rows} - images.keys())
    if missing:
        raise ValueError(f"VQA-RAD missing decoded image files: {len(missing)}")
    records = []
    for raw_row in rows:
        if raw_row["phrase_type"] not in ("test_freeform", "test_para"):
            continue
        info = images[raw_row["image_name"]]
        row = sample("vqa_rad", manifest["resolved_revision"], raw_row["qid"], "medical_vqa", raw_row["question"], str(raw_row["answer"]), "test",
                     original_split=raw_row["phrase_type"], source_url="https://osf.io/89kps/", license_ref=LICENSE,
                     fetched_at=manifest["retrieved_at"], raw_sha256=manifest["files"][0]["sha256"],
                     metadata={key: raw_row.get(key) for key in ("phrase_type", "qid_linked_id", "image_case_url", "image_name", "image_organ", "evaluation", "question_relation", "question_type", "answer_type")})
        row["input"]["images"] = [info["path"]]
        # The raw QA JSON and the referenced JPEG are distinct verified assets.
        # Keep both hashes so the shared renderer validates the actual image.
        row["provenance"].update({"image_sha256": info["sha256"],
                                  "transformation_version": "medvlm.vqarad.image-provenance.v2"})
        row["group_ids"].update({"image": "vqa_rad:" + info["decoded_pixel_hash"], "study": "vqa_rad:case:" + digest(raw_row["image_case_url"]), "parent": "vqa_rad:" + str(raw_row["qid_linked_id"])})
        row["grouping_quality"] = {"status": "verified", "level": "image_and_source_case", "patient_group_verified": False}
        row["metadata"].update({"image_sha256": info["sha256"], "decoded_pixel_hash": info["decoded_pixel_hash"], "image_decode_verified": True,
                                "image_width": info["width"], "image_height": info["height"], "image_required": True})
        row["restrictions"]["sft_allowed"] = False
        records.append(row)
    from .pipeline import write_source
    counts = dict(Counter(r["phrase_type"] for r in rows))
    stats = {"source_id": "vqa_rad", "original_split_counts": counts, "actual_downloaded_count": len(rows),
             "image_file_count": len(images), "referenced_image_count": len({r["image_name"] for r in rows}),
             "test_image_count": len({r["metadata"]["decoded_pixel_hash"] for r in records}), "decode_errors": 0,
             "protocol": "official phrase_type test_freeform/test_para only; external evaluation; no SFT",
             "excluded_counts": {"not_official_test": len(rows) - len(records)}, "status": "SUCCEEDED"}
    result = write_source(root, "vqa_rad", records, stats)
    manifest.update({"status": "SUCCEEDED", "evidence_level": "CPU_TESTED", "actual_downloaded_count": len(rows),
                     "used_counts": result["canonical_counts"], "canonical_files": result["canonical_files"], "image_file_count": len(images),
                     "original_split_counts": counts, "excluded_counts": stats["excluded_counts"]})
    atomic_json(source_manifest, manifest)
    atomic_json(root / "manifests" / "vqa_rad_image_inventory.json", images)
    return {k: v for k, v in result.items() if k != "canonical_files"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.data_root, args.fetch)))
