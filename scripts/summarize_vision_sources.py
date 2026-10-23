"""Produce a small public aggregate; never publish questions, answers or images."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/public_summary/vision_sources.json"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    slake = json.loads((root / "manifests/slake_image_audit.json").read_text())
    sources = {}
    for source in ("slake", "vqa_rad"):
        manifest = json.loads((root / f"manifests/sources/{source}.json").read_text())
        source_summary = {key: manifest.get(key) for key in ("source_id", "official_url", "resolved_revision", "retrieved_at", "status", "evidence_level", "total_bytes", "declared_license", "resolved_applicable_license", "code_license", "original_content_license", "license_evidence", "access_status", "acquisition_status", "used_counts", "excluded_counts", "original_split_counts", "raw_counts")}
        source_summary["files"] = [{"name": Path(f["path"]).name, "bytes": f["bytes"], "sha256": f["sha256"], "url": f["url"]} for f in manifest["files"] if source == "slake" or f["path"].endswith(".json")]
        source_summary["canonical_files"] = {split: {key: value[key] for key in ("count", "sha256")} for split, value in manifest["canonical_files"].items()}
        source_summary["all_fetched_file_manifest_digest"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
        if source == "vqa_rad":
            image_manifest = json.loads((root / "manifests/vqa_rad_image_inventory.json").read_text())
            source_summary["verified_image_count"] = len(image_manifest)
            source_summary["actual_downloaded_rows"] = manifest["actual_downloaded_count"]
            source_summary["sft_allowed"] = False
        sources[source] = source_summary
    sources["slake"].update({key: slake[key] for key in ("english_original_counts", "verified_image_count", "used_image_count", "invalid_images", "overlap_exclusions_by_cause", "near_duplicate_policy", "largest_component_image_count", "direct_original_image_cross_split_count", "protocol", "patient_group_verified")})
    sources["slake"].update({"near_duplicate_candidate_pairs": len(slake["near_duplicate_candidates"]), "exact_pixel_duplicate_pairs": len(slake["exact_pixel_duplicate_pairs"]), "image_component_count": len(slake["image_component_statistics"]), "largest_components": slake["image_component_statistics"][:5], "cross_split_image_components_before_exclusion": len(slake["cross_split_image_components"])})
    slake_pixels = {r["decoded_pixel_hash"] for r in slake["image_inventory"].values()}
    vqa_pixels = {r["decoded_pixel_hash"] for r in image_manifest.values()}
    sources.update({
        "medqa": {"status": "BLOCKED_LICENSE", "evidence_level": "IMPLEMENTED_ONLY", "reason_code": "EXTERNAL_DATA_LICENSE_SCOPE_UNRESOLVED", "actual_downloaded_rows": 0,
                  "official_url": "https://github.com/jind11/MedQA", "repository_revision": "27b02f66aac217933c9648a06f82e9f720377925", "declared_repository_license": "MIT", "resolved_applicable_dataset_license": None,
                  "official_distribution_url": "https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view", "access_status": "public_archive_landing_page_accessible", "next_action": "Resolve exact US/4_options QA rights before acquiring external archive containing textbooks; then supplemental.medqa_records"},
        "medmcqa": {"status": "BLOCKED_LICENSE", "evidence_level": "CPU_TESTED", "reason_code": "EXTERNAL_DATA_LICENSE_SCOPE_UNRESOLVED", "actual_downloaded_rows": 0,
                    "official_url": "https://github.com/medmcqa/medmcqa", "repository_revision": "c59ef14ca1990266c4107c7864b45a20fd93e5e0", "hf_card_revision": "91c6572c454088bf71b679ad90aa8dffcd0d5868", "declared_repository_license": "MIT", "declared_hf_card_license": "Apache-2.0", "resolved_applicable_dataset_license": None,
                    "official_distribution_url": "https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky", "official_test_gold_public": False, "next_action": "Resolve exact artifact data-license scope; explicitly verify raw cop base and keep single choice; no official-test accuracy"},
        "aihub_71875": {"status": "BLOCKED_ACCESS", "evidence_level": "IMPLEMENTED_ONLY", "reason_code": "AUTHORIZED_DOWNLOAD_CREDENTIALS_NOT_CONFIGURED", "actual_downloaded_rows": 0,
                        "official_url": "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71875", "declared_version": "1.1", "source_includes_llm_generation": True, "per_record_expert_authorship_verified": False, "next_action": "Use existing authorized AI-Hub credentials or supplied authorized files, preserve source terms and qa_id/question/answer/q_type/domain"},
        "aihub_71762": {"status": "BLOCKED_ACCESS", "evidence_level": "CPU_TESTED", "reason_code": "AUTHORIZED_DOWNLOAD_CREDENTIALS_NOT_CONFIGURED", "actual_downloaded_rows": 0,
                        "official_url": "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71762", "declared_version": "1.1", "pairing_status": "UNRESOLVED_PAIRING", "next_action": "Obtain authorized files plus guide-defined explicit matching ID and parent ID; never join by position, disease or intent"},
        "rocov2": {"status": "PREPARED_NOT_RUN", "evidence_level": "IMPLEMENTED_ONLY", "reason_code": "REGISTRY_ONLY_V1_SCOPE", "actual_downloaded_rows": 0,
                   "official_url": "https://zenodo.org/records/10821435", "pack_record_id": "8333645", "resolved_record_id": "10821435", "version": "2.0.1", "dataset_license": "CC-BY-NC-4.0", "task": "medical_image_caption", "all_file_bytes": 6360329403, "license_metadata_bytes": 9446130, "next_action": "Separate later-scale caption campaign with per-image license_information.csv and PMCID/figure grouping"},
        "pmc_vqa": {"status": "PREPARED_NOT_RUN", "evidence_level": "IMPLEMENTED_ONLY", "reason_code": "REGISTRY_ONLY_V1_SCOPE", "actual_downloaded_rows": 0,
                    "official_url": "https://huggingface.co/datasets/RadGenome/PMC-VQA", "resolved_revision": "b56ae594f794867893143b337b4118a835794647", "dataset_license": "CC BY-SA (exact version not specified by card)", "original_article_licenses": "CC0/CC BY per author card; retain PMCID license list", "code_license": "MIT repository only", "image_archive_bytes": {"images.zip": 18945102275, "images_2.zip": 2206255503}, "next_action": "Separate later-scale QA campaign retaining train/test/test_clean and version2 artifacts, PMCID/figure/license metadata"},
    })
    output = {"schema_version": "medvlm.vision-sources.public.v1", "checked_date_asia_seoul": "2026-10-23", "sources": sources,
              "implemented_tests": {"command": "PYTHONPATH=src python -m pytest tests/test_slake.py tests/test_supplemental.py tests/test_vqarad.py -q", "passed": 13},
              "resume_commands": ["PYTHONPATH=src python -m mediflow_medvlm.data.slake --data-root ../medvlm-data --fetch", "PYTHONPATH=src python -m mediflow_medvlm.data.vqarad --data-root ../medvlm-data --fetch", "python scripts/summarize_vision_sources.py --data-root ../medvlm-data"],
              "cross_source_exact_pixel_duplicate_count": len(slake_pixels & vqa_pixels),
              "cross_source_perceptual_equivalence_review": "not performed; exact-pixel screen does not prove different patients",
              "raw_data_redistributed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "source_status": {k: v["status"] for k, v in sources.items()}}))


if __name__ == "__main__":
    main()
