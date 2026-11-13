"""Publish retained experiment artifacts without distributing source datasets.

The exporter is deliberately separate from frozen training/evaluation code. It
reads private artifacts, hashes their original bytes, and writes portable public
copies. It never changes a checkpoint, score, split, or prediction on disk.
"""
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

from .evaluation.metrics import score_rows


SCHEMA = "medvlm.public-release.v1"
BLOCKED = {
    "input", "question", "context", "options", "supervision", "answer",
    "rationale", "messages", "text", "raw_output", "parsed_answer", "labels",
    "input_ids", "attention_mask", "generated_token_ids", "instruction_text",
    "stderr", "stdout", "hostname", "host", "environment", "env",
    "questions", "answers", "contexts", "prompt", "prompts", "source_text",
    "reference_answer", "reference_text", "gold_answer",
    "images",
}
SECRET_KEYS = re.compile(r"^(?:api_key|hf_token|authorization|password|secret|access_token|refresh_token|auth_token|token)$", re.I)
SECRET_VALUE = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})")
LOCAL_PATH = re.compile(r"/(?:home|tmp|mnt|root|media)/[^\s\"'<>;,\]\}]+")
PREDICTION_FIELDS = {
    "at", "run_id", "pair_id", "input_hash", "model_identity", "attempted",
    "status", "image_condition", "vision_ingested", "pixel_tensor_keys",
    "generation_settings", "timing", "prompt_token_sha256",
    "generation_termination", "finish_reason", "output_tokens",
    "instruction_language", "maximum", "instruction_text_sha256",
}
SCORE_FIELDS = {
    "source_id", "task_type", "language", "subject", "answer_type", "modality",
    "status", "attempted", "valid", "exact_match", "token_f1",
    "finish_reason", "latency_seconds", "predicted_label",
}
NOTICE = ("HAI-DEF is provided under and subject to the Health AI Developer "
          "Foundations Terms of Use found at "
          "https://developers.google.com/health-ai-developer-foundations/terms\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def portable_string(value, roots=()):
    for path, replacement in sorted(roots, key=lambda item: -len(str(item[0]))):
        value = value.replace(str(path), replacement)
    return LOCAL_PATH.sub("<LOCAL_PATH>", SECRET_VALUE.sub("<REDACTED_CREDENTIAL>", value))


def metadata(value, roots=()):
    """Remove source-content subtrees and credentials, including nested ones."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k.lower() in BLOCKED or SECRET_KEYS.fullmatch(k):
                continue
            if k.endswith("sample_id") or k.endswith("group_id"):
                out[k.replace("sample_id", "case_key").replace("group_id", "group_key")] = key(v)
            elif k.endswith("sample_ids") or k.endswith("group_ids"):
                if not isinstance(v, list):
                    raise ValueError("identifier list has unexpected shape")
                out[k.replace("sample_ids", "case_keys").replace("group_ids", "group_keys")] = [key(x) for x in v]
            elif k == "samples" and isinstance(v, dict):
                out["cases"] = {key(sid): metadata(item, roots) for sid, item in v.items()}
            else:
                out[portable_string(k, roots)] = metadata(v, roots)
        return out
    if isinstance(value, list):
        return [metadata(v, roots) for v in value]
    if isinstance(value, str):
        return portable_string(value, roots)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def public_prediction(pred, score, roots=()):
    out = metadata({k: v for k, v in pred.items() if k in PREDICTION_FIELDS}, roots)
    out["case_key"] = key(pred["sample_id"])
    # Explicit exception: these are model generations, never dataset inputs/gold.
    for field in ("raw_output", "parsed_answer"):
        value = pred.get(field)
        out[field] = portable_string(value, roots) if isinstance(value, str) else value
    out["raw_output_sha256"] = key(pred.get("raw_output", ""))
    out["output_redacted"] = out["raw_output"] != pred.get("raw_output")
    if "generated_token_ids" in pred and not out["output_redacted"]:
        out["generated_token_ids"] = pred["generated_token_ids"]
    out["score"] = metadata({k: v for k, v in score.items() if k in SCORE_FIELDS}, roots)
    out["score"]["group_key"] = key(score.get("group_id", pred["sample_id"]))
    # Only discrete classification labels; no free-form reference text.
    ref = score.get("reference_label")
    if ref in {"A", "B", "C", "D", "E", "yes", "no", "maybe"}:
        out["score"]["reference_label"] = ref
    return out


def safe_member(name):
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name


def extra_results(data_root, result, roots, receipt):
    """Numeric audit, paired comparison and masking evidence, without inputs."""
    paths = set((data_root / "runs").glob("*comparison*.json"))
    for folder in [data_root / "v2/text", data_root / "v2/vqa", data_root / "v4"]:
        for p in folder.glob("*.json"):
            if ("-base-vs-" in p.name or "audit" in p.name or "verification" in p.name
                or "preflight" in p.name or p.name.startswith(("verified-", "ci-"))
                or p.name in {"corpus.json", "gpu_completion.json", "historical_baseline.json"}):
                paths.add(p)
    for path in sorted(paths):
        write(result / "audits_comparisons" / path.relative_to(data_root), metadata(read(path), roots))
        receipt(path)
    count = 0
    for path in sorted(data_root.rglob("mask_audit.jsonl")):
        target = result / "masking_audits" / path.relative_to(data_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        items = rows(path)
        target.write_text("".join(json.dumps(metadata(x, roots), allow_nan=False) + "\n" for x in items))
        count += len(items)
        receipt(path)
    return {"structured_audit_comparison_files": len(paths), "masking_audit_records": count}


def archive(folder, destination, modified_files=()):
    """Deterministic gzip/tar; no symlinks, unsafe names, or private mtimes."""
    with Path(destination).open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(Path(folder).rglob("*")):
                    if path.is_symlink():
                        raise ValueError("symlink in public artifact")
                    if not path.is_file():
                        continue
                    name = path.relative_to(folder).as_posix()
                    if not safe_member(name):
                        raise ValueError("unsafe archive member")
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    if path.name in modified_files:
                        info.pax_headers["comment"] = "Mediflow SFT adaptation by Hajoon Park; see MODIFICATIONS.txt and LICENSE.txt in this directory."
                    with path.open("rb") as f:
                        tar.addfile(info, f)
    # Read every byte back, not just the tar header, and check original hashes.
    verified = 0
    with tarfile.open(destination, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not safe_member(member.name):
                raise ValueError("unsafe archive")
            h = hashlib.sha256()
            with tar.extractfile(member) as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != sha(Path(folder) / member.name):
                raise ValueError("archive byte mismatch")
            verified += 1
    return {"name": Path(destination).name, "bytes": Path(destination).stat().st_size,
            "sha256": sha(destination), "verified_members": verified}


def export(data_root, repo, out, licenses):
    data_root, repo, out, licenses = map(lambda p: Path(p).resolve(), (data_root, repo, out, licenses))
    if out == data_root or data_root in out.parents:
        raise ValueError("release output must be outside private data root")
    out.mkdir(parents=True, exist_ok=True)
    result = out / "results"
    if result.exists():
        raise ValueError("output already contains results; use a new directory")
    result.mkdir()
    roots = [(data_root, "${MEDVLM_DATA_ROOT}"), (repo, "${REPO_ROOT}"),
             (repo.parent.parent, "${WORKSPACE}")]
    inventory = {"schema": SCHEMA, "retained_checkpoints": [], "evaluation_files": [],
                 "training_events": [], "source_receipts": [], "exclusions": [], "assets": []}
    inventory["exporter_sha256"] = sha(__file__)
    inventory["source_results_commit"] = "c0bd28c869abe1f7772307ec772a3e36e5bb3fd3"

    def receipt(path, reason=None):
        p = Path(path)
        if data_root in p.parents:
            relative = p.relative_to(data_root).as_posix()
        else:
            relative = portable_string(str(p), roots)
        item = {"path": relative, "bytes": p.stat().st_size, "sha256": sha(p)}
        if reason:
            item["reason"] = reason
            inventory["exclusions"].append(item)
        else:
            inventory["source_receipts"].append(item)
        return item

    # All retained predictions, including validation, failures and diagnostics.
    for pred_path in sorted(data_root.rglob("predictions.jsonl")):
        relative = pred_path.relative_to(data_root)
        predictions = rows(pred_path)
        metric_path = pred_path.with_name("metrics.json")
        if metric_path.exists():
            metrics = read(metric_path)
            scores = {p["sample_id"]: p for p in metrics["scores"]}
            if len(scores) != len(metrics["scores"]):
                raise ValueError("duplicate score")
            score_for = [scores[p["sample_id"]] for p in predictions]
            if len(scores) != len(predictions):
                raise ValueError("prediction/score count mismatch")
            write(result / "evaluations" / relative.parent / "metrics.json", metadata(metrics, roots))
            receipt(metric_path)
        else:
            if relative.parts[0] != "v5":
                raise ValueError("missing metrics outside v5")
            samples = rows(data_root / "v5/validation.jsonl")
            indexed = {}
            for lang, maximum in sorted({(p["instruction_language"], p["maximum"]) for p in predictions}):
                cell = [p for p in predictions if (p["instruction_language"], p["maximum"]) == (lang, maximum)]
                for score in score_rows(samples, cell):
                    indexed[(lang, maximum, score["sample_id"])] = score
            score_for = [indexed[(p["instruction_language"], p["maximum"], p["sample_id"])] for p in predictions]
        target = result / "evaluations" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as f:
            for pred, score in zip(predictions, score_for):
                f.write(json.dumps(public_prediction(pred, score, roots), ensure_ascii=False, allow_nan=False) + "\n")
        source = receipt(pred_path)
        inventory["evaluation_files"].append({**source, "records": len(predictions),
            "export_sha256": sha(target), "statuses": dict(Counter(p["status"] for p in predictions)),
            "stage": relative.parts[0] if relative.parts[0] != "runs" else "v1"})

    training_dirs = sorted({p.parent for p in data_root.rglob("events.jsonl") if (p.parent / "config.json").exists()})
    for folder in training_dirs:
        event_path = folder / "events.jsonl"
        event_rows = rows(event_path)
        target = result / "training" / event_path.relative_to(data_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(metadata(row, roots), allow_nan=False) + "\n" for row in event_rows))
        inventory["training_events"].append({**receipt(event_path), "records": len(event_rows),
            "event_counts": dict(Counter(p["event"] for p in event_rows))})

    # Only structured execution metadata. Never source JSONL/review/reload output.
    names = {"config.json", "run_manifest.json", "manifest.json", "selection.json",
             "reload.json", "latest.json", "best.json"}
    folders = set(training_dirs) | {p.parent for p in data_root.rglob("predictions.jsonl")}
    for folder in sorted(folders):
        for name in sorted(names):
            path = folder / name
            if path.exists():
                write(result / "metadata" / path.relative_to(data_root), metadata(read(path), roots))
                receipt(path)
    # Campaign state/plans, integrity and CI receipts contain no source rows.
    for folder in [data_root / "runtime", data_root / "v2/text", data_root / "v2/vqa", data_root / "v4", data_root / "v5"]:
        for path in sorted(folder.glob("*.json")):
            if path.name in {"gpu_budget.json", "state.json", "plan.json", "preflight.json", "verification.json", "release_verification.json"} or path.name.startswith("ci-"):
                write(result / "metadata" / path.relative_to(data_root), metadata(read(path), roots))
                receipt(path)
    workspace = repo.parent.parent
    for folder in sorted((workspace / "outputs").glob("v*")):
        for path in sorted(folder.glob("*.json")):
            if path.name.endswith(("receipt.json", "verification.json")) or path.name == "stage_status.json":
                write(result / "delivery_receipts" / folder.name / path.name, metadata(read(path), roots))
                receipt(path)

    inventory["additional_evidence"] = extra_results(data_root, result, roots, receipt)
    report_count = 0
    for p in sorted((repo / "reports/public_summary").rglob("*")):
        if p.is_file():
            target = result / "reports" / p.relative_to(repo / "reports/public_summary")
            target.parent.mkdir(parents=True, exist_ok=True)
            # These are existing public reports, preserved exactly.
            shutil.copyfile(p, target)
            report_count += 1

    model_folders = {}
    for weight in sorted(data_root.rglob("adapter_model.safetensors")):
        folder = weight.parent
        checkpoint = read(folder / "checkpoint.json")
        model = checkpoint["identity"]["model_id"]
        revision = checkpoint["identity"]["revision"]
        slug = model.split("/")[-1].lower()
        bundle = out / ("lora-" + slug)
        model_folders[model] = bundle
        rel = folder.relative_to(data_root)
        target = bundle / rel
        target.mkdir(parents=True, exist_ok=True)
        config = read(folder / "adapter_config.json")
        if config["base_model_name_or_path"] != model:
            raise ValueError("adapter base is not the canonical model id")
        selection_path = folder.parent.parent / "selection.json"
        selection = read(selection_path) if selection_path.exists() else {}
        selected = selection.get("checkpoint") == str(folder)
        smoke = "smoke" in str(rel)
        item = {"checkpoint": rel.as_posix(), "model_id": model, "base_revision": revision,
            "smoke": smoke, "selected_in_run": selected,
            "selected_for_main_evaluation": selected and not smoke,
            "optimizer_steps": checkpoint["optimizer_steps"],
            "samples_processed": checkpoint["samples_processed"],
            "supervised_tokens": checkpoint["supervised_tokens"],
            "adapter_bytes": weight.stat().st_size, "adapter_sha256": sha(weight),
            "config_sha256": sha(folder / "adapter_config.json")}
        inventory["retained_checkpoints"].append(item)
        for name in ["adapter_model.safetensors", "adapter_config.json"]:
            original = folder / name
            expected = checkpoint["artifacts"]["files"][name]
            if sha(original) != expected:
                raise ValueError("checkpoint does not match frozen artifact hash")
            shutil.copyfile(original, target / name)
            receipt(original)
        write(target / "checkpoint.json", metadata(checkpoint, roots))
        write(target / "RELEASE_METADATA.json", item)
        receipt(folder / "checkpoint.json")
        state = folder / "training_state.pt"
        if state.exists():
            receipt(state, "Optimizer/RNG training state retained privately; not LoRA weights.")
        modification = (f"Created LoRA adaptation by Hajoon Park (Mediflow local SFT).\n"
            f"Files: adapter_model.safetensors, adapter_config.json\nBase: {model}@{revision}\n"
            f"Checkpoint: {rel}\nThese are trained modifications, not upstream base weights.\n"
            "Original retained adapter bytes are preserved; this notice also accompanies each file in tar PAX metadata.\n"
            "No upstream endorsement. Research experiment; clinical validation and expert review not performed.\n")
        (target / "MODIFICATIONS.txt").write_text(modification)
        if model.startswith("google/"):
            for name in ["HAI-DEF-TERMS.txt", "HAI-DEF-PROHIBITED-USE-POLICY.txt"]:
                shutil.copyfile(licenses / name, target / name)
            (target / "Notice.txt").write_text(NOTICE)
            (target / "LICENSE.txt").write_text(
                "Mediflow MedGemma LoRA adaptation license\n\n"
                "Use, reproduction and distribution of this adaptation are licensed subject to the enclosed full HAI-DEF Terms of Use. "
                "By using or distributing this adaptation you agree to those terms. Section 3.2 use restrictions are expressly incorporated as enforceable provisions of this license, "
                "including the enclosed HAI-DEF Prohibited Use Policy, the prohibition on uses causing a Health Regulatory Authority to deem Google a medical device manufacturer, "
                "and the prohibition on uses violating applicable laws and regulations. "
                "Subsequent recipients must be notified of these restrictions and receive this license, the full terms, policy and Notice.txt. "
                "Applicable regulatory authorization requirements in Section 3.1 remain binding.\n"
                "No repository Apache license overrides these terms for the MedGemma weights.\n")
        else:
            shutil.copyfile(licenses / (model.split("/")[-1] + "-LICENSE.txt"), target / "LICENSE.txt")

    total_records = sum(p["records"] for p in inventory["evaluation_files"])
    inventory["totals"] = {"prediction_files": len(inventory["evaluation_files"]), "prediction_records": total_records,
        "training_runs": len(training_dirs), "training_events": sum(p["records"] for p in inventory["training_events"]),
        "retained_lora_checkpoints": len(inventory["retained_checkpoints"]),
        "adapter_bytes": sum(p["adapter_bytes"] for p in inventory["retained_checkpoints"]), "existing_public_report_files": report_count}
    inventory["scope"] = {
        "includes": ["all retained predictions (model generations), per-case scores, failure statuses", "training events and portable configuration/execution/reload metadata", "all retained LoRA weights including smoke and selected/latest checkpoints", "existing public reports and historical verification receipts"],
        "excludes": ["source datasets/questions/context/images/free-form references", "base pretrained weights/tokenizers/processors", "optimizer/RNG state", "private human-review packets and blind A/B keys", "secrets, arbitrary stdout/stderr and local identity/paths", "already-pruned historical checkpoints (not retained)"],
        "case_key": "SHA256 of original sample_id, consistent across runs; not a claim of irreversible anonymization",
        "prediction_records_are_not_unique_questions": True,
        "generated_outputs_are_not_medical_advice_or_gold_labels": True,
        "v5_scores": "same unchanged score_rows applied independently to each frozen validation cell; no new inference",
        "resume_limits": "Adapters support model reload. Exact training continuation requires excluded local optimizer/RNG state and separately acquired datasets/base model.",
    }
    write(result / "INVENTORY.json", {k: v for k, v in inventory.items() if k != "assets"})
    readme = ("# Retained Mediflow experiment results v1-v5\n\n"
        "Public release authorized by Hajoon Park on 2026-11-13, explicitly including LoRA weights.\n"
        "Model generations may be incorrect; no expert/clinical validation has been performed.\n"
        "See INVENTORY.json for exact scope, counts, hashes, exclusions and provenance.\n"
        "Prediction files include validation, test and diagnostics; counts are not unique questions.\n"
        "Original questions, contexts, images and free-form reference answers are not redistributed.\n"
        "Class labels and lexical scores are research metrics, not clinical accuracy.\n"
        "For trained adapters obtain the exact base model/revision stated in RELEASE_METADATA.json through its normal upstream access process.\n"
        "Load using the existing native runtime and PEFT PeftModel.from_pretrained(base_model, checkpoint_directory).\n"
        "Keep LICENSE.txt, MODIFICATIONS.txt and any HAI-DEF notices with the checkpoint.\n"
        "Smoke and nonselected checkpoints are preserved for audit, not recommended models.\n"
        "Exact runtime versions and native model/template hashes are in metadata; pyproject.toml declares dependency bounds. Recreate recorded versions when reproducing inference.\n"
        "Use sha256sum -c SHA256SUMS before extraction. Archive member names are relative and verified against retained source bytes.\n")
    (result / "README.md").write_text(readme)
    for model, bundle in sorted(model_folders.items()):
        (bundle / "README.md").write_text(readme + f"\nBase model family: {model}\n")
        write(bundle / "CHECKPOINTS.json", [i for i in inventory["retained_checkpoints"] if i["model_id"] == model])
        shutil.copyfile(licenses / "sources.json", bundle / "LICENSE_SOURCES.json")
        inventory["assets"].append(archive(bundle, out / (bundle.name + ".tar.gz"), {"adapter_model.safetensors", "adapter_config.json"}))
    inventory["assets"].append(archive(result, out / "medvlm-results-v1-v5.tar.gz"))
    write(out / "manifest.json", inventory)
    (out / "README.md").write_text(readme)
    checks = [f"{p['sha256']}  {p['name']}" for p in inventory["assets"]]
    checks += [f"{sha(out / name)}  {name}" for name in ["manifest.json", "README.md"]]
    (out / "SHA256SUMS").write_text("\n".join(checks) + "\n")
    return inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--licenses", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(export(a.data_root, a.repo, a.output, a.licenses)["totals"], indent=2))


if __name__ == "__main__":
    main()
