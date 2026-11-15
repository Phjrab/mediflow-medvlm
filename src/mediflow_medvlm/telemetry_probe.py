"""Prepare bounded native 20-update T1/V1 probes on CPU; explicit owner launch.

This is not a supervisor or queue. owner-run is exclusively for TRAINING_OWNER
following hash-bound approval: it acquires the existing two physical lock files,
keeps their descriptors inherited by existing worker children, and runs one smoke.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from . import campaign_gpu_smoke as smoke
from . import telemetry as t


def prepare(preflight_receipt, output_dir, cell_id, config_path, *, run_id):
    if cell_id not in {"T1", "V1"}: raise ValueError("BOUNDED_T1_V1_PROBE_ONLY")
    config = smoke.build_config(cell_id, preflight_receipt, output_dir, run_id=run_id)
    # Inputs and old freeze/model preparation are read-only; only this new config
    # and its chosen new output location belong to the integration candidate.
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if config_path.exists():
        if json.loads(config_path.read_text()) != config: raise ValueError("IMMUTABLE_PROBE_CONFIG_CONFLICT")
    else: t.atomic_json(config_path, config)
    return {"cell_id": cell_id, "config_sha256": t.file_hash(config_path),
        "config_hash": config["config_hash"], "source_hash": config["source_hash"], "data_hash": config["data_hash"],
        "model_hash": config["model_hash"], "optimizer_steps": 20, "samples_total": 320}


def owner_run(config_path, receipt_path, *, expected_sha256):
    """Explicit GPU operation; never called by prepare/import/tests automatically."""
    config_path = Path(config_path)
    raw = config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256: raise ValueError("PROBE_CONFIG_HASH_CHANGED")
    config = json.loads(raw)
    if config.get("cell_id") not in {"T1", "V1"} or config.get("recipe", {}).get("optimizer_steps") != 20:
        raise ValueError("BOUNDED_T1_V1_PROBE_ONLY")
    # Validate before acquiring leases; this is CPU-only and verifies exact native
    # input/config/source identity. No scheduling state is read or modified.
    smoke._validate(config)
    descriptors = []
    os.environ.pop("MEDVLM_SUPERVISOR_PROGRESS_PATH", None)
    try:
        for name in config["supervisor_lock_paths"]:
            path = Path(name)
            if path.is_symlink(): raise ValueError("PHYSICAL_EXISTING_LOCK_REQUIRED")
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)  # intentionally no O_CREAT
            if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (path.stat().st_dev, path.stat().st_ino):
                os.close(fd); raise ValueError("LOCK_INODE_CHANGED")
            descriptors.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.set_inheritable(fd, True)
        for key in smoke.IDENTITY_KEYS: os.environ["MEDVLM_SUPERVISOR_" + key.upper()] = config[key]
        return smoke.run_smoke(config_path, Path(receipt_path))
    finally:
        for fd in reversed(descriptors): os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("prepare")
    plan.add_argument("--preflight-receipt", type=Path, required=True); plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--cell-id", choices=("T1", "V1"), required=True); plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--run-id", required=True)
    run = commands.add_parser("owner-run")
    run.add_argument("--config", type=Path, required=True); run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.preflight_receipt, args.output_dir, args.cell_id, args.config, run_id=args.run_id)
    else:
        result = owner_run(args.config, args.receipt, expected_sha256=args.expected_sha256)
        result = {"status": result.get("status"), "scope": result.get("scope"), "optimizer_steps": result.get("optimizer_steps")}
    print(t.canonical(result))
    return 0


if __name__ == "__main__": raise SystemExit(main())
