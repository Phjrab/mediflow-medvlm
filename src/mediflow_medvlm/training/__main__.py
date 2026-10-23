"""Executable model probes, actual training, and separate-process reload."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m mediflow_medvlm.training")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--model-id", required=True)
    probe.add_argument("--output", required=True)
    probe.add_argument("--cache-dir")
    probe.add_argument("--tokenizer-check", action="store_true")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--resume")
    train_parser.add_argument("--migrate-legacy-from", help="verified preserved source_snapshot directory")
    reload_parser = commands.add_parser("reload")
    reload_parser.add_argument("--config", required=True)
    reload_parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        from .models import probe_model
        result = probe_model(args.model_id, args.output, cache_dir=args.cache_dir, tokenizer_check=args.tokenizer_check)
    elif args.command == "train":
        from .runner import train
        result = train(args.config, args.resume, legacy_source_snapshot=args.migrate_legacy_from)
    else:
        from .runner import reload_checkpoint
        result = reload_checkpoint(args.config, args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"SUCCEEDED", "PREPARED_NOT_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
