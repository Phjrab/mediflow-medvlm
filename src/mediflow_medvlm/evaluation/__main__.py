import argparse
import json
from .runner import freeze_protocol, evaluate, summarize, compare


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen benchmark evaluation; raw outputs stay local.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--dataset", action="append", required=True)
    p.add_argument("--output", required=True); p.add_argument("--split", default="test"); p.add_argument("--cap", type=int, default=64)
    p = sub.add_parser("evaluate"); p.add_argument("--config", required=True); p.add_argument("--protocol", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--adapter"); p.add_argument("--pair-id")
    p = sub.add_parser("summarize"); p.add_argument("--protocol", required=True); p.add_argument("--predictions", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("compare"); p.add_argument("--base", required=True); p.add_argument("--tuned", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        value = freeze_protocol(args.dataset, args.output, split=args.split, cap=args.cap)
        print(json.dumps({"protocol_hash": value["protocol_hash"], "scheduled": len(value["sample_ids"])}))
    elif args.command == "evaluate":
        value = evaluate(args.config, args.protocol, args.output_dir, adapter_path=args.adapter, pair_id=args.pair_id)
        print(json.dumps({key: value.get(key) for key in ("run_id", "status", "evidence_level", "predictions", "reason_code")}))
    elif args.command == "summarize":
        value = summarize(args.protocol, args.predictions, args.output)
        print(json.dumps({"scheduled": value["scheduled"], "predictions": value["predictions"]}))
    else:
        value = compare(args.base, args.tuned, args.output); print(json.dumps(value["comparison"]))


if __name__ == "__main__":
    main()
