import argparse
import json
from pathlib import Path
from .pipeline import create_run, default_config, dump, finish_run, process_episode, reviewed_decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description="双侧夹爪标签与可视化审阅；源数据只读")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="合成契约示例，不调用模型")
    demo.add_argument("--output", required=True)
    run = sub.add_parser("run", help="读取实际episode；默认只检测，--qwen才调用模型")
    run.add_argument("--dataset", required=True)
    run.add_argument("--episode", type=int, action="append", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config")
    run.add_argument("--decisions", help="包含episode_sha256及decisions的复核JSON")
    run.add_argument("--qwen", action="store_true")
    run.add_argument("--api-key-file")
    run.add_argument("--max-requests", type=int, default=3)
    sample = sub.add_parser("sample", help="建立按来源分组的审阅清单；不调用模型")
    sample.add_argument("--dataset", action="append", required=True)
    sample.add_argument("--limit", type=int, default=48)
    sample.add_argument("--output", required=True)
    batch = sub.add_parser("batch", help="按冻结清单逐episode检查一个代表事件")
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--config")
    batch.add_argument("--workers", type=int, default=4)
    batch.add_argument("--qwen", action="store_true")
    batch.add_argument("--api-key-file")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import examples
            config = default_config()
            directory = create_run(args.output)
            records = [process_episode(ep, directory, config, decisions=ds) for ep,ds in examples()]
        elif args.command == "batch":
            from .batch import run_batch
            config = json.loads(Path(args.config).read_text()) if args.config else {}
            if args.api_key_file:
                config["api_key_file"] = args.api_key_file
            run_batch(args.manifest, args.output, config=config, workers=args.workers, call_model=args.qwen)
            return 0
        elif args.command == "sample":
            from .sampling import build_manifest
            directory = create_run(args.output, args.dataset)
            manifest = build_manifest(args.dataset, args.limit)
            dump(directory / "manifest.json", manifest)
            print(f"{len(manifest)} sample candidates: {directory / 'manifest.json'}")
            return 0
        else:
            from .dataset import load_episode
            if args.max_requests < 1:
                raise ValueError("max-requests must be positive")
            if args.decisions and (args.qwen or len(args.episode) != 1):
                raise ValueError("decisions require exactly one episode and no --qwen")
            overrides = json.loads(Path(args.config).read_text()) if args.config else {}
            config = default_config(overrides)
            config["max_requests"] = args.max_requests
            if args.api_key_file:
                config["api_key_file"] = args.api_key_file
            directory = create_run(args.output, [args.dataset])
            client = None
            if args.qwen:
                from .vision import QwenClient
                client = QwenClient(config)
            envelope = json.loads(Path(args.decisions).read_text()) if args.decisions else None
            records = []
            for index in args.episode:
                print(f"Read episode {index}; source remains read-only", flush=True)
                ep = load_episode(args.dataset, index)
                decisions = reviewed_decisions(ep, envelope) if envelope is not None else None
                records.append(process_episode(ep, directory, config, decisions=decisions,
                                               client=client, candidate_limit=args.max_requests))
        finish_run(directory, records, config)
        print(f"Review: {directory / 'review.html'}")
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        # CLI errors may include credential/provider contents. Only expose the type.
        print(f"Run failed ({type(exc).__name__}); check inputs, optional dependencies and a fresh output path.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
