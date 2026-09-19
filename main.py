#!/usr/bin/env python3
"""Run a mini-vllm benchmark."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from benchmarks import BENCHMARKS
from mini_vllm import clear_server_sockets
from mini_vllm import stop_server
from proto_loader import generate_proto_modules


DEFAULT_REPOSITORY = Path(__file__).resolve().parent.parent / "mini-vllm-rs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mini-vllm benchmark.")
    parser.add_argument(
        "--repo_path",
        type=Path,
        default=DEFAULT_REPOSITORY,
        help=f"where mini-vllm-rs is located (default: {DEFAULT_REPOSITORY})",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="benchmark to run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = BENCHMARKS.get(args.benchmark)
    if benchmark is None:
        available_benchmarks = ", ".join(BENCHMARKS)
        raise SystemExit(
            f"Unknown benchmark: {args.benchmark}. Available benchmarks: {available_benchmarks}"
        )
    repository = args.repo_path.expanduser().resolve()
    if not (repository / "Cargo.toml").is_file():
        raise SystemExit(
            f"Could not find mini-vllm-rs at {repository}. "
            "Use --repo_path to provide its location."
        )

    with tempfile.TemporaryDirectory(prefix="mini-vllm-eval-proto-") as directory:
        proto = generate_proto_modules(repository, Path(directory))
        results = []
        for case in benchmark.cases():
            clear_server_sockets()
            logger.warning("Starting benchmark case: {}", case.name)
            try:
                process = subprocess.Popen(
                    benchmark.build_server_command(case),
                    cwd=repository,
                    env={**os.environ, **dict(case.environment)},
                    start_new_session=True,
                )
            except FileNotFoundError as error:
                raise SystemExit(
                    "Could not start mini-vllm-rs because Cargo is not installed or is not on "
                    "PATH."
                ) from error

            try:
                result = benchmark.run(process, proto, case)
                exit_code = process.wait()
                if exit_code != 0:
                    raise RuntimeError(
                        f"mini-vllm-rs exited with status {exit_code} for case {case.name}"
                    )
                results.append((case, result))
            except KeyboardInterrupt:
                logger.info("Benchmark interrupted")
                return stop_server(process, proto)
            except Exception as error:
                logger.error("Benchmark case {} failed: {}", case.name, error)
                if process.poll() is None:
                    stop_server(process, proto)
                return 1

        try:
            benchmark.report_results(results)
        except Exception as error:
            logger.error("Benchmark validation failed: {}", error)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
