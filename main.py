#!/usr/bin/env python3
"""Start the mini-vllm server."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from cli import parse_args
from mini_vllm import build_command
from mini_vllm import send_example
from mini_vllm import stop_server
from proto_loader import generate_proto_modules


def main() -> int:
    args = parse_args()
    repository = args.repo_path.expanduser().resolve()
    if not (repository / "Cargo.toml").is_file():
        raise SystemExit(
            f"Could not find mini-vllm-rs at {repository}. "
            "Use --repo_path to provide its location."
        )

    with tempfile.TemporaryDirectory(prefix="mini-vllm-eval-proto-") as directory:
        proto = generate_proto_modules(repository, Path(directory))
        logger.info("Launching mini-vllm-rs")
        try:
            process = subprocess.Popen(
                build_command(args),
                cwd=repository,
                start_new_session=True,
            )
        except FileNotFoundError as error:
            raise SystemExit(
                "Could not start mini-vllm-rs because Cargo is not installed or is not on PATH."
            ) from error

        try:
            send_example(process, proto)
            return process.wait()
        except KeyboardInterrupt:
            logger.info("Stopping mini-vllm-rs")
            return stop_server(process, proto)
        except Exception as error:
            logger.error("Example request failed: {}", error)
            return stop_server(process, proto)


if __name__ == "__main__":
    raise SystemExit(main())
