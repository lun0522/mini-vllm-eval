#!/usr/bin/env python3
"""Start the mini-vllm server."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from pathlib import Path


DEFAULT_REPOSITORY = Path(__file__).resolve().parent.parent / "mini-vllm-rs"
CONTROL_SOCKET = Path("/tmp/mini-vllm-main-process.sock")
SHUTDOWN_TIMEOUT_SECONDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo_path",
        type=Path,
        default=DEFAULT_REPOSITORY,
        help=f"where mini-vllm-rs is located (default: {DEFAULT_REPOSITORY})",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="use the GPU for faster model inference",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    command = ["cargo", "run", "--release"]
    if args.use_gpu:
        command.extend(["--features", "metal"])
    command.extend(["--", "--control-socket", str(CONTROL_SOCKET)])
    return command


def send_shutdown() -> None:
    try:
        import grpc
    except ImportError as error:
        raise RuntimeError(
            "gRPC is not installed. Run 'python3 -m pip install -r requirements.txt'."
        ) from error

    target = f"unix:{CONTROL_SOCKET}"
    # grpcio otherwise derives an invalid HTTP/2 authority from the Unix socket
    # path, which tonic rejects with RST_STREAM(PROTOCOL_ERROR).
    with grpc.insecure_channel(
        target,
        options=(("grpc.default_authority", "localhost"),),
    ) as channel:
        shutdown = channel.unary_unary(
            "/main_process.MainProcessService/Shutdown",
            request_serializer=lambda _request: b"",
            response_deserializer=lambda _response: None,
        )
        shutdown(None, timeout=SHUTDOWN_TIMEOUT_SECONDS)


def stop_server(process: subprocess.Popen[bytes]) -> int:
    try:
        send_shutdown()
    except Exception as error:
        try:
            return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"Could not send the shutdown command: {error}")
            print("Falling back to Ctrl-C.")
            os.killpg(process.pid, signal.SIGINT)
    return process.wait()


def main() -> int:
    args = parse_args()
    repository = args.repo_path.expanduser().resolve()
    if not (repository / "Cargo.toml").is_file():
        raise SystemExit(
            f"Could not find mini-vllm-rs at {repository}. "
            "Use --repo_path to provide its location."
        )

    print("Launching mini-vllm-rs...", flush=True)
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
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping mini-vllm-rs...")
        return stop_server(process)


if __name__ == "__main__":
    raise SystemExit(main())
