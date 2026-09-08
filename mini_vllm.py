"""Helpers for launching and operating mini-vllm-rs."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import grpc
from loguru import logger

from proto_loader import ProtoModules


CONTROL_SOCKET = Path("/tmp/mini-vllm-main-process.sock")
REQUEST_SOCKET = Path("/tmp/mini-vllm-request-handler.sock")
SHUTDOWN_TIMEOUT_SECONDS = 5

def create_channel(socket_path: Path) -> grpc.Channel:
    # grpcio otherwise derives an invalid HTTP/2 authority from the Unix socket
    # path, which tonic rejects with RST_STREAM(PROTOCOL_ERROR).
    return grpc.insecure_channel(
        f"unix:{socket_path}",
        options=(("grpc.default_authority", "localhost"),),
    )


def send_shutdown(proto: ProtoModules) -> None:
    with create_channel(CONTROL_SOCKET) as channel:
        client = proto.main_process_grpc.MainProcessServiceStub(channel)
        client.Shutdown(
            proto.main_process.Shutdown(),
            timeout=SHUTDOWN_TIMEOUT_SECONDS,
        )


def wait_for_server(process: subprocess.Popen[bytes], channel: grpc.Channel) -> None:
    logger.info("Waiting for mini-vllm-rs to be ready")
    while process.poll() is None:
        try:
            grpc.channel_ready_future(channel).result(timeout=1)
            logger.info("mini-vllm-rs is ready")
            return
        except grpc.FutureTimeoutError:
            pass
    raise RuntimeError(f"mini-vllm-rs exited with status {process.returncode}")


def stop_server(process: subprocess.Popen[bytes], proto: ProtoModules) -> int:
    try:
        send_shutdown(proto)
    except Exception as error:
        try:
            return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error("Could not send the shutdown command: {}", error)
            logger.warning("Falling back to Ctrl-C")
            os.killpg(process.pid, signal.SIGINT)
    return process.wait()
