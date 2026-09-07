"""Helpers for launching and operating mini-vllm-rs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import grpc
from loguru import logger

from cli import DEFAULT_PAGE_SIZE
from proto_loader import ProtoModules


CONTROL_SOCKET = Path("/tmp/mini-vllm-main-process.sock")
REQUEST_SOCKET = Path("/tmp/mini-vllm-request-handler.sock")
SHUTDOWN_TIMEOUT_SECONDS = 5
EXAMPLE_PROMPT = (
    "Explain in detail how continuous batching improves throughput in an LLM "
    "inference server. Compare it with static batching, describe how requests "
    "enter and leave a running batch, and discuss the key scheduling and KV-cache "
    "challenges an implementation must handle."
)


def build_command(args: Namespace) -> list[str]:
    command = ["cargo", "run", "--release"]
    if args.use_gpu:
        command.extend(["--features", "metal"])
    cache_type = args.cache_type
    if cache_type != "contiguous":
        cache_type = f"{cache_type}:{args.page_size or DEFAULT_PAGE_SIZE}"
    command.extend(
        [
            "--",
            "--kv-cache-type",
            cache_type,
            "--control-socket",
            str(CONTROL_SOCKET),
            "--request-socket",
            str(REQUEST_SOCKET),
        ]
    )
    return command


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


def format_tokens_per_second(token_count: int, duration_ms: int) -> str:
    if duration_ms == 0:
        return "unavailable"
    return f"{token_count * 1000 / duration_ms:.2f}"


def send_example(process: subprocess.Popen[bytes], proto: ProtoModules) -> None:
    with create_channel(REQUEST_SOCKET) as channel:
        wait_for_server(process, channel)
        client = proto.request_handler_grpc.RequestHandlerServiceStub(channel)
        request = proto.request_handler.GenerateText(
            prompt=EXAMPLE_PROMPT,
            max_new_tokens=1024,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
        )
        logger.info("Sending example request:\n{}", EXAMPLE_PROMPT)
        logger.info("Response:")
        received_stats = False
        for response in client.GenerateText(request):
            event = response.WhichOneof("event")
            if event == "text":
                sys.stdout.write(response.text)
                sys.stdout.flush()
            elif event == "stats":
                received_stats = True
                stats = response.stats
                prefill_rate = format_tokens_per_second(
                    stats.input_token_count,
                    stats.prefill_duration_milliseconds,
                )
                decode_rate = format_tokens_per_second(
                    max(stats.output_token_count - 1, 0),
                    stats.decode_duration_milliseconds,
                )
                draft_rate = ""
                if stats.HasField("draft_token_acceptance_rate"):
                    draft_rate = (
                        f", draft acceptance: "
                        f"{stats.draft_token_acceptance_rate * 100:.1f}%"
                    )
                sys.stdout.write("\n")
                logger.info(
                    "Generated {} tokens (prefill: {} tokens/s, "
                    "decode: {} tokens/s{})",
                    stats.output_token_count,
                    prefill_rate,
                    decode_rate,
                    draft_rate,
                )
        if not received_stats:
            sys.stdout.write("\n")
            logger.warning("Generation ended without final statistics")


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
