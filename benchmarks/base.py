"""Shared benchmark lifecycle."""

from __future__ import annotations

import subprocess
from abc import ABC
from abc import abstractmethod
from typing import Any

from loguru import logger

from mini_vllm import CONTROL_SOCKET
from mini_vllm import REQUEST_SOCKET
from mini_vllm import create_channel
from mini_vllm import send_shutdown
from mini_vllm import wait_for_server
from proto_loader import ProtoModules


QWEN_TARGET_MODEL = (
    'model_id: "bartowski/Qwen2.5-7B-Instruct-GGUF" '
    'model_filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf" '
    'tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"'
)
QWEN_DRAFT_MODEL = (
    'model_id: "bartowski/Qwen2.5-0.5B-Instruct-GGUF" '
    'model_filename: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" '
    'tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"'
)


class Benchmark(ABC):
    def build_server_command(self) -> list[str]:
        return [
            "cargo",
            "run",
            "--release",
            "--",
            *self.server_flags(),
            "--control-socket",
            str(CONTROL_SOCKET),
            "--request-socket",
            str(REQUEST_SOCKET),
        ]

    def run(self, process: subprocess.Popen[bytes], proto: ProtoModules) -> None:
        with create_channel(REQUEST_SOCKET) as channel:
            wait_for_server(process, channel)
            client = proto.request_handler_grpc.RequestHandlerServiceStub(channel)
            try:
                self.run_benchmark(client, proto)
            finally:
                logger.info("Benchmark finished; stopping mini-vllm-rs")
                send_shutdown(proto)

    @abstractmethod
    def server_flags(self) -> list[str]:
        pass

    @abstractmethod
    def run_benchmark(self, client: Any, proto: ProtoModules) -> None:
        pass

    @staticmethod
    def format_tokens_per_second(token_count: int, duration_us: int) -> str:
        if duration_us == 0:
            return "unavailable"
        return f"{token_count * 1_000_000 / duration_us:.2f}"

    @staticmethod
    def print_generation_stats(stats: Any) -> None:
        draft_acceptance_rate = "unavailable"
        if stats.HasField("draft_token_acceptance_rate"):
            draft_acceptance_rate = f"{stats.draft_token_acceptance_rate * 100:.1f}%"
        logger.info(
            "Generation stats:\n"
            "\tInput tokens: {}\n"
            "\tOutput tokens: {}\n"
            "\tPrefill: {} us ({} tokens/s)\n"
            "\tDecode: {} us ({} tokens/s)\n"
            "\tDraft acceptance: {}",
            stats.input_token_count,
            stats.output_token_count,
            stats.prefill_duration_microseconds,
            Benchmark.format_tokens_per_second(
                stats.input_token_count,
                stats.prefill_duration_microseconds,
            ),
            stats.decode_duration_microseconds,
            Benchmark.format_tokens_per_second(
                max(stats.output_token_count - 1, 0),
                stats.decode_duration_microseconds,
            ),
            draft_acceptance_rate,
        )
