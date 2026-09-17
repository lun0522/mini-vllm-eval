"""Shared benchmark lifecycle."""

from __future__ import annotations

import subprocess
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from loguru import logger

from mini_vllm import CONTROL_SOCKET
from mini_vllm import REQUEST_SOCKET
from mini_vllm import create_channel
from mini_vllm import send_shutdown
from mini_vllm import wait_for_server
from proto_loader import ProtoModules


QWEN_LARGE_MODEL = (
    'model_id: "bartowski/Qwen2.5-7B-Instruct-GGUF" '
    'model_filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf" '
    'tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"'
)
QWEN_SMALL_MODEL = (
    'model_id: "bartowski/Qwen2.5-0.5B-Instruct-GGUF" '
    'model_filename: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" '
    'tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"'
)
LLAMA_LARGE_MODEL = (
    'model_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" '
    'model_filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" '
    'tokenizer_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"'
)
LLAMA_SMALL_MODEL = (
    'model_id: "bartowski/Llama-3.2-1B-Instruct-GGUF" '
    'model_filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf" '
    'tokenizer_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"'
)

EXAMPLE_PROMPT_1 = (
    "Explain why leaves change color in autumn. Describe the roles of chlorophyll, "
    "carotenoids, and anthocyanins, and explain how shorter days and cooler temperatures "
    "affect the chemical processes inside a leaf. Discuss why different tree species "
    "produce different colors, why weather conditions can make some autumn displays more "
    "vivid than others, and what eventually causes a leaf to detach from its branch. Present "
    "the explanation for a curious reader without a biology background, using clear examples "
    "and avoiding unnecessary technical language. Conclude by explaining how this seasonal "
    "change helps a deciduous tree survive winter and prepare for growth in spring."
)
EXAMPLE_PROMPT_2 = (
    "Describe how a compiler turns source code into an executable program. Begin with lexical "
    "analysis and parsing, then explain semantic analysis, intermediate representations, "
    "optimization, machine-code generation, and linking. Show how errors can be detected at "
    "different stages and clarify the distinction between compile-time and runtime failures. "
    "Use one small imaginary function as an example and follow it through the major stages "
    "without relying on language-specific syntax. Also explain how debug and release builds "
    "may differ, why optimization can make debugging harder, and how the final executable "
    "interacts with libraries and the operating system when a user launches it."
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    server_flags: tuple[str, ...]


@dataclass(frozen=True)
class GenerationMetrics:
    input_token_count: int
    output_token_count: int
    time_to_first_token_microseconds: int | None
    end_to_end_latency_microseconds: int | None
    draft_token_acceptance_rate: float | None


class Benchmark(ABC):
    def cases(self) -> tuple[BenchmarkCase, ...]:
        return (BenchmarkCase("default", tuple(self.server_flags())),)

    def build_server_command(self, case: BenchmarkCase) -> list[str]:
        return [
            "cargo",
            "run",
            "--release",
            "--",
            *case.server_flags,
            "--control-socket",
            str(CONTROL_SOCKET),
            "--request-socket",
            str(REQUEST_SOCKET),
        ]

    def run(
        self,
        process: subprocess.Popen[bytes],
        proto: ProtoModules,
        case: BenchmarkCase,
    ) -> Any:
        with create_channel(REQUEST_SOCKET) as channel:
            wait_for_server(process, channel)
            client = proto.request_handler_grpc.RequestHandlerServiceStub(channel)
            try:
                return self.run_benchmark(client, proto, case)
            finally:
                logger.info("Benchmark finished; stopping mini-vllm-rs")
                send_shutdown(proto)

    # TODO: Make this an abstractmethod.
    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        pass

    @abstractmethod
    def server_flags(self) -> list[str]:
        pass

    @abstractmethod
    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
    ) -> Any:
        pass

    @staticmethod
    def generation_metrics(stats: Any) -> GenerationMetrics:
        draft_token_acceptance_rate = None
        if stats.HasField("draft_token_acceptance_rate"):
            draft_token_acceptance_rate = stats.draft_token_acceptance_rate
        time_to_first_token_microseconds = None
        end_to_end_latency_microseconds = None
        if stats.HasField("token_generation_latency"):
            time_to_first_token_microseconds = (
                stats.token_generation_latency.time_to_first_token_microseconds
            )
            end_to_end_latency_microseconds = (
                stats.token_generation_latency.end_to_end_latency_microseconds
            )
        return GenerationMetrics(
            input_token_count=stats.input_token_count,
            output_token_count=stats.output_token_count,
            time_to_first_token_microseconds=time_to_first_token_microseconds,
            end_to_end_latency_microseconds=end_to_end_latency_microseconds,
            draft_token_acceptance_rate=draft_token_acceptance_rate,
        )

    @classmethod
    def print_generation_stats(cls, stats: Any) -> None:
        metrics = cls.generation_metrics(stats)
        draft_acceptance_rate = "unavailable"
        if metrics.draft_token_acceptance_rate is not None:
            draft_acceptance_rate = (
                f"{metrics.draft_token_acceptance_rate * 100:.1f}%"
            )
        logger.info(
            "Generation stats:\n"
            "\tInput tokens: {}\n"
            "\tOutput tokens: {}\n"
            "\tTime to first token: {} us\n"
            "\tEnd-to-end latency: {} us\n"
            "\tDraft acceptance: {}",
            metrics.input_token_count,
            metrics.output_token_count,
            _format_optional_metric(metrics.time_to_first_token_microseconds),
            _format_optional_metric(metrics.end_to_end_latency_microseconds),
            draft_acceptance_rate,
        )


def _format_optional_metric(value: int | None) -> int | str:
    return "unavailable" if value is None else value
