"""Shared benchmark lifecycle."""

from __future__ import annotations

import subprocess
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
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
QWEN_SMALL_DRAFT_MODEL = (
    f"model {{ {QWEN_SMALL_MODEL} }} "
    "token_count_policy { fixed { draft_token_count: 4 } }"
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


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    server_flags: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DraftTokenMetrics:
    accepted_token_count: int
    proposed_token_count: int
    selected_token_count_histogram: tuple[tuple[int, int], ...]

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_token_count == 0:
            return 0.0
        return self.accepted_token_count / self.proposed_token_count


@dataclass(frozen=True)
class GenerationMetrics:
    input_token_count: int
    output_token_count: int
    time_to_first_token_microseconds: int | None
    end_to_end_latency_microseconds: int | None
    draft_token_metrics: DraftTokenMetrics | None


class Benchmark(ABC):
    def cases(self) -> tuple[BenchmarkCase, ...]:
        return (BenchmarkCase("default", tuple(self.server_flags())),)

    def build_server_command(
        self,
        case: BenchmarkCase,
        trace_directory: Path | None = None,
    ) -> list[str]:
        command = [
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
        if trace_directory is not None:
            command.extend(("--trace-directory", str(trace_directory)))
        return command

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
                return self.run_benchmark(client, proto, case, process)
            finally:
                logger.info("Benchmark finished; stopping mini-vllm-rs")
                send_shutdown(proto)

    @abstractmethod
    def server_flags(self) -> list[str]:
        pass

    @abstractmethod
    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        process: subprocess.Popen[bytes],
    ) -> Any:
        pass

    @abstractmethod
    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        pass

    @staticmethod
    def generation_metrics(stats: Any) -> GenerationMetrics:
        draft_token_metrics = None
        if stats.HasField("draft_token_stats"):
            draft_stats = stats.draft_token_stats
            draft_token_metrics = DraftTokenMetrics(
                accepted_token_count=draft_stats.accepted_token_count,
                proposed_token_count=draft_stats.proposed_token_count,
                selected_token_count_histogram=tuple(
                    (bucket.draft_token_count, bucket.usage_count)
                    for bucket in draft_stats.selected_token_count_histogram
                ),
            )
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
            draft_token_metrics=draft_token_metrics,
        )

    @classmethod
    def print_generation_stats(cls, stats: Any) -> None:
        metrics = cls.generation_metrics(stats)
        draft_acceptance_rate = "unavailable"
        draft_token_totals = "unavailable"
        draft_token_histogram = "unavailable"
        if metrics.draft_token_metrics is not None:
            draft_metrics = metrics.draft_token_metrics
            draft_acceptance_rate = f"{draft_metrics.acceptance_rate * 100:.1f}%"
            draft_token_totals = (
                f"{draft_metrics.accepted_token_count}/"
                f"{draft_metrics.proposed_token_count}"
            )
            draft_token_histogram = ", ".join(
                f"{token_count}:{usage_count}"
                for token_count, usage_count in (
                    draft_metrics.selected_token_count_histogram
                )
            )
        logger.info(
            "Generation stats:\n"
            "\tInput tokens: {}\n"
            "\tOutput tokens: {}\n"
            "\tTime to first token: {} us\n"
            "\tEnd-to-end latency: {} us\n"
            "\tDraft acceptance: {}\n"
            "\tDraft accepted/proposed: {}\n"
            "\tDraft selected count:uses: {}",
            metrics.input_token_count,
            metrics.output_token_count,
            _format_optional_metric(metrics.time_to_first_token_microseconds),
            _format_optional_metric(metrics.end_to_end_latency_microseconds),
            draft_acceptance_rate,
            draft_token_totals,
            draft_token_histogram,
        )


def _format_optional_metric(value: int | None) -> int | str:
    return "unavailable" if value is None else value
