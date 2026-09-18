"""Benchmark two concurrently submitted generation requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_SMALL_MODEL
from benchmarks.example_prompts import EXAMPLE_LONG_PROMPTS
from proto_loader import ProtoModules


@dataclass(frozen=True)
class ConcurrentRequestResult:
    request_number: int
    output: str
    metrics: GenerationMetrics


@dataclass(frozen=True)
class ConcurrentCaseResult:
    requests: tuple[ConcurrentRequestResult, ...]
    wall_time_seconds: float


class ConcurrentRequestsBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return self._server_flags(use_draft_model=True, use_continuous_batching=True)

    def cases(self) -> tuple[BenchmarkCase, ...]:
        return (
            BenchmarkCase(
                "Sequential",
                tuple(self._server_flags(max_active_request_count=1)),
            ),
            BenchmarkCase(
                "Concurrent 2",
                tuple(self._server_flags(max_active_request_count=2)),
            ),
            BenchmarkCase(
                "Concurrent 4",
                tuple(self._server_flags(max_active_request_count=4)),
            ),
        )

    @staticmethod
    def _server_flags(max_active_request_count: int) -> list[str]:
        return [
            "--model",
            QWEN_SMALL_MODEL,
            "--kv-cache-type",
            "paged-prefix:16",
            "--max-active-request-count",
            str(max_active_request_count),
        ]

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
    ) -> ConcurrentCaseResult:
        started_at = perf_counter()
        prompts = EXAMPLE_LONG_PROMPTS[:4]
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [
                executor.submit(
                    self._send_request,
                    client,
                    proto,
                    case.name,
                    request_number,
                    prompt,
                )
                for request_number, prompt in enumerate(prompts, start=1)
            ]
            requests = tuple(future.result() for future in futures)
        return ConcurrentCaseResult(
            requests=requests,
            wall_time_seconds=perf_counter() - started_at,
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        case_results = [
            (case, self._verify_result_type(case, result))
            for case, result in results
        ]
        logger.info("Metrics comparison:\n{}", self._format_metrics_table(case_results))

    @staticmethod
    def _verify_result_type(
        case: BenchmarkCase,
        result: Any,
    ) -> ConcurrentCaseResult:
        if not isinstance(result, ConcurrentCaseResult):
            raise RuntimeError(f"case {case.name} did not return concurrent results")
        return result

    @classmethod
    def _format_metrics_table(
        cls,
        results: list[tuple[BenchmarkCase, ConcurrentCaseResult]],
    ) -> str:
        headers = (
            "Case",
            "Request",
            "Input",
            "Output",
            "TTFT (us)",
            "E2E (us)",
            "Draft accept",
            "Wall (s)",
            "Output tok/s",
        )
        rows = []
        for case, result in results:
            total_output_tokens = sum(
                request.metrics.output_token_count for request in result.requests
            )
            output_tokens_per_second = total_output_tokens / result.wall_time_seconds
            for request in result.requests:
                metrics = request.metrics
                rows.append(
                    (
                        case.name,
                        str(request.request_number),
                        str(metrics.input_token_count),
                        str(metrics.output_token_count),
                        cls._format_optional(metrics.time_to_first_token_microseconds),
                        cls._format_optional(metrics.end_to_end_latency_microseconds),
                        cls._format_acceptance(metrics.draft_token_acceptance_rate),
                        f"{result.wall_time_seconds:.3f}",
                        f"{output_tokens_per_second:.2f}",
                    )
                )
        return tabulate(rows, headers=headers, tablefmt="simple")

    @staticmethod
    def _format_optional(value: int | None) -> str:
        return "-" if value is None else str(value)

    @staticmethod
    def _format_acceptance(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.1f}%"

    def _send_request(
        self,
        client: Any,
        proto: ProtoModules,
        case_name: str,
        request_number: int,
        prompt: str,
    ) -> ConcurrentRequestResult:
        request = proto.request_handler.GenerateText(
            prompt=prompt,
            max_new_tokens=1024,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        logger.warning("Sending concurrent request {}:\n{}", request_number, prompt)
        output_parts: list[str] = []
        result = None
        for response in client.GenerateText(request):
            event = response.WhichOneof("event")
            if event == "text":
                output_parts.append(response.text)
            elif event == "stats":
                result = ConcurrentRequestResult(
                    request_number=request_number,
                    output="".join(output_parts),
                    metrics=self.generation_metrics(response.stats),
                )
                logger.warning(
                    "{} request {} output:\n{}",
                    case_name,
                    request_number,
                    result.output,
                )
        if result is None:
            raise RuntimeError(f"request {request_number} completed without statistics")
        return result
