"""Measure prefix publication and exact-input single-flight coordination."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import fmean
from threading import Event
from time import perf_counter_ns
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_SMALL_MODEL
from proto_loader import ProtoModules


COMPLETION_INDEPENDENT = "Completion publication / independent requests"
INCREMENTAL_INDEPENDENT = "Incremental publication / independent requests"
COMPLETION_SINGLE_FLIGHT = "Completion publication / exact-input single flight"
INCREMENTAL_SINGLE_FLIGHT = "Incremental publication / exact-input single flight"
PAGE_TOKEN_COUNT = 16
TARGET_KV_CACHE_SIZE_BYTES = 128 * 1024 * 1024
MAX_ACTIVE_REQUEST_COUNT = 8
MAX_BATCHED_TOKEN_COUNT = 4096
CONCURRENCY_LEVELS = (2, 4, 8)
OUTPUT_TOKEN_COUNTS = (16, 128)
DISTINCT_CONTROL_CONCURRENCY = 8


@dataclass(frozen=True)
class BurstResult:
    workload: str
    concurrency: int
    requested_output_token_count: int
    metrics: tuple[GenerationMetrics, ...]
    wall_time_microseconds: int


def _document(namespace: str, paragraph_count: int = 12) -> str:
    return "\n".join(
        f"{namespace} paragraph {index}: A distributed inference engine schedules "
        "requests, allocates cache pages, executes model layers, and streams tokens. "
        "This deterministic paragraph supplies context for a cache coordination test."
        for index in range(paragraph_count)
    )


def _question(namespace: str) -> str:
    return f"\n{namespace} question: Summarize the document in one sentence."


class PrefixCacheConcurrentPopulationBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return self._server_flags(incremental=True, single_flight=True)

    def cases(self) -> tuple[BenchmarkCase, ...]:
        return (
            BenchmarkCase(
                COMPLETION_INDEPENDENT,
                tuple(self._server_flags(incremental=False, single_flight=False)),
            ),
            BenchmarkCase(
                INCREMENTAL_INDEPENDENT,
                tuple(self._server_flags(incremental=True, single_flight=False)),
            ),
            BenchmarkCase(
                COMPLETION_SINGLE_FLIGHT,
                tuple(self._server_flags(incremental=False, single_flight=True)),
            ),
            BenchmarkCase(
                INCREMENTAL_SINGLE_FLIGHT,
                tuple(self._server_flags(incremental=True, single_flight=True)),
            ),
        )

    @staticmethod
    def _server_flags(incremental: bool, single_flight: bool) -> list[str]:
        publication_policy = (
            "PREFIX_CACHE_PUBLICATION_POLICY_INCREMENTAL"
            if incremental
            else "PREFIX_CACHE_PUBLICATION_POLICY_ON_REQUEST_COMPLETION"
        )
        coordination_policy = (
            "PREFIX_CACHE_COORDINATION_POLICY_EXACT_INPUT_SINGLE_FLIGHT"
            if single_flight
            else "PREFIX_CACHE_COORDINATION_POLICY_INDEPENDENT"
        )
        return [
            "--model",
            QWEN_SMALL_MODEL,
            "--kv-cache-config",
            (
                "kv_cache_type: KV_CACHE_TYPE_PAGED_PREFIX "
                f"per_page_token_count: {PAGE_TOKEN_COUNT} "
                f"target_kv_cache_size_bytes: {TARGET_KV_CACHE_SIZE_BYTES} "
                f"prefix_cache_publication_policy: {publication_policy}"
            ),
            "--scheduler-config",
            (
                f"max_batched_token_count: {MAX_BATCHED_TOKEN_COUNT} "
                f"max_active_request_count: {MAX_ACTIVE_REQUEST_COUNT} "
                f"prefix_cache_coordination_policy: {coordination_policy}"
            ),
        ]

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        _case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> tuple[BurstResult, ...]:
        results = []
        for output_token_count in OUTPUT_TOKEN_COUNTS:
            for concurrency in CONCURRENCY_LEVELS:
                namespace = f"exact-c{concurrency}-o{output_token_count}"
                prompt = _document(namespace) + _question(namespace)
                logger.warning(
                    "Running cold exact-input burst: concurrency={} output_tokens={}",
                    concurrency,
                    output_token_count,
                )
                results.append(
                    self._run_burst(
                        client,
                        proto,
                        "identical",
                        tuple(prompt for _ in range(concurrency)),
                        output_token_count,
                    )
                )

        for concurrency in CONCURRENCY_LEVELS:
            namespace = f"inflight-c{concurrency}"
            prompt = _document(namespace) + _question(namespace)
            logger.warning(
                "Running in-flight exact-input fan-out: concurrency={} output_tokens={}",
                concurrency,
                OUTPUT_TOKEN_COUNTS[-1],
            )
            results.append(
                self._run_inflight_fan_out(
                    client,
                    proto,
                    prompt,
                    concurrency,
                    OUTPUT_TOKEN_COUNTS[-1],
                )
            )

        logger.warning("Running distinct-input control burst")
        results.append(
            self._run_burst(
                client,
                proto,
                "distinct",
                tuple(
                    _document(f"distinct-{index}") + _question(f"distinct-{index}")
                    for index in range(DISTINCT_CONTROL_CONCURRENCY)
                ),
                OUTPUT_TOKEN_COUNTS[0],
            )
        )
        telemetry = client.GetPrefixCacheTelemetry(
            proto.model_runner.GetPrefixCacheTelemetryRequest()
        )
        for backend in telemetry.backends:
            self.print_prefix_cache_telemetry(backend)
        return tuple(results)

    def _run_burst(
        self,
        client: Any,
        proto: ProtoModules,
        workload: str,
        prompts: tuple[str, ...],
        max_new_tokens: int,
    ) -> BurstResult:
        started_at = perf_counter_ns()
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [
                executor.submit(
                    self._send_request,
                    client,
                    proto,
                    prompt,
                    max_new_tokens,
                )
                for prompt in prompts
            ]
            metrics = tuple(future.result() for future in futures)
        wall_time_microseconds = (perf_counter_ns() - started_at) // 1_000
        return BurstResult(
            workload,
            len(prompts),
            max_new_tokens,
            metrics,
            wall_time_microseconds,
        )

    def _run_inflight_fan_out(
        self,
        client: Any,
        proto: ProtoModules,
        prompt: str,
        concurrency: int,
        max_new_tokens: int,
    ) -> BurstResult:
        first_text_received = Event()
        started_at = perf_counter_ns()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            leader = executor.submit(
                self._send_request,
                client,
                proto,
                prompt,
                max_new_tokens,
                first_text_received,
            )
            if not first_text_received.wait(timeout=120):
                if leader.done():
                    leader.result()
                raise RuntimeError("in-flight leader did not emit text within 120 seconds")
            followers = tuple(
                executor.submit(
                    self._send_request,
                    client,
                    proto,
                    prompt,
                    max_new_tokens,
                )
                for _ in range(concurrency - 1)
            )
            metrics = (leader.result(), *(future.result() for future in followers))
        return BurstResult(
            "inflight",
            concurrency,
            max_new_tokens,
            metrics,
            (perf_counter_ns() - started_at) // 1_000,
        )

    def _send_request(
        self,
        client: Any,
        proto: ProtoModules,
        prompt: str,
        max_new_tokens: int,
        first_text_received: Event | None = None,
    ) -> GenerationMetrics:
        request = proto.request_handler.GenerateText(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        metrics = None
        for response in client.GenerateText(request):
            event = response.WhichOneof("event")
            if event == "text" and first_text_received is not None:
                first_text_received.set()
            elif event == "stats":
                metrics = self.generation_metrics(response.stats)
        if metrics is None:
            raise RuntimeError("request completed without generation statistics")
        return metrics

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, tuple[BurstResult, ...]]],
    ) -> None:
        for case, case_results in results:
            logger.warning("Results for {}", case.name)
            rows = []
            for result in case_results:
                ttft_values = tuple(
                    metric.time_to_first_token_microseconds
                    for metric in result.metrics
                )
                if any(value is None for value in ttft_values):
                    raise ValueError("completed request omitted time to first token")
                restored_token_count = sum(
                    metric.target_prefix_cache_metrics.restored_token_count
                    for metric in result.metrics
                )
                indexed_token_count = sum(
                    metric.target_prefix_cache_metrics.newly_indexed_token_count
                    for metric in result.metrics
                )
                mean_ttft = round(
                    fmean(value for value in ttft_values if value is not None)
                )
                max_e2e = max(
                    metric.end_to_end_latency_microseconds
                    for metric in result.metrics
                )
                rows.append(
                    (
                        result.workload,
                        result.concurrency,
                        result.requested_output_token_count,
                        restored_token_count,
                        indexed_token_count,
                        mean_ttft,
                        max_e2e,
                        result.wall_time_microseconds,
                    )
                )
                self._report_machine_row(case, result, mean_ttft, max_e2e)
            logger.info(
                "Prefix-cache coordination workloads:\n{}",
                tabulate(
                    rows,
                    headers=(
                        "Inputs",
                        "Concurrency",
                        "Output",
                        "Restored",
                        "Indexed",
                        "Mean TTFT us",
                        "Max E2E us",
                        "Wall us",
                    ),
                    tablefmt="rounded_outline",
                ),
            )

    @staticmethod
    def _report_machine_row(
        case: BenchmarkCase,
        result: BurstResult,
        mean_ttft_microseconds: int,
        max_e2e_microseconds: int,
    ) -> None:
        case_name = {
            COMPLETION_INDEPENDENT: "completion_independent",
            INCREMENTAL_INDEPENDENT: "incremental_independent",
            COMPLETION_SINGLE_FLIGHT: "completion_single_flight",
            INCREMENTAL_SINGLE_FLIGHT: "incremental_single_flight",
        }[case.name]
        metrics = result.metrics
        logger.info(
            "PREFIX_CACHE_CONCURRENT_POPULATION_RESULT "
            "case={} workload={} concurrency={} requested_output_tokens={} "
            "requests={} input_tokens={} generated_tokens={} restored_tokens={} "
            "indexed_tokens={} sum_prefill_us={} mean_ttft_us={} max_e2e_us={} wall_us={}",
            case_name,
            result.workload,
            result.concurrency,
            result.requested_output_token_count,
            len(metrics),
            sum(metric.input_token_count for metric in metrics),
            sum(metric.output_token_count for metric in metrics),
            sum(
                metric.target_prefix_cache_metrics.restored_token_count
                for metric in metrics
            ),
            sum(
                metric.target_prefix_cache_metrics.newly_indexed_token_count
                for metric in metrics
            ),
            sum(metric.prefill_duration_microseconds for metric in metrics),
            mean_ttft_microseconds,
            max_e2e_microseconds,
            result.wall_time_microseconds,
        )
