"""Measure prefix-cache reuse, concurrency, publication timing, and eviction."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_SMALL_MODEL
from proto_loader import ProtoModules


NO_PREFIX_CACHE = "Paged cache (no prefix caching)"
PREFIX_CACHE = "Paged prefix cache"
PAGE_TOKEN_COUNT = 16
# Large enough to admit the four-request cold burst, but smaller than the
# eight-prefix churn working set.
TARGET_KV_CACHE_SIZE_BYTES = 128 * 1024 * 1024
MAX_ACTIVE_REQUEST_COUNT = 8
MAX_BATCHED_TOKEN_COUNT = 2048
SHORT_MAX_NEW_TOKENS = 16
CONCURRENT_MAX_NEW_TOKENS = 128
OVERLAP_PERCENTAGES = (0, 25, 50, 75, 100)
FAN_OUT_COUNTS = (1, 2, 4, 8)
BURST_SIZE = 4
CHURN_PREFIX_COUNT = 8
CHURN_CYCLES = 2


@dataclass(frozen=True)
class RequestResult:
    label: str
    metrics: GenerationMetrics


@dataclass(frozen=True)
class PrefixCacheReuseResult:
    overlap_sweep: tuple[RequestResult, ...]
    warm_fan_out: tuple[tuple[int, tuple[RequestResult, ...]], ...]
    cold_burst: tuple[RequestResult, ...]
    warm_burst: tuple[RequestResult, ...]
    churn: tuple[RequestResult, ...]


def _document(namespace: str, paragraph_count: int = 24) -> str:
    """Build a stable long prompt whose namespace prevents cross-test reuse."""
    return "\n".join(
        f"{namespace} paragraph {index}: A distributed inference engine schedules "
        "requests, allocates cache pages, runs model layers, and streams generated "
        "tokens to its clients. This paragraph provides deterministic benchmark context."
        for index in range(paragraph_count)
    )


def _question(namespace: str, variant: int) -> str:
    return f"\n{namespace} question {variant}: Summarize the context in one sentence."


class PrefixCacheReuseBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return self._server_flags(prefix_caching=True)

    def cases(self) -> tuple[BenchmarkCase, ...]:
        return (
            BenchmarkCase(NO_PREFIX_CACHE, tuple(self._server_flags(False))),
            BenchmarkCase(PREFIX_CACHE, tuple(self._server_flags(True))),
        )

    @staticmethod
    def _server_flags(prefix_caching: bool) -> list[str]:
        cache_type = (
            "KV_CACHE_TYPE_PAGED_PREFIX"
            if prefix_caching
            else "KV_CACHE_TYPE_PAGED"
        )
        return [
            "--model",
            QWEN_SMALL_MODEL,
            "--kv-cache-config",
            (
                f"kv_cache_type: {cache_type} "
                f"per_page_token_count: {PAGE_TOKEN_COUNT} "
                f"target_kv_cache_size_bytes: {TARGET_KV_CACHE_SIZE_BYTES}"
            ),
            "--scheduler-config",
            (
                f"max_batched_token_count: {MAX_BATCHED_TOKEN_COUNT} "
                f"max_active_request_count: {MAX_ACTIVE_REQUEST_COUNT}"
            ),
        ]

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        _case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> PrefixCacheReuseResult:
        logger.warning("Running prefix-overlap sweep")
        overlap_sweep = self._run_overlap_sweep(client, proto)
        logger.warning("Running warm fan-out")
        warm_fan_out = self._run_warm_fan_out(client, proto)
        logger.warning("Running cold-versus-warm burst")
        cold_burst, warm_burst = self._run_cold_warm_burst(client, proto)
        # Churn deliberately disturbs the whole cache, so it must remain last.
        logger.warning("Running working-set churn")
        churn = self._run_churn(client, proto)

        telemetry = client.GetPrefixCacheTelemetry(
            proto.model_runner.GetPrefixCacheTelemetryRequest()
        )
        for backend in telemetry.backends:
            self.print_prefix_cache_telemetry(backend)
        return PrefixCacheReuseResult(
            overlap_sweep, warm_fan_out, cold_burst, warm_burst, churn
        )

    def _run_overlap_sweep(
        self,
        client: Any,
        proto: ProtoModules,
    ) -> tuple[RequestResult, ...]:
        paragraphs = _document("overlap").splitlines()
        self._send_request(
            client,
            proto,
            "overlap seed",
            "\n".join(paragraphs) + _question("overlap-seed", 0),
        )
        results = []
        for percentage in OVERLAP_PERCENTAGES:
            count = len(paragraphs) * percentage // 100
            shared = (
                "\n".join(paragraphs[:count])
                if count
                else _document("overlap-zero", 2)
            )
            results.append(
                self._send_request(
                    client,
                    proto,
                    f"overlap {percentage}%",
                    shared + _question("overlap-probe", percentage),
                )
            )
        return tuple(results)

    def _run_warm_fan_out(
        self, client: Any, proto: ProtoModules
    ) -> tuple[tuple[int, tuple[RequestResult, ...]], ...]:
        prefix = _document("fan-out")
        self._send_request(
            client, proto, "fan-out seed", prefix + _question("fan-out-seed", 0)
        )
        return tuple(
            (
                count,
                self._send_concurrently(
                    client,
                    proto,
                    tuple(
                        (
                            f"fan-out {count} request {index + 1}",
                            prefix + _question(f"fan-out-{count}", index),
                        )
                        for index in range(count)
                    ),
                    CONCURRENT_MAX_NEW_TOKENS,
                ),
            )
            for count in FAN_OUT_COUNTS
        )

    def _run_cold_warm_burst(
        self, client: Any, proto: ProtoModules
    ) -> tuple[tuple[RequestResult, ...], tuple[RequestResult, ...]]:
        prompt = _document("cold-burst") + _question("cold-burst", 0)
        cold = self._send_concurrently(
            client,
            proto,
            tuple((f"cold burst request {i + 1}", prompt) for i in range(BURST_SIZE)),
            CONCURRENT_MAX_NEW_TOKENS,
        )
        warm = self._send_concurrently(
            client,
            proto,
            tuple((f"warm burst request {i + 1}", prompt) for i in range(BURST_SIZE)),
            CONCURRENT_MAX_NEW_TOKENS,
        )
        return cold, warm

    def _run_churn(
        self,
        client: Any,
        proto: ProtoModules,
    ) -> tuple[RequestResult, ...]:
        prompts = tuple(
            _document(f"churn-{i}") + _question(f"churn-{i}", 0)
            for i in range(CHURN_PREFIX_COUNT)
        )
        return tuple(
            self._send_request(
                client, proto, f"churn cycle {cycle + 1} prefix {i + 1}", prompt
            )
            for cycle in range(CHURN_CYCLES)
            for i, prompt in enumerate(prompts)
        )

    def _send_concurrently(
        self,
        client: Any,
        proto: ProtoModules,
        requests: tuple[tuple[str, str], ...],
        max_new_tokens: int,
    ) -> tuple[RequestResult, ...]:
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = [
                executor.submit(
                    self._send_request,
                    client,
                    proto,
                    label,
                    prompt,
                    max_new_tokens,
                )
                for label, prompt in requests
            ]
            return tuple(future.result() for future in futures)

    def _send_request(
        self,
        client: Any,
        proto: ProtoModules,
        label: str,
        prompt: str,
        max_new_tokens: int = SHORT_MAX_NEW_TOKENS,
    ) -> RequestResult:
        logger.info("Sending {}", label)
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
            if response.WhichOneof("event") == "stats":
                metrics = self.generation_metrics(response.stats)
        if metrics is None:
            raise RuntimeError(f"{label} completed without generation statistics")
        logger.info(
            "{}: input={} restored={} indexed={} prefill_us={} ttft_us={} e2e_us={}",
            label,
            metrics.input_token_count,
            metrics.target_prefix_cache_metrics.restored_token_count,
            metrics.target_prefix_cache_metrics.newly_indexed_token_count,
            metrics.prefill_duration_microseconds,
            metrics.time_to_first_token_microseconds,
            metrics.end_to_end_latency_microseconds,
        )
        return RequestResult(label, metrics)

    def report_results(
        self, results: list[tuple[BenchmarkCase, PrefixCacheReuseResult]]
    ) -> None:
        for case, result in results:
            logger.warning("Results for {}", case.name)
            self._report_overlap(result.overlap_sweep)
            self._report_fan_out(result.warm_fan_out)
            self._report_burst(result.cold_burst, result.warm_burst)
            self._report_churn(result.churn)
            self._report_machine_readable(case, result)

    @classmethod
    def _report_machine_readable(
        cls,
        case: BenchmarkCase,
        result: PrefixCacheReuseResult,
    ) -> None:
        case_name = "prefix_cache" if case.name == PREFIX_CACHE else "no_prefix_cache"
        for percentage, request in zip(
            OVERLAP_PERCENTAGES,
            result.overlap_sweep,
            strict=True,
        ):
            cls._report_machine_row(
                case_name,
                "overlap",
                str(percentage),
                (request,),
            )
        for request_count, requests in result.warm_fan_out:
            cls._report_machine_row(
                case_name,
                "fan_out",
                str(request_count),
                requests,
            )
        cls._report_machine_row(
            case_name,
            "burst",
            "cold",
            result.cold_burst,
        )
        cls._report_machine_row(
            case_name,
            "burst",
            "warm",
            result.warm_burst,
        )
        for cycle in range(CHURN_CYCLES):
            requests = result.churn[
                cycle * CHURN_PREFIX_COUNT : (cycle + 1) * CHURN_PREFIX_COUNT
            ]
            cls._report_machine_row(
                case_name,
                "churn",
                str(cycle + 1),
                requests,
            )

    @staticmethod
    def _report_machine_row(
        case_name: str,
        workload: str,
        variant: str,
        requests: tuple[RequestResult, ...],
    ) -> None:
        metrics = tuple(request.metrics for request in requests)
        ttft_values = tuple(
            metric.time_to_first_token_microseconds for metric in metrics
        )
        if any(value is None for value in ttft_values):
            raise ValueError("completed request omitted time to first token")
        logger.info(
            "PREFIX_CACHE_REUSE_RESULT "
            "case={} workload={} variant={} requests={} input_tokens={} "
            "output_tokens={} restored_tokens={} indexed_tokens={} "
            "mean_prefill_us={} mean_ttft_us={} max_e2e_us={}",
            case_name,
            workload,
            variant,
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
            round(fmean(metric.prefill_duration_microseconds for metric in metrics)),
            round(fmean(value for value in ttft_values if value is not None)),
            max(metric.end_to_end_latency_microseconds for metric in metrics),
        )

    @staticmethod
    def _report_overlap(results: tuple[RequestResult, ...]) -> None:
        rows = [
            (
                r.label,
                r.metrics.input_token_count,
                r.metrics.target_prefix_cache_metrics.restored_token_count,
                r.metrics.prefill_duration_microseconds,
                r.metrics.time_to_first_token_microseconds,
            )
            for r in results
        ]
        logger.info(
            "Prefix-overlap sweep:\n{}",
            tabulate(
                rows,
                headers=("Overlap", "Input", "Restored", "Prefill us", "TTFT us"),
                tablefmt="rounded_outline",
            ),
        )

    @staticmethod
    def _report_fan_out(
        results: tuple[tuple[int, tuple[RequestResult, ...]], ...]
    ) -> None:
        rows = [
            (
                count,
                round(
                    fmean(
                        r.metrics.target_prefix_cache_metrics.restored_token_count
                        for r in requests
                    )
                ),
                round(
                    fmean(r.metrics.prefill_duration_microseconds for r in requests)
                ),
                round(
                    fmean(
                        r.metrics.time_to_first_token_microseconds or 0
                        for r in requests
                    )
                ),
                max(r.metrics.end_to_end_latency_microseconds for r in requests),
            )
            for count, requests in results
        ]
        logger.info(
            "Warm fan-out:\n{}",
            tabulate(
                rows,
                headers=(
                    "Requests",
                    "Mean restored",
                    "Mean prefill us",
                    "Mean TTFT us",
                    "Max E2E us",
                ),
                tablefmt="rounded_outline",
            ),
        )

    @staticmethod
    def _report_burst(
        cold: tuple[RequestResult, ...], warm: tuple[RequestResult, ...]
    ) -> None:
        rows = []
        for phase, requests in (("Cold", cold), ("Warm", warm)):
            rows.append(
                (
                    phase,
                    sum(
                        r.metrics.target_prefix_cache_metrics.restored_token_count
                        for r in requests
                    ),
                    sum(
                        r.metrics.target_prefix_cache_metrics.newly_indexed_token_count
                        for r in requests
                    ),
                    round(
                        fmean(
                            r.metrics.prefill_duration_microseconds for r in requests
                        )
                    ),
                    round(
                        fmean(
                            r.metrics.time_to_first_token_microseconds or 0
                            for r in requests
                        )
                    ),
                )
            )
        logger.info(
            "Cold versus warm burst:\n{}",
            tabulate(
                rows,
                headers=(
                    "Phase",
                    "Restored",
                    "Newly indexed",
                    "Mean prefill us",
                    "Mean TTFT us",
                ),
                tablefmt="rounded_outline",
            ),
        )

    @staticmethod
    def _report_churn(results: tuple[RequestResult, ...]) -> None:
        rows = []
        for cycle in range(CHURN_CYCLES):
            cycle_results = results[
                cycle * CHURN_PREFIX_COUNT : (cycle + 1) * CHURN_PREFIX_COUNT
            ]
            rows.append(
                (
                    cycle + 1,
                    sum(
                        r.metrics.target_prefix_cache_metrics.restored_token_count
                        for r in cycle_results
                    ),
                    sum(
                        r.metrics.target_prefix_cache_metrics.newly_indexed_token_count
                        for r in cycle_results
                    ),
                    round(
                        fmean(
                            r.metrics.prefill_duration_microseconds
                            for r in cycle_results
                        )
                    ),
                )
            )
        logger.info(
            "Working-set churn:\n{}",
            tabulate(
                rows,
                headers=("Cycle", "Restored", "Newly indexed", "Mean prefill us"),
                tablefmt="rounded_outline",
            ),
        )
