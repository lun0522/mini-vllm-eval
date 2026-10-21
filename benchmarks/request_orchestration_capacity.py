"""Measure whether CPU capacity helps a decode-heavy hot-prefix workload."""

from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from time import sleep
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_SMALL_MODEL
from proto_loader import ProtoModules


TARGET_KV_CACHE_SIZE_BYTES = 256 * 1024 * 1024
MAX_BATCHED_TOKEN_COUNT = 2048
MAX_ACTIVE_REQUEST_COUNT = 4
REQUEST_INTERVAL_SECONDS = 0.050
MAX_NEW_TOKEN_COUNT = 128
PROFILE_FILE = "benchmarks/backend_throughput_profiles.textproto"


@dataclass(frozen=True)
class WorkloadDefinition:
    name: str
    request_count: int


@dataclass(frozen=True)
class CaseDefinition:
    workload: WorkloadDefinition
    routing_mode: str
    measured_backend_id: int

    @property
    def name(self) -> str:
        return f"{self.workload.name}_{self.routing_mode}"


WORKLOADS = (
    WorkloadDefinition("gpu_only", 4),
    WorkloadDefinition("crossover_boundary", 7),
    WorkloadDefinition("saturated", 48),
)
ROUTING_MODES = (
    # In mixed mode backend 2 is GPU; zero leaves selection to the orchestrator.
    ("forced_gpu", 2),
    ("mixed_routing", 0),
)
CASES = tuple(
    CaseDefinition(workload, routing_mode, backend_id)
    for workload in WORKLOADS
    for routing_mode, backend_id in ROUTING_MODES
)
CASES_BY_NAME = {case.name: case for case in CASES}
SEED_BACKEND_IDS = (1, 2)


@dataclass(frozen=True)
class SeedResult:
    requested_backend_id: int
    actual_backend_id: int
    input_token_count: int
    indexed_token_count: int


@dataclass(frozen=True)
class RequestResult:
    label: str
    metrics: GenerationMetrics
    request_id: int
    submission_offset_seconds: float


@dataclass(frozen=True)
class CapacityResult:
    case_name: str
    seeds: tuple[SeedResult, ...]
    requests: tuple[RequestResult, ...]
    wall_time_seconds: float


def _hot_prefix() -> str:
    # Qwen tokenizes each repeated word as approximately one token. The fixed
    # framing and 480 repeated words produce an input close to the measured
    # 512-token throughput profile after chat-template formatting.
    words = " ".join("routing" for _ in range(480))
    return (
        "Shared routing-capacity study context. The following stable sequence "
        f"is reused by every measured request: {words}"
    )


HOT_PREFIX = _hot_prefix()


class RequestOrchestrationCapacityBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return self._server_flags()

    def cases(self) -> tuple[BenchmarkCase, ...]:
        return tuple(
            BenchmarkCase(case.name, tuple(self._server_flags())) for case in CASES
        )

    @staticmethod
    def _server_flags() -> list[str]:
        return [
            "--model",
            QWEN_SMALL_MODEL,
            "--inference-device",
            "mixed",
            "--activation-dtype",
            "f32",
            "--input-preprocessing-thread-count",
            "1",
            "--kv-cache-config",
            (
                "kv_cache_type: KV_CACHE_TYPE_PAGED_PREFIX "
                "per_page_token_count: 16 "
                f"target_kv_cache_size_bytes: {TARGET_KV_CACHE_SIZE_BYTES}"
            ),
            "--scheduler-config",
            (
                f"max_batched_token_count: {MAX_BATCHED_TOKEN_COUNT} "
                f"max_active_request_count: {MAX_ACTIVE_REQUEST_COUNT}"
            ),
            "--request-orchestrator-config",
            (
                "routing_policy: "
                "REQUEST_ROUTING_POLICY_CACHE_AWARE_LEAST_ESTIMATED_WORK"
            ),
            "--backend-throughput-profiles-file",
            PROFILE_FILE,
        ]

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> CapacityResult:
        definition = CASES_BY_NAME[case.name]
        seeds = tuple(
            self._seed_backend(client, proto, case.name, backend_id)
            for backend_id in SEED_BACKEND_IDS
        )
        request_count = definition.workload.request_count
        labels = tuple(
            f"{definition.workload.name}-{index + 1:02d}"
            for index in range(request_count)
        )
        logger.info(
            "REQUEST_ORCHESTRATION_CAPACITY_REPLAY case={} workload={} "
            "routing_mode={} version=2 interval_ms=50 preprocessing_threads=1 "
            "requests={} order={}",
            case.name,
            definition.workload.name,
            definition.routing_mode,
            request_count,
            ",".join(labels),
        )

        started_at = perf_counter()
        with ThreadPoolExecutor(max_workers=request_count) as executor:
            futures = []
            for position, label in enumerate(labels):
                delay = started_at + position * REQUEST_INTERVAL_SECONDS - perf_counter()
                if delay > 0:
                    sleep(delay)
                submission_offset = perf_counter() - started_at
                responses = self._start_request(
                    client,
                    proto,
                    label,
                    definition.measured_backend_id,
                )
                futures.append(
                    executor.submit(
                        self._collect_response,
                        case.name,
                        label,
                        responses,
                        submission_offset,
                    )
                )
            requests = tuple(future.result() for future in futures)
        wall_time_seconds = perf_counter() - started_at

        telemetry = client.GetPrefixCacheTelemetry(
            proto.model_runner.GetPrefixCacheTelemetryRequest()
        )
        for backend in telemetry.backends:
            self.print_prefix_cache_telemetry(backend)
        return CapacityResult(case.name, seeds, requests, wall_time_seconds)

    @staticmethod
    def _seed_backend(
        client: Any,
        proto: ProtoModules,
        case_name: str,
        backend_id: int,
    ) -> SeedResult:
        logger.warning("Seeding hot prefix on backend {} for {}", backend_id, case_name)
        request = proto.request_handler.GenerateText(
            prompt=f"{HOT_PREFIX}\nSeed request: acknowledge the shared context.",
            max_new_tokens=1,
            repeat_penalty=1.0,
            repeat_last_n=0,
            stream_output=True,
            ignore_eos_tokens=True,
            benchmark_backend_id=backend_id,
        )
        stats = None
        for response in client.GenerateText(request):
            if response.WhichOneof("event") == "stats":
                stats = response.stats
        if stats is None:
            raise RuntimeError(f"{case_name} backend {backend_id} seed omitted statistics")
        result = SeedResult(
            requested_backend_id=backend_id,
            actual_backend_id=stats.backend_id,
            input_token_count=stats.input_token_count,
            indexed_token_count=(
                stats.target_prefix_cache_stats.newly_indexed_token_count
            ),
        )
        logger.info(
            "REQUEST_ORCHESTRATION_CAPACITY_SEED case={} requested_backend_id={} "
            "actual_backend_id={} input_tokens={} indexed_tokens={}",
            case_name,
            result.requested_backend_id,
            result.actual_backend_id,
            result.input_token_count,
            result.indexed_token_count,
        )
        return result

    @staticmethod
    def _start_request(
        client: Any,
        proto: ProtoModules,
        label: str,
        benchmark_backend_id: int,
    ) -> Any:
        logger.info("Sending hot-prefix request {}", label)
        request = proto.request_handler.GenerateText(
            prompt=(
                f"{HOT_PREFIX}\nGive a concise conclusion about the shared context."
            ),
            max_new_tokens=MAX_NEW_TOKEN_COUNT,
            repeat_penalty=1.0,
            repeat_last_n=0,
            stream_output=True,
            ignore_eos_tokens=True,
            benchmark_backend_id=benchmark_backend_id,
        )
        return client.GenerateText(request)

    @classmethod
    def _collect_response(
        cls,
        case_name: str,
        label: str,
        responses: Any,
        submission_offset_seconds: float,
    ) -> RequestResult:
        stats = None
        for response in responses:
            if response.WhichOneof("event") == "stats":
                stats = response.stats
        if stats is None:
            raise RuntimeError(f"{case_name} {label} omitted statistics")
        metrics = cls.generation_metrics(stats)
        if metrics.output_token_count != MAX_NEW_TOKEN_COUNT:
            raise RuntimeError(
                f"{case_name} {label} produced {metrics.output_token_count} tokens "
                f"instead of {MAX_NEW_TOKEN_COUNT}"
            )
        return RequestResult(label, metrics, stats.request_id, submission_offset_seconds)

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_case = self._validated_results(results)
        logger.info(
            "Hot-prefix orchestration comparison:\n{}",
            self._summary_table(results_by_case),
        )
        for workload in WORKLOADS:
            forced_gpu = results_by_case[f"{workload.name}_forced_gpu"]
            mixed = results_by_case[f"{workload.name}_mixed_routing"]
            logger.info(
                "Hot-prefix capacity workload={}: speedup={:.3f}x "
                "wall_time_change={:+.2f}%",
                workload.name,
                forced_gpu.wall_time_seconds / mixed.wall_time_seconds,
                (mixed.wall_time_seconds / forced_gpu.wall_time_seconds - 1) * 100,
            )
        logger.info(
            "Hot-prefix request metrics:\n{}",
            self._request_table(results_by_case),
        )
        self._report_machine_readable(results_by_case)

    @classmethod
    def _validated_results(
        cls,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> dict[str, CapacityResult]:
        results_by_case: dict[str, CapacityResult] = {}
        for benchmark_case, result in results:
            if not isinstance(result, CapacityResult):
                raise RuntimeError(
                    f"case {benchmark_case.name} returned unexpected results"
                )
            if result.case_name != benchmark_case.name:
                raise RuntimeError(f"case {benchmark_case.name} returned mismatched results")
            if benchmark_case.name in results_by_case:
                raise RuntimeError(f"received duplicate result for {benchmark_case.name}")
            definition = CASES_BY_NAME[benchmark_case.name]
            if (
                result.wall_time_seconds <= 0
                or len(result.requests) != definition.workload.request_count
            ):
                raise RuntimeError(f"case {benchmark_case.name} returned invalid results")
            if tuple(seed.requested_backend_id for seed in result.seeds) != (
                SEED_BACKEND_IDS
            ):
                raise RuntimeError(f"case {benchmark_case.name} returned unexpected seeds")
            for seed in result.seeds:
                if seed.actual_backend_id != seed.requested_backend_id:
                    raise RuntimeError(
                        f"case {benchmark_case.name} seed requested backend "
                        f"{seed.requested_backend_id} but used {seed.actual_backend_id}"
                    )
                if seed.indexed_token_count <= 0:
                    raise RuntimeError(
                        f"case {benchmark_case.name} seed did not populate backend "
                        f"{seed.actual_backend_id}"
                    )
            for request in result.requests:
                metrics = request.metrics
                if metrics.output_token_count != MAX_NEW_TOKEN_COUNT:
                    raise RuntimeError(
                        f"case {benchmark_case.name} {request.label} returned invalid output"
                    )
                if metrics.target_prefix_cache_metrics.restored_token_count <= 0:
                    raise RuntimeError(
                        f"case {benchmark_case.name} {request.label} did not restore the prefix"
                    )
                if metrics.backend_id not in SEED_BACKEND_IDS:
                    raise RuntimeError(
                        f"case {benchmark_case.name} {request.label} used invalid backend "
                        f"{metrics.backend_id}"
                    )
            results_by_case[benchmark_case.name] = result
        if set(results_by_case) != set(CASES_BY_NAME):
            raise RuntimeError(
                "hot-prefix benchmark requires forced-GPU and mixed-routing cases"
            )
        mixed_cpu_counts = []
        for workload in WORKLOADS:
            forced_gpu = results_by_case[f"{workload.name}_forced_gpu"]
            mixed = results_by_case[f"{workload.name}_mixed_routing"]
            forced_inputs = tuple(
                request.metrics.input_token_count for request in forced_gpu.requests
            )
            mixed_inputs = tuple(
                request.metrics.input_token_count for request in mixed.requests
            )
            if forced_inputs != mixed_inputs:
                raise RuntimeError(
                    f"{workload.name} cases encoded to different input token counts"
                )
            if any(request.metrics.backend_id != 2 for request in forced_gpu.requests):
                raise RuntimeError(
                    f"{workload.name} forced-GPU control routed a request away from GPU"
                )
            mixed_cpu_counts.append(
                sum(request.metrics.backend_id == 1 for request in mixed.requests)
            )
        if mixed_cpu_counts != sorted(mixed_cpu_counts):
            raise RuntimeError(
                "CPU request counts decreased as hot-prefix workload size increased"
            )
        return results_by_case

    @classmethod
    def _summary_table(cls, results: dict[str, CapacityResult]) -> str:
        rows = []
        for definition in CASES:
            result = results[definition.name]
            metrics = cls._summary_metrics(result)
            rows.append(
                (
                    definition.workload.name,
                    definition.routing_mode,
                    definition.name,
                    cls._backend_distribution(result),
                    f"{result.wall_time_seconds:.3f}",
                    f"{metrics['output_tokens_per_second']:.2f}",
                    f"{metrics['median_ttft_milliseconds']:.3f}",
                    f"{metrics['p95_ttft_milliseconds']:.3f}",
                    f"{metrics['median_e2e_seconds']:.3f}",
                    f"{metrics['p95_e2e_seconds']:.3f}",
                    metrics["restored_token_count"],
                )
            )
        return tabulate(
            rows,
            headers=(
                "Workload",
                "Routing",
                "Case",
                "Backend requests",
                "Wall (s)",
                "Output tok/s",
                "Median TTFT (ms)",
                "p95 TTFT (ms)",
                "Median E2E (s)",
                "p95 E2E (s)",
                "Restored",
            ),
            tablefmt="simple",
        )

    @staticmethod
    def _backend_distribution(result: CapacityResult) -> str:
        cpu_count = sum(request.metrics.backend_id == 1 for request in result.requests)
        return f"CPU:{cpu_count}, GPU:{len(result.requests) - cpu_count}"

    @classmethod
    def _summary_metrics(cls, result: CapacityResult) -> dict[str, float | int]:
        ttft = tuple(cls._required_ttft(request.metrics) for request in result.requests)
        e2e = tuple(
            request.metrics.end_to_end_latency_microseconds
            for request in result.requests
        )
        return {
            "output_tokens_per_second": sum(
                request.metrics.output_token_count for request in result.requests
            )
            / result.wall_time_seconds,
            "median_ttft_milliseconds": median(ttft) / 1_000,
            "p95_ttft_milliseconds": cls._percentile(ttft, 0.95) / 1_000,
            "median_e2e_seconds": median(e2e) / 1_000_000,
            "p95_e2e_seconds": cls._percentile(e2e, 0.95) / 1_000_000,
            "restored_token_count": sum(
                request.metrics.target_prefix_cache_metrics.restored_token_count
                for request in result.requests
            ),
        }

    @staticmethod
    def _percentile(values: tuple[int, ...], percentile: float) -> int:
        ordered = sorted(values)
        return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]

    @staticmethod
    def _required_ttft(metrics: GenerationMetrics) -> int:
        if metrics.time_to_first_token_microseconds is None:
            raise RuntimeError("completed request omitted time to first token")
        return metrics.time_to_first_token_microseconds

    @classmethod
    def _request_table(cls, results: dict[str, CapacityResult]) -> str:
        rows = []
        for definition in CASES:
            for request in results[definition.name].requests:
                metrics = request.metrics
                device = "GPU" if metrics.backend_id == 2 else "CPU"
                rows.append(
                    (
                        definition.name,
                        request.label,
                        device,
                        metrics.input_token_count,
                        metrics.output_token_count,
                        metrics.target_prefix_cache_metrics.restored_token_count,
                        f"{cls._required_ttft(metrics) / 1_000:.3f}",
                        f"{metrics.end_to_end_latency_microseconds / 1_000_000:.3f}",
                    )
                )
        return tabulate(
            rows,
            headers=(
                "Case",
                "Request",
                "Device",
                "Input",
                "Output",
                "Restored",
                "TTFT (ms)",
                "E2E (s)",
            ),
            tablefmt="simple",
        )

    @classmethod
    def _report_machine_readable(
        cls,
        results: dict[str, CapacityResult],
    ) -> None:
        for definition in CASES:
            result = results[definition.name]
            metrics = cls._summary_metrics(result)
            logger.info(
                "REQUEST_ORCHESTRATION_CAPACITY_RESULT case={} backend_requests={} "
                "workload={} routing_mode={} request_count={} wall_seconds={:.6f} "
                "output_tokens_per_second={:.6f} "
                "median_ttft_ms={:.6f} p95_ttft_ms={:.6f} "
                "median_e2e_seconds={:.6f} p95_e2e_seconds={:.6f} "
                "restored_tokens={}",
                definition.name,
                cls._backend_distribution(result)
                .lower()
                .replace(" ", ""),
                definition.workload.name,
                definition.routing_mode,
                definition.workload.request_count,
                result.wall_time_seconds,
                metrics["output_tokens_per_second"],
                metrics["median_ttft_milliseconds"],
                metrics["p95_ttft_milliseconds"],
                metrics["median_e2e_seconds"],
                metrics["p95_e2e_seconds"],
                metrics["restored_token_count"],
            )
            for request in result.requests:
                request_metrics = request.metrics
                logger.info(
                    "REQUEST_ORCHESTRATION_CAPACITY_REQUEST case={} request={} "
                    "input_tokens={} output_tokens={} backend_id={} restored_tokens={} "
                    "prefill_us={} decode_us={} ttft_us={} e2e_us={} request_id={} "
                    "submitted_ms={:.3f}",
                    definition.name,
                    request.label,
                    request_metrics.input_token_count,
                    request_metrics.output_token_count,
                    request_metrics.backend_id,
                    request_metrics.target_prefix_cache_metrics.restored_token_count,
                    request_metrics.prefill_duration_microseconds,
                    request_metrics.decode_duration_microseconds,
                    cls._required_ttft(request_metrics),
                    request_metrics.end_to_end_latency_microseconds,
                    request.request_id,
                    request.submission_offset_seconds * 1_000,
                )
