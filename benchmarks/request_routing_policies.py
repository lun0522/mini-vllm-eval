"""Compare mixed CPU/GPU request-routing policies and throughput profiles."""

from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace
from statistics import median
from time import perf_counter
from time import sleep
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_0_5B_MODEL
from proto_loader import ProtoModules


TARGET_KV_CACHE_SIZE_BYTES = 256 * 1024 * 1024
MAX_BATCHED_TOKEN_COUNT = 2048
MAX_ACTIVE_REQUEST_COUNT = 4
REQUEST_INTERVAL_SECONDS = 0.050
PROFILE_FILE = "benchmarks/backend_throughput_profiles.textproto"


@dataclass(frozen=True)
class WorkloadShape:
    name: str
    approximate_input_token_count: int
    max_output_token_count: int


SHORT = WorkloadShape("short", 128, 16)
BACKLOG = WorkloadShape("backlog", 128, 512)
MEDIUM = WorkloadShape("medium", 512, 64)
PREFIX_PROBE = WorkloadShape("prefix", 1024, 16)
SCENARIOS = ("load", "affinity")
SCHEDULES = ("a", "b")


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    policy: str
    profile_mode: str
    policy_comparison: bool
    scenario: str = "load"
    schedule: str = "a"

    @property
    def requests(self) -> tuple[RequestDefinition, ...]:
        requests = REQUESTS_BY_SCENARIO[self.scenario]
        # Swap adjacent positions, including the two initial backlog requests.
        # Each request occupies the opposite round-robin parity in schedule B.
        if self.schedule == "b":
            return tuple(requests[index ^ 1] for index in range(len(requests)))
        return requests


POLICY_CONFIGURATIONS = (
    CaseDefinition("policy_round_robin", "round-robin", "not-used", True),
    CaseDefinition("policy_least_requests", "least-requests", "not-used", True),
    CaseDefinition(
        "policy_least_estimated_work_detailed",
        "least-estimated-work",
        "detailed",
        True,
    ),
    CaseDefinition(
        "policy_most_cached_prefix",
        "most-cached-prefix",
        "not-used",
        True,
    ),
    CaseDefinition(
        "policy_cache_aware_least_estimated_work_detailed",
        "cache-aware-least-estimated-work",
        "detailed",
        True,
    ),
    CaseDefinition(
        "profile_least_estimated_work_fallback",
        "least-estimated-work",
        "fallback",
        False,
    ),
    CaseDefinition(
        "profile_cache_aware_least_estimated_work_fallback",
        "cache-aware-least-estimated-work",
        "fallback",
        False,
    ),
)
CASES = tuple(
    replace(
        config,
        name=f"{config.name}__{scenario}__{schedule}",
        scenario=scenario,
        schedule=schedule,
    )
    for scenario in SCENARIOS
    for schedule in SCHEDULES
    for config in POLICY_CONFIGURATIONS
)
CASES_BY_NAME = {case.name: case for case in CASES}
PROFILE_COMPARISON_CASES = tuple(case for case in CASES if "estimated-work" in case.policy)


@dataclass(frozen=True)
class RequestDefinition:
    label: str
    kind: str
    shape: WorkloadShape
    prompt: str


@dataclass(frozen=True)
class RequestResult:
    definition: RequestDefinition
    metrics: GenerationMetrics
    request_id: int
    submission_offset_seconds: float


@dataclass(frozen=True)
class RoutingPoliciesResult:
    case_name: str
    requests: tuple[RequestResult, ...]
    wall_time_seconds: float


def _shared_prefix(word_count: int) -> str:
    words = " ".join("routing" for _ in range(word_count))
    return (
        "Shared routing study context. The following stable sequence is reused by "
        f"several requests: {words}"
    )


def _shared_prompt(shape: WorkloadShape) -> str:
    word_count = max(1, shape.approximate_input_token_count - 48)
    return (
        f"{_shared_prefix(word_count)}\n"
        f"Shared {shape.name} question: summarize the routing study."
    )


def _unique_prompt(shape: WorkloadShape, label: str) -> str:
    word_count = max(1, shape.approximate_input_token_count - 48)
    words = " ".join("inference" for _ in range(word_count))
    return (
        f"{label}: Unique routing study context. This request must not reuse another "
        f"workload's body: {words}\nSummarize this unique routing study."
    )


def _request(label: str, shape: WorkloadShape, kind: str = "unique") -> RequestDefinition:
    prompt = _shared_prompt(shape) if kind == "shared" else _unique_prompt(shape, label)
    return RequestDefinition(label, kind, shape, prompt)


# Short contexts keep the decode budget affordable. Both schedules establish
# work first, then offer cheap requests that can use otherwise idle CPU capacity.
REQUESTS_BY_SCENARIO = {
    "load": (
        _request("amber-backlog", BACKLOG),
        _request("birch-backlog", BACKLOG),
        _request("cedar-short", SHORT),
        _request("delta-medium", MEDIUM),
        _request("elm-short", SHORT),
        _request("fern-short", SHORT),
        _request("grove-short", SHORT),
        _request("hazel-medium", MEDIUM),
        _request("iris-short", SHORT),
        _request("juniper-short", SHORT),
        _request("kelp-short", SHORT),
        _request("larch-short", SHORT),
    ),
    "affinity": (
        _request("amber-backlog", BACKLOG),
        _request("birch-backlog", BACKLOG),
        _request("cedar-short-shared", SHORT, "shared"),
        _request("delta-prefix-unique", PREFIX_PROBE),
        _request("elm-short-unique", SHORT),
        _request("fern-prefix-shared", PREFIX_PROBE, "shared"),
        _request("grove-short-unique", SHORT),
        _request("hazel-short-shared", SHORT, "shared"),
        _request("iris-prefix-shared", PREFIX_PROBE, "shared"),
        _request("juniper-short-unique", SHORT),
        _request("kelp-short-shared", SHORT, "shared"),
        _request("larch-prefix-unique", PREFIX_PROBE),
    ),
}


class RequestRoutingPoliciesBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return self._server_flags(CASES[0])

    def cases(self) -> tuple[BenchmarkCase, ...]:
        return tuple(
            BenchmarkCase(case.name, tuple(self._server_flags(case))) for case in CASES
        )

    @staticmethod
    def _server_flags(case: CaseDefinition) -> list[str]:
        routing_policy = (
            "REQUEST_ROUTING_POLICY_" + case.policy.upper().replace("-", "_")
        )
        flags = [
            "--model",
            QWEN_0_5B_MODEL,
            "--inference-device",
            "cpu",
            "--inference-device",
            "gpu",
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
            f"routing_policy: {routing_policy}",
        ]
        if case.profile_mode == "detailed":
            flags.extend(("--backend-throughput-profiles-file", PROFILE_FILE))
        return flags

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> RoutingPoliciesResult:
        definition = CASES_BY_NAME[case.name]
        requests_to_send = definition.requests
        if definition.scenario == "affinity":
            self._seed_shared_prefix(client, proto, case.name)
        logger.info(
            "REQUEST_ROUTING_POLICIES_REPLAY case={} version=2 interval_ms=50 "
            "preprocessing_threads=1 order={}",
            case.name,
            ",".join(request.label for request in requests_to_send),
        )
        started_at = perf_counter()
        with ThreadPoolExecutor(max_workers=len(requests_to_send)) as executor:
            futures = []
            for position, definition in enumerate(requests_to_send):
                # Fixed start times keep response handling concurrent without racing
                # client threads to submit requests. Do not wait for generation.
                delay = started_at + position * REQUEST_INTERVAL_SECONDS - perf_counter()
                if delay > 0:
                    sleep(delay)
                submission_offset = perf_counter() - started_at
                responses = self._send_request(client, proto, case.name, definition)
                futures.append(
                    executor.submit(
                        self._collect_response,
                        case.name,
                        definition,
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
        return RoutingPoliciesResult(
            case_name=case.name,
            requests=requests,
            wall_time_seconds=wall_time_seconds,
        )

    @staticmethod
    def _seed_shared_prefix(client: Any, proto: ProtoModules, case_name: str) -> None:
        prompt = (
            f"{_shared_prefix(PREFIX_PROBE.approximate_input_token_count)}\n"
            "Seed question: acknowledge the routing study."
        )
        logger.warning("Seeding the shared prefix for {}", case_name)
        request = proto.request_handler.GenerateText(
            prompt=prompt,
            max_new_tokens=1,
            repeat_penalty=1.0,
            repeat_last_n=0,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        stats = None
        for response in client.GenerateText(request):
            if response.WhichOneof("event") == "stats":
                stats = response.stats
        if stats is None:
            raise RuntimeError(f"{case_name} shared-prefix seed omitted statistics")
        logger.info(
            "REQUEST_ROUTING_POLICIES_SEED case={} request_id={} backend_id={} "
            "input_tokens={} indexed_tokens={}",
            case_name,
            stats.request_id,
            stats.backend_id,
            stats.input_token_count,
            stats.target_prefix_cache_stats.newly_indexed_token_count,
        )

    @staticmethod
    def _send_request(
        client: Any,
        proto: ProtoModules,
        case_name: str,
        definition: RequestDefinition,
    ) -> Any:
        logger.info("Sending {} request for {}", definition.label, case_name)
        request = proto.request_handler.GenerateText(
            prompt=definition.prompt,
            max_new_tokens=definition.shape.max_output_token_count,
            repeat_penalty=1.0,
            repeat_last_n=0,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        return client.GenerateText(request)

    @classmethod
    def _collect_response(
        cls,
        case_name: str,
        definition: RequestDefinition,
        responses: Any,
        submission_offset_seconds: float,
    ) -> RequestResult:
        stats = None
        for response in responses:
            if response.WhichOneof("event") == "stats":
                stats = response.stats
        if stats is None:
            raise RuntimeError(f"{case_name} {definition.label} omitted statistics")
        metrics = cls.generation_metrics(stats)
        if metrics.output_token_count != definition.shape.max_output_token_count:
            raise RuntimeError(
                f"{case_name} {definition.label} produced {metrics.output_token_count} "
                f"tokens instead of {definition.shape.max_output_token_count}"
            )
        return RequestResult(definition, metrics, stats.request_id, submission_offset_seconds)

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_case = self._validated_results(results)
        logger.info(
            "Request-routing policy comparison:\n{}",
            self._summary_table(
                tuple(case for case in CASES if case.policy_comparison),
                results_by_case,
            ),
        )
        logger.info(
            "Request-routing throughput-profile comparison:\n{}",
            self._summary_table(
                PROFILE_COMPARISON_CASES,
                results_by_case,
            ),
        )
        logger.info(
            "Request-routing backend distribution:\n{}",
            self._backend_table(results_by_case),
        )
        logger.info(
            "Request-routing request metrics:\n{}",
            self._request_table(results_by_case),
        )
        self._report_machine_readable(results_by_case)

    @classmethod
    def _validated_results(
        cls,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> dict[str, RoutingPoliciesResult]:
        results_by_case: dict[str, RoutingPoliciesResult] = {}
        for benchmark_case, result in results:
            if not isinstance(result, RoutingPoliciesResult):
                raise RuntimeError(
                    f"case {benchmark_case.name} returned unexpected results"
                )
            if result.case_name != benchmark_case.name:
                raise RuntimeError(f"case {benchmark_case.name} returned mismatched results")
            if benchmark_case.name in results_by_case:
                raise RuntimeError(f"received duplicate result for {benchmark_case.name}")
            definitions = CASES_BY_NAME[benchmark_case.name].requests
            if result.wall_time_seconds <= 0 or len(result.requests) != len(definitions):
                raise RuntimeError(f"case {benchmark_case.name} returned invalid results")
            for definition, request in zip(definitions, result.requests, strict=True):
                cls._validate_request(benchmark_case.name, definition, request)
            results_by_case[benchmark_case.name] = result
        if set(results_by_case) != set(CASES_BY_NAME):
            raise RuntimeError("request_routing_policies benchmark requires all 28 cases")
        cls._validate_matching_input_token_counts(results_by_case)
        return results_by_case

    @staticmethod
    def _validate_request(
        case_name: str,
        definition: RequestDefinition,
        request: RequestResult,
    ) -> None:
        metrics = request.metrics
        if request.definition != definition:
            raise RuntimeError(f"{case_name} returned requests in an unexpected order")
        if metrics.backend_id not in (1, 2):
            raise RuntimeError(
                f"{case_name} {definition.label} used invalid backend {metrics.backend_id}"
            )
        if metrics.input_token_count <= 0:
            raise RuntimeError(f"{case_name} {definition.label} reported no input tokens")
        if metrics.output_token_count != definition.shape.max_output_token_count:
            raise RuntimeError(f"{case_name} {definition.label} returned invalid output")
        if (
            metrics.time_to_first_token_microseconds is None
            or metrics.time_to_first_token_microseconds <= 0
            or metrics.end_to_end_latency_microseconds
            < metrics.time_to_first_token_microseconds
        ):
            raise RuntimeError(f"{case_name} {definition.label} returned invalid latency")
        if metrics.draft_token_metrics is not None:
            raise RuntimeError(f"{case_name} unexpectedly used a draft model")

    @staticmethod
    def _validate_matching_input_token_counts(
        results_by_case: dict[str, RoutingPoliciesResult],
    ) -> None:
        expected = {}
        for case in CASES:
            for request in results_by_case[case.name].requests:
                key = (case.scenario, request.definition.label)
                count = request.metrics.input_token_count
                if expected.setdefault(key, count) != count:
                    raise RuntimeError(f"input token counts differ for {case.name}/{key}")

    @classmethod
    def _summary_table(
        cls,
        cases: tuple[CaseDefinition, ...],
        results_by_case: dict[str, RoutingPoliciesResult],
    ) -> str:
        rows = []
        for case in cases:
            result = results_by_case[case.name]
            metrics = cls._summary_metrics(result)
            rows.append(
                (
                    case.name,
                    case.profile_mode,
                    cls._backend_distribution(result),
                    f"{result.wall_time_seconds:.3f}",
                    f"{metrics['output_tokens_per_second']:.2f}",
                    f"{metrics['median_ttft_milliseconds']:.3f}",
                    f"{metrics['p95_ttft_milliseconds']:.3f}",
                    f"{metrics['median_e2e_seconds']:.3f}",
                    f"{metrics['p95_e2e_seconds']:.3f}",
                    metrics["shared_restored_token_count"],
                    metrics["unique_restored_token_count"],
                )
            )
        return tabulate(
            rows,
            headers=(
                "Case",
                "Profiles",
                "Backend requests",
                "Wall (s)",
                "Output tok/s",
                "Median TTFT (ms)",
                "p95 TTFT (ms)",
                "Median E2E (s)",
                "p95 E2E (s)",
                "Shared restored",
                "Unique restored",
            ),
            tablefmt="simple",
        )

    @staticmethod
    def _backend_distribution(result: RoutingPoliciesResult) -> str:
        counts = {
            backend_id: sum(
                request.metrics.backend_id == backend_id
                for request in result.requests
            )
            for backend_id in (1, 2)
        }
        return f"CPU:{counts[1]}, GPU:{counts[2]}"

    @classmethod
    def _summary_metrics(cls, result: RoutingPoliciesResult) -> dict[str, float | int]:
        ttft_microseconds = tuple(
            cls._required_ttft(request.metrics) for request in result.requests
        )
        e2e_microseconds = tuple(
            request.metrics.end_to_end_latency_microseconds
            for request in result.requests
        )
        return {
            "output_tokens_per_second": sum(
                request.metrics.output_token_count for request in result.requests
            )
            / result.wall_time_seconds,
            "median_ttft_milliseconds": median(ttft_microseconds) / 1_000,
            "p95_ttft_milliseconds": cls._percentile(ttft_microseconds, 0.95)
            / 1_000,
            "median_e2e_seconds": median(e2e_microseconds) / 1_000_000,
            "p95_e2e_seconds": cls._percentile(e2e_microseconds, 0.95)
            / 1_000_000,
            "shared_restored_token_count": sum(
                request.metrics.target_prefix_cache_metrics.restored_token_count
                for request in result.requests
                if request.definition.kind == "shared"
            ),
            "unique_restored_token_count": sum(
                request.metrics.target_prefix_cache_metrics.restored_token_count
                for request in result.requests
                if request.definition.kind == "unique"
            ),
        }

    @staticmethod
    def _percentile(values: tuple[int, ...], percentile: float) -> int:
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[index]

    @staticmethod
    def _required_ttft(metrics: GenerationMetrics) -> int:
        if metrics.time_to_first_token_microseconds is None:
            raise RuntimeError("completed request omitted time to first token")
        return metrics.time_to_first_token_microseconds

    @staticmethod
    def _backend_table(
        results_by_case: dict[str, RoutingPoliciesResult],
    ) -> str:
        rows = []
        for case in CASES:
            result = results_by_case[case.name]
            for backend_id in (1, 2):
                requests = tuple(
                    request
                    for request in result.requests
                    if request.metrics.backend_id == backend_id
                )
                rows.append(
                    (
                        case.name,
                        case.profile_mode,
                        "CPU" if backend_id == 1 else "GPU",
                        len(requests),
                        sum(request.metrics.input_token_count for request in requests),
                        sum(request.metrics.output_token_count for request in requests),
                        sum(
                            request.metrics.target_prefix_cache_metrics.restored_token_count
                            for request in requests
                        ),
                    )
                )
        return tabulate(
            rows,
            headers=(
                "Case",
                "Profiles",
                "Device",
                "Requests",
                "Input tokens",
                "Output tokens",
                "Restored tokens",
            ),
            tablefmt="simple",
        )

    @classmethod
    def _request_table(
        cls,
        results_by_case: dict[str, RoutingPoliciesResult],
    ) -> str:
        rows = []
        for case in CASES:
            for request in results_by_case[case.name].requests:
                metrics = request.metrics
                rows.append(
                    (
                        case.name,
                        case.profile_mode,
                        request.definition.label,
                        metrics.input_token_count,
                        metrics.output_token_count,
                        "CPU" if metrics.backend_id == 1 else "GPU",
                        metrics.target_prefix_cache_metrics.restored_token_count,
                        f"{cls._required_ttft(metrics) / 1_000:.3f}",
                        f"{metrics.end_to_end_latency_microseconds / 1_000_000:.3f}",
                    )
                )
        return tabulate(
            rows,
            headers=(
                "Case",
                "Profiles",
                "Request",
                "Input",
                "Output",
                "Device",
                "Restored",
                "TTFT (ms)",
                "E2E (s)",
            ),
            tablefmt="simple",
        )

    @classmethod
    def _report_machine_readable(
        cls,
        results_by_case: dict[str, RoutingPoliciesResult],
    ) -> None:
        for case in CASES:
            result = results_by_case[case.name]
            metrics = cls._summary_metrics(result)
            logger.info(
                "REQUEST_ROUTING_POLICIES_RESULT "
                "case={} policy={} profiles={} backend_requests={} wall_seconds={:.6f} "
                "output_tokens_per_second={:.6f} median_ttft_ms={:.6f} "
                "p95_ttft_ms={:.6f} median_e2e_seconds={:.6f} p95_e2e_seconds={:.6f} "
                "shared_restored_tokens={} unique_restored_tokens={}",
                case.name,
                case.policy,
                case.profile_mode,
                cls._backend_distribution(result).lower().replace(" ", ""),
                result.wall_time_seconds,
                metrics["output_tokens_per_second"],
                metrics["median_ttft_milliseconds"],
                metrics["p95_ttft_milliseconds"],
                metrics["median_e2e_seconds"],
                metrics["p95_e2e_seconds"],
                metrics["shared_restored_token_count"],
                metrics["unique_restored_token_count"],
            )
            for request in result.requests:
                request_metrics = request.metrics
                logger.info(
                    "REQUEST_ROUTING_POLICIES_REQUEST "
                    "case={} policy={} profiles={} request={} kind={} input_tokens={} "
                    "output_tokens={} backend_id={} restored_tokens={} prefill_us={} "
                    "decode_us={} ttft_us={} e2e_us={} request_id={} submitted_ms={:.3f}",
                    case.name,
                    case.policy,
                    case.profile_mode,
                    request.definition.label,
                    request.definition.kind,
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
