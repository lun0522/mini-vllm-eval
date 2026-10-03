"""Compare serial admission with continuous batching."""

from __future__ import annotations

import subprocess
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import fmean
from threading import Event
from time import perf_counter
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import DraftTokenMetrics
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_LARGE_MODEL
from benchmarks.base import QWEN_SMALL_DRAFT_MODEL
from benchmarks.example_prompts import EXAMPLE_LONG_PROMPTS
from benchmarks.example_prompts import EXAMPLE_SHORT_PROMPTS
from proto_loader import ProtoModules


PAIR_WORKLOAD = "Two-request handoff"
HETEROGENEOUS_WORKLOAD = "Heterogeneous four-request workload"
PAIR_SERIAL = "Two requests / limit 1"
PAIR_BATCHED = "Two requests / limit 2"
HETEROGENEOUS_SERIAL = "Four requests / limit 1"
HETEROGENEOUS_BATCHED = "Four requests / limit 4"
MAX_BATCHED_TOKEN_COUNT = 1024
TARGET_KV_CACHE_SIZE_BYTES = 1024 * 1024 * 1024
PAIR_MAX_NEW_TOKENS = 512
HETEROGENEOUS_MAX_NEW_TOKENS = 256
MINIMUM_LONG_TO_SHORT_INPUT_RATIO = 3
GPU_ENVIRONMENT = (("MINI_VLLM_METAL_GEMV_MAX_ROWS", "4"),)


@dataclass(frozen=True)
class RequestDefinition:
    role: str
    prompt: str
    max_new_tokens: int
    is_follower: bool
    prefill_length: str


@dataclass(frozen=True)
class WorkloadDefinition:
    name: str
    requests: tuple[RequestDefinition, ...]


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    workload: WorkloadDefinition
    max_active_request_count: int


@dataclass(frozen=True)
class ContinuousBatchingRequestResult:
    role: str
    output: str
    metrics: GenerationMetrics


@dataclass(frozen=True)
class ContinuousBatchingCaseResult:
    workload: str
    max_active_request_count: int
    requests: tuple[ContinuousBatchingRequestResult, ...]
    wall_time_seconds: float


PAIR_REQUESTS = WorkloadDefinition(
    PAIR_WORKLOAD,
    (
        RequestDefinition(
            "Anchor (short prefill)",
            EXAMPLE_SHORT_PROMPTS[0],
            PAIR_MAX_NEW_TOKENS,
            False,
            "short",
        ),
        RequestDefinition(
            "Follower (short prefill)",
            EXAMPLE_SHORT_PROMPTS[1],
            PAIR_MAX_NEW_TOKENS,
            True,
            "short",
        ),
    ),
)
HETEROGENEOUS_REQUESTS = WorkloadDefinition(
    HETEROGENEOUS_WORKLOAD,
    (
        RequestDefinition(
            "Anchor (short prefill)",
            EXAMPLE_SHORT_PROMPTS[2],
            HETEROGENEOUS_MAX_NEW_TOKENS,
            False,
            "short",
        ),
        RequestDefinition(
            "Follower 1 (long prefill)",
            EXAMPLE_LONG_PROMPTS[0],
            HETEROGENEOUS_MAX_NEW_TOKENS,
            True,
            "long",
        ),
        RequestDefinition(
            "Follower 2 (short prefill)",
            EXAMPLE_SHORT_PROMPTS[3],
            HETEROGENEOUS_MAX_NEW_TOKENS,
            True,
            "short",
        ),
        RequestDefinition(
            "Follower 3 (long prefill)",
            EXAMPLE_LONG_PROMPTS[1],
            HETEROGENEOUS_MAX_NEW_TOKENS,
            True,
            "long",
        ),
    ),
)
CASE_DEFINITIONS = (
    CaseDefinition(PAIR_SERIAL, PAIR_REQUESTS, 1),
    CaseDefinition(PAIR_BATCHED, PAIR_REQUESTS, 2),
    CaseDefinition(HETEROGENEOUS_SERIAL, HETEROGENEOUS_REQUESTS, 1),
    CaseDefinition(HETEROGENEOUS_BATCHED, HETEROGENEOUS_REQUESTS, 4),
)
CASE_DEFINITIONS_BY_NAME = {
    definition.name: definition for definition in CASE_DEFINITIONS
}
COMPARISONS = (
    (PAIR_WORKLOAD, PAIR_SERIAL, PAIR_BATCHED),
    (
        HETEROGENEOUS_WORKLOAD,
        HETEROGENEOUS_SERIAL,
        HETEROGENEOUS_BATCHED,
    ),
)


class ContinuousBatchingBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return self._server_flags(max_active_request_count=1)

    def cases(self) -> tuple[BenchmarkCase, ...]:
        return tuple(
            BenchmarkCase(
                definition.name,
                tuple(
                    self._server_flags(
                        max_active_request_count=definition.max_active_request_count
                    )
                ),
                GPU_ENVIRONMENT,
            )
            for definition in CASE_DEFINITIONS
        )

    @staticmethod
    def _server_flags(max_active_request_count: int) -> list[str]:
        return [
            "--model",
            QWEN_LARGE_MODEL,
            "--draft-model",
            QWEN_SMALL_DRAFT_MODEL,
            "--kv-cache-config",
            (
                "kv_cache_type: KV_CACHE_TYPE_PAGED per_page_token_count: 16 "
                f"target_kv_cache_size_bytes: {TARGET_KV_CACHE_SIZE_BYTES}"
            ),
            "--inference-device",
            "gpu",
            "--activation-dtype",
            "f32",
            "--scheduler-config",
            (
                f"max_batched_token_count: {MAX_BATCHED_TOKEN_COUNT} "
                f"max_active_request_count: {max_active_request_count} "
                "scheduling_policy: SCHEDULING_POLICY_FIRST_COME_FIRST_SERVED"
            ),
        ]

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> ContinuousBatchingCaseResult:
        definition = CASE_DEFINITIONS_BY_NAME[case.name]
        self._send_request(
            client,
            proto,
            case.name,
            RequestDefinition(
                "Warm-up",
                EXAMPLE_SHORT_PROMPTS[7],
                1,
                False,
                "short",
            ),
        )
        return self._run_workload(client, proto, definition)

    def _run_workload(
        self,
        client: Any,
        proto: ProtoModules,
        case: CaseDefinition,
    ) -> ContinuousBatchingCaseResult:
        first_text_received = Event()
        anchor, *followers = case.workload.requests
        started_at = perf_counter()
        with ThreadPoolExecutor(max_workers=len(case.workload.requests)) as executor:
            anchor_future = executor.submit(
                self._send_request,
                client,
                proto,
                case.name,
                anchor,
                first_text_received,
            )
            self._wait_for_first_text(first_text_received, anchor_future)
            logger.warning(
                "{} anchor emitted its first text; submitting {} follower request(s)",
                case.name,
                len(followers),
            )
            follower_futures = [
                executor.submit(
                    self._send_request,
                    client,
                    proto,
                    case.name,
                    request,
                )
                for request in followers
            ]
            requests = (
                anchor_future.result(),
                *(future.result() for future in follower_futures),
            )
        return ContinuousBatchingCaseResult(
            workload=case.workload.name,
            max_active_request_count=case.max_active_request_count,
            requests=requests,
            wall_time_seconds=perf_counter() - started_at,
        )

    @staticmethod
    def _wait_for_first_text(
        first_text_received: Event,
        anchor_future: Future[ContinuousBatchingRequestResult],
    ) -> None:
        while not first_text_received.wait(timeout=0.1):
            if anchor_future.done():
                anchor_future.result()
                raise RuntimeError("anchor request completed without streaming text")

    def _send_request(
        self,
        client: Any,
        proto: ProtoModules,
        case_name: str,
        definition: RequestDefinition,
        first_text_received: Event | None = None,
    ) -> ContinuousBatchingRequestResult:
        request = proto.request_handler.GenerateText(
            prompt=definition.prompt,
            max_new_tokens=definition.max_new_tokens,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        logger.warning(
            "Sending {} {} request with max_new_tokens={}",
            case_name,
            definition.role,
            definition.max_new_tokens,
        )
        output_parts: list[str] = []
        metrics = None
        for response in client.GenerateText(request):
            event = response.WhichOneof("event")
            if event == "text":
                output_parts.append(response.text)
                if first_text_received is not None:
                    first_text_received.set()
            elif event == "stats":
                metrics = self.generation_metrics(response.stats)
        if metrics is None:
            raise RuntimeError(
                f"{case_name} {definition.role} completed without statistics"
            )
        return ContinuousBatchingRequestResult(
            role=definition.role,
            output="".join(output_parts),
            metrics=metrics,
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_case = self._validated_results(results)
        logger.info(
            "Continuous-batching request metrics:\n{}",
            self._request_metrics_table(results_by_case),
        )
        logger.info(
            "Continuous-batching case summary:\n{}",
            self._case_summary_table(results_by_case),
        )
        logger.info(
            "Continuous-batching comparison:\n{}",
            self._comparison_table(results_by_case),
        )

    @classmethod
    def _validated_results(
        cls,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> dict[str, ContinuousBatchingCaseResult]:
        results_by_case: dict[str, ContinuousBatchingCaseResult] = {}
        for case, result in results:
            if not isinstance(result, ContinuousBatchingCaseResult):
                raise RuntimeError(
                    f"case {case.name} did not return continuous-batching results"
                )
            if case.name in results_by_case:
                raise RuntimeError(f"received duplicate result for case {case.name}")
            definition = CASE_DEFINITIONS_BY_NAME.get(case.name)
            if definition is None:
                raise RuntimeError(f"received unexpected case {case.name}")
            if (
                result.workload != definition.workload.name
                or result.max_active_request_count
                != definition.max_active_request_count
            ):
                raise RuntimeError(f"case {case.name} returned mismatched metadata")
            cls._validate_case(definition, result)
            results_by_case[case.name] = result

        expected_cases = set(CASE_DEFINITIONS_BY_NAME)
        if set(results_by_case) != expected_cases:
            raise RuntimeError("continuous-batching benchmark requires all four cases")
        cls._validate_matching_input_counts(results_by_case)
        cls._validate_heterogeneous_input_lengths(results_by_case)
        return results_by_case

    @classmethod
    def _validate_case(
        cls,
        definition: CaseDefinition,
        result: ContinuousBatchingCaseResult,
    ) -> None:
        expected_requests = definition.workload.requests
        if len(result.requests) != len(expected_requests):
            raise RuntimeError(
                f"case {definition.name} returned an unexpected request count"
            )
        for request_definition, request_result in zip(
            expected_requests, result.requests, strict=True
        ):
            if request_result.role != request_definition.role:
                raise RuntimeError(
                    f"case {definition.name} returned requests out of order"
                )
            cls._validate_request(
                definition.name,
                request_definition,
                request_result,
            )
        if result.wall_time_seconds <= 0:
            raise RuntimeError(f"case {definition.name} returned invalid wall time")

    @classmethod
    def _validate_request(
        cls,
        case_name: str,
        definition: RequestDefinition,
        result: ContinuousBatchingRequestResult,
    ) -> None:
        metrics = result.metrics
        if not result.output:
            raise RuntimeError(f"{case_name} {definition.role} produced no text")
        if metrics.output_token_count != definition.max_new_tokens:
            raise RuntimeError(
                f"{case_name} {definition.role} produced {metrics.output_token_count} "
                f"tokens instead of {definition.max_new_tokens}"
            )
        ttft = metrics.time_to_first_token_microseconds
        e2e = metrics.end_to_end_latency_microseconds
        if ttft is None or e2e is None or ttft <= 0 or e2e < ttft:
            raise RuntimeError(
                f"{case_name} {definition.role} returned invalid latency"
            )
        draft_metrics = metrics.draft_token_metrics
        if draft_metrics is None:
            raise RuntimeError(
                f"{case_name} {definition.role} omitted draft statistics"
            )
        cls._validate_draft_metrics(case_name, definition.role, draft_metrics)

    @staticmethod
    def _validate_draft_metrics(
        case_name: str,
        role: str,
        metrics: DraftTokenMetrics,
    ) -> None:
        if metrics.proposed_token_count <= 0 or not (
            0 <= metrics.accepted_token_count <= metrics.proposed_token_count
        ):
            raise RuntimeError(f"{case_name} {role} returned invalid draft totals")
        histogram = metrics.selected_token_count_histogram
        if len(histogram) != 1 or histogram[0][0] != 4 or histogram[0][1] <= 0:
            raise RuntimeError(f"{case_name} {role} did not use fixed-4 speculation")

    @staticmethod
    def _validate_matching_input_counts(
        results_by_case: dict[str, ContinuousBatchingCaseResult],
    ) -> None:
        for _workload, serial_name, batched_name in COMPARISONS:
            serial = results_by_case[serial_name]
            batched = results_by_case[batched_name]
            serial_counts = {
                request.role: request.metrics.input_token_count
                for request in serial.requests
            }
            batched_counts = {
                request.role: request.metrics.input_token_count
                for request in batched.requests
            }
            if serial_counts != batched_counts:
                raise RuntimeError(
                    "input token counts differ between "
                    f"{serial_name} and {batched_name}"
                )

    @staticmethod
    def _validate_heterogeneous_input_lengths(
        results_by_case: dict[str, ContinuousBatchingCaseResult],
    ) -> None:
        result = results_by_case[HETEROGENEOUS_SERIAL]
        definition = CASE_DEFINITIONS_BY_NAME[HETEROGENEOUS_SERIAL]
        counts_by_role = {
            request.role: request.metrics.input_token_count
            for request in result.requests
        }
        short_counts = [
            counts_by_role[request.role]
            for request in definition.workload.requests
            if request.prefill_length == "short"
        ]
        long_counts = [
            counts_by_role[request.role]
            for request in definition.workload.requests
            if request.prefill_length == "long"
        ]
        if min(long_counts) < MINIMUM_LONG_TO_SHORT_INPUT_RATIO * max(short_counts):
            raise RuntimeError(
                "long prefills are not substantially longer than short prefills"
            )
        follower_input_count = sum(
            counts_by_role[request.role]
            for request in definition.workload.requests
            if request.is_follower
        )
        if follower_input_count <= MAX_BATCHED_TOKEN_COUNT:
            raise RuntimeError(
                "heterogeneous follower prefills do not exceed the scheduling budget"
            )

    @classmethod
    def _request_metrics_table(
        cls,
        results_by_case: dict[str, ContinuousBatchingCaseResult],
    ) -> str:
        headers = (
            "Case",
            "Role",
            "Input",
            "Output",
            "TTFT (ms)",
            "E2E (s)",
            "Draft accepted/proposed",
        )
        rows = []
        for definition in CASE_DEFINITIONS:
            result = results_by_case[definition.name]
            for request in result.requests:
                metrics = request.metrics
                draft = cls._required_draft_metrics(metrics)
                rows.append(
                    (
                        definition.name,
                        request.role,
                        str(metrics.input_token_count),
                        str(metrics.output_token_count),
                        f"{cls._required_ttft(metrics) / 1_000:.3f}",
                        f"{cls._required_e2e(metrics) / 1_000_000:.3f}",
                        f"{draft.accepted_token_count}/{draft.proposed_token_count}",
                    )
                )
        return tabulate(rows, headers=headers, tablefmt="simple")

    @classmethod
    def _case_summary_table(
        cls,
        results_by_case: dict[str, ContinuousBatchingCaseResult],
    ) -> str:
        headers = (
            "Case",
            "Follower TTFT mean (ms)",
            "Follower TTFT max (ms)",
            "Anchor E2E (s)",
            "Wall (s)",
            "Output tok/s",
            "Draft acceptance",
        )
        rows = []
        for definition in CASE_DEFINITIONS:
            result = results_by_case[definition.name]
            follower_ttfts = [
                cls._required_ttft(request.metrics)
                for request_definition, request in zip(
                    definition.workload.requests, result.requests, strict=True
                )
                if request_definition.is_follower
            ]
            accepted, proposed = cls._aggregate_draft_totals(result)
            rows.append(
                (
                    definition.name,
                    f"{fmean(follower_ttfts) / 1_000:.3f}",
                    f"{max(follower_ttfts) / 1_000:.3f}",
                    f"{cls._required_e2e(result.requests[0].metrics) / 1_000_000:.3f}",
                    f"{result.wall_time_seconds:.3f}",
                    f"{cls._output_throughput(result):.2f}",
                    f"{accepted / proposed * 100:.1f}% ({accepted}/{proposed})",
                )
            )
        return tabulate(rows, headers=headers, tablefmt="simple")

    @classmethod
    def _comparison_table(
        cls,
        results_by_case: dict[str, ContinuousBatchingCaseResult],
    ) -> str:
        headers = (
            "Workload",
            "Follower mean TTFT reduction",
            "Output throughput speedup",
            "Anchor E2E change",
        )
        rows = []
        for workload, serial_name, batched_name in COMPARISONS:
            serial_definition = CASE_DEFINITIONS_BY_NAME[serial_name]
            batched_definition = CASE_DEFINITIONS_BY_NAME[batched_name]
            serial = results_by_case[serial_name]
            batched = results_by_case[batched_name]
            serial_ttft = cls._mean_follower_ttft(serial_definition, serial)
            batched_ttft = cls._mean_follower_ttft(batched_definition, batched)
            serial_anchor_e2e = cls._required_e2e(serial.requests[0].metrics)
            batched_anchor_e2e = cls._required_e2e(batched.requests[0].metrics)
            throughput_speedup = (
                cls._output_throughput(batched) / cls._output_throughput(serial)
            )
            rows.append(
                (
                    workload,
                    f"{(1.0 - batched_ttft / serial_ttft) * 100:+.1f}%",
                    f"{throughput_speedup:.2f}x",
                    f"{(batched_anchor_e2e / serial_anchor_e2e - 1.0) * 100:+.1f}%",
                )
            )
        return tabulate(rows, headers=headers, tablefmt="simple")

    @classmethod
    def _mean_follower_ttft(
        cls,
        definition: CaseDefinition,
        result: ContinuousBatchingCaseResult,
    ) -> float:
        return fmean(
            cls._required_ttft(request.metrics)
            for request_definition, request in zip(
                definition.workload.requests, result.requests, strict=True
            )
            if request_definition.is_follower
        )

    @staticmethod
    def _output_throughput(result: ContinuousBatchingCaseResult) -> float:
        output_token_count = sum(
            request.metrics.output_token_count for request in result.requests
        )
        return output_token_count / result.wall_time_seconds

    @classmethod
    def _aggregate_draft_totals(
        cls,
        result: ContinuousBatchingCaseResult,
    ) -> tuple[int, int]:
        accepted = 0
        proposed = 0
        for request in result.requests:
            draft = cls._required_draft_metrics(request.metrics)
            accepted += draft.accepted_token_count
            proposed += draft.proposed_token_count
        return accepted, proposed

    @staticmethod
    def _required_ttft(metrics: GenerationMetrics) -> int:
        value = metrics.time_to_first_token_microseconds
        if value is None:
            raise RuntimeError("validated request omitted TTFT")
        return value

    @staticmethod
    def _required_e2e(metrics: GenerationMetrics) -> int:
        value = metrics.end_to_end_latency_microseconds
        if value is None:
            raise RuntimeError("validated request omitted E2E latency")
        return value

    @staticmethod
    def _required_draft_metrics(metrics: GenerationMetrics) -> DraftTokenMetrics:
        value = metrics.draft_token_metrics
        if value is None:
            raise RuntimeError("validated request omitted draft statistics")
        return value
