"""Compare FCFS, shortest-prefill-first, and round-robin scheduling."""

from __future__ import annotations

import subprocess
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import fmean
from threading import Event
from time import perf_counter
from time import sleep
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_LARGE_MODEL
from benchmarks.example_prompts import EXAMPLE_LONG_PROMPTS
from benchmarks.example_prompts import EXAMPLE_SHORT_PROMPTS
from proto_loader import ProtoModules


POLICIES = (
    "first-come-first-served",
    "shortest-prefill-first",
    "round-robin",
)
MAX_BATCHED_TOKEN_COUNT = 128
MAX_ACTIVE_REQUEST_COUNT = 7
TARGET_KV_CACHE_SIZE_BYTES = 1024 * 1024 * 1024
ANCHOR_MAX_NEW_TOKENS = 192
FOLLOWER_MAX_NEW_TOKENS = 16
FOLLOWER_SUBMISSION_GAP_SECONDS = 0.5
MINIMUM_LONG_TO_SHORT_INPUT_RATIO = 3
GPU_ENVIRONMENT = (("MINI_VLLM_METAL_GEMV_MAX_ROWS", "4"),)


@dataclass(frozen=True)
class RequestDefinition:
    role: str
    prompt: str
    max_new_tokens: int
    is_anchor: bool
    prefill_length: str


@dataclass(frozen=True)
class WorkloadDefinition:
    name: str
    requests: tuple[RequestDefinition, ...]


@dataclass(frozen=True)
class SchedulingRequestResult:
    role: str
    output: str
    metrics: GenerationMetrics


@dataclass(frozen=True)
class SchedulingWorkloadResult:
    name: str
    requests: tuple[SchedulingRequestResult, ...]
    wall_time_seconds: float


@dataclass(frozen=True)
class SchedulingPolicyResult:
    policy: str
    workloads: tuple[SchedulingWorkloadResult, ...]


HEAD_OF_LINE_WORKLOAD = WorkloadDefinition(
    "head-of-line",
    (
        RequestDefinition(
            "Anchor",
            EXAMPLE_SHORT_PROMPTS[0],
            ANCHOR_MAX_NEW_TOKENS,
            True,
            "short",
        ),
        RequestDefinition(
            "Oldest long follower",
            EXAMPLE_LONG_PROMPTS[0],
            FOLLOWER_MAX_NEW_TOKENS,
            False,
            "long",
        ),
        RequestDefinition(
            "Short follower",
            EXAMPLE_SHORT_PROMPTS[1],
            FOLLOWER_MAX_NEW_TOKENS,
            False,
            "short",
        ),
    ),
)

LONG_FAIRNESS_WORKLOAD = WorkloadDefinition(
    "long-fairness",
    (
        RequestDefinition(
            "Anchor",
            EXAMPLE_SHORT_PROMPTS[2],
            ANCHOR_MAX_NEW_TOKENS,
            True,
            "short",
        ),
        RequestDefinition(
            "Oldest long follower",
            EXAMPLE_LONG_PROMPTS[1],
            FOLLOWER_MAX_NEW_TOKENS,
            False,
            "long",
        ),
        RequestDefinition(
            "Long peer 1",
            EXAMPLE_LONG_PROMPTS[2],
            FOLLOWER_MAX_NEW_TOKENS,
            False,
            "long",
        ),
        RequestDefinition(
            "Long peer 2",
            EXAMPLE_LONG_PROMPTS[3],
            FOLLOWER_MAX_NEW_TOKENS,
            False,
            "long",
        ),
    ),
)

WORKLOADS = (HEAD_OF_LINE_WORKLOAD, LONG_FAIRNESS_WORKLOAD)


class SchedulingPoliciesBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return self._server_flags(POLICIES[0])

    def cases(self) -> tuple[BenchmarkCase, ...]:
        return tuple(
            BenchmarkCase(
                policy,
                tuple(self._server_flags(policy)),
                GPU_ENVIRONMENT,
            )
            for policy in POLICIES
        )

    @staticmethod
    def _server_flags(policy: str) -> list[str]:
        return [
            "--model",
            QWEN_LARGE_MODEL,
            "--kv-cache-type",
            "paged:16",
            "--inference-device",
            "gpu",
            "--activation-dtype",
            "f32",
            "--target-kv-cache-size-bytes",
            str(TARGET_KV_CACHE_SIZE_BYTES),
            "--max-batched-token-count",
            str(MAX_BATCHED_TOKEN_COUNT),
            "--max-active-request-count",
            str(MAX_ACTIVE_REQUEST_COUNT),
            "--input-preprocessing-thread-count",
            "1",
            "--scheduling-policy",
            policy,
        ]

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> SchedulingPolicyResult:
        self._send_request(
            client,
            proto,
            case.name,
            "warm-up",
            RequestDefinition(
                "Warm-up",
                EXAMPLE_SHORT_PROMPTS[7],
                1,
                True,
                "short",
            ),
        )
        return SchedulingPolicyResult(
            policy=case.name,
            workloads=tuple(
                self._run_workload(client, proto, case.name, workload)
                for workload in WORKLOADS
            ),
        )

    def _run_workload(
        self,
        client: Any,
        proto: ProtoModules,
        policy: str,
        workload: WorkloadDefinition,
    ) -> SchedulingWorkloadResult:
        anchor, oldest_follower, *later_followers = workload.requests
        anchor_first_text = Event()
        oldest_follower_submission_started = Event()
        started_at = perf_counter()
        with ThreadPoolExecutor(max_workers=len(workload.requests)) as executor:
            anchor_future = executor.submit(
                self._send_request,
                client,
                proto,
                policy,
                workload.name,
                anchor,
                anchor_first_text,
            )
            self._wait_for_first_text(anchor_first_text, anchor_future)
            logger.warning(
                "{} {} anchor emitted its first text; submitting the oldest follower",
                policy,
                workload.name,
            )
            oldest_follower_future = executor.submit(
                self._send_request,
                client,
                proto,
                policy,
                workload.name,
                oldest_follower,
                None,
                oldest_follower_submission_started,
            )
            if not oldest_follower_submission_started.wait(timeout=5):
                raise RuntimeError("oldest follower did not begin submission")
            sleep(FOLLOWER_SUBMISSION_GAP_SECONDS)
            logger.warning(
                "{} {} oldest follower was submitted; submitting {} later follower(s)",
                policy,
                workload.name,
                len(later_followers),
            )
            later_futures = [
                executor.submit(
                    self._send_request,
                    client,
                    proto,
                    policy,
                    workload.name,
                    request,
                )
                for request in later_followers
            ]
            requests = (
                anchor_future.result(),
                oldest_follower_future.result(),
                *(future.result() for future in later_futures),
            )
        return SchedulingWorkloadResult(
            name=workload.name,
            requests=requests,
            wall_time_seconds=perf_counter() - started_at,
        )

    @staticmethod
    def _wait_for_first_text(
        first_text_received: Event,
        anchor_future: Future[SchedulingRequestResult],
    ) -> None:
        while not first_text_received.wait(timeout=0.1):
            if anchor_future.done():
                anchor_future.result()
                raise RuntimeError("anchor request completed without streaming text")

    def _send_request(
        self,
        client: Any,
        proto: ProtoModules,
        policy: str,
        workload: str,
        definition: RequestDefinition,
        first_text_received: Event | None = None,
        submission_started: Event | None = None,
    ) -> SchedulingRequestResult:
        request = proto.request_handler.GenerateText(
            prompt=definition.prompt,
            max_new_tokens=definition.max_new_tokens,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        logger.warning(
            "Sending {} {} {} request", policy, workload, definition.role
        )
        if submission_started is not None:
            submission_started.set()
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
                f"{policy} {workload} {definition.role} omitted statistics"
            )
        return SchedulingRequestResult(
            role=definition.role,
            output="".join(output_parts),
            metrics=metrics,
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_policy = self._validated_results(results)
        logger.info(
            "Scheduling-policy request metrics:\n{}",
            self._request_metrics_table(results_by_policy),
        )
        logger.info(
            "Scheduling-policy workload summary:\n{}",
            self._summary_table(results_by_policy),
        )
        logger.info(
            "Scheduling-policy comparison against FCFS:\n{}",
            self._comparison_table(results_by_policy),
        )

    @classmethod
    def _validated_results(
        cls,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> dict[str, SchedulingPolicyResult]:
        results_by_policy: dict[str, SchedulingPolicyResult] = {}
        for case, result in results:
            if not isinstance(result, SchedulingPolicyResult):
                raise RuntimeError(f"case {case.name} returned unexpected results")
            if case.name not in POLICIES or result.policy != case.name:
                raise RuntimeError(f"case {case.name} returned mismatched policy")
            if case.name in results_by_policy:
                raise RuntimeError(f"received duplicate result for {case.name}")
            cls._validate_case(result)
            results_by_policy[case.name] = result
        if set(results_by_policy) != set(POLICIES):
            raise RuntimeError("scheduling-policy benchmark requires all three policies")
        cls._validate_matching_input_counts(results_by_policy)
        for workload in WORKLOADS:
            cls._validate_workload_shape(
                workload,
                cls._result_for_workload(results_by_policy[POLICIES[0]], workload.name),
            )
        return results_by_policy

    @classmethod
    def _validate_case(cls, result: SchedulingPolicyResult) -> None:
        if tuple(workload.name for workload in result.workloads) != tuple(
            workload.name for workload in WORKLOADS
        ):
            raise RuntimeError(f"{result.policy} returned unexpected workloads")
        for definition, workload_result in zip(
            WORKLOADS, result.workloads, strict=True
        ):
            if (
                len(workload_result.requests) != len(definition.requests)
                or workload_result.wall_time_seconds <= 0
            ):
                raise RuntimeError(
                    f"{result.policy} returned invalid {definition.name} results"
                )
            for request_definition, request_result in zip(
                definition.requests, workload_result.requests, strict=True
            ):
                cls._validate_request(
                    result.policy,
                    definition.name,
                    request_definition,
                    request_result,
                )

    @staticmethod
    def _validate_request(
        policy: str,
        workload: str,
        definition: RequestDefinition,
        result: SchedulingRequestResult,
    ) -> None:
        metrics = result.metrics
        if result.role != definition.role or not result.output:
            raise RuntimeError(f"{policy} returned invalid {workload} {definition.role}")
        if metrics.output_token_count != definition.max_new_tokens:
            raise RuntimeError(
                f"{policy} {workload} {definition.role} produced "
                f"{metrics.output_token_count} tokens instead of "
                f"{definition.max_new_tokens}"
            )
        ttft = metrics.time_to_first_token_microseconds
        e2e = metrics.end_to_end_latency_microseconds
        if ttft is None or e2e is None or ttft <= 0 or e2e < ttft:
            raise RuntimeError(
                f"{policy} {workload} {definition.role} returned invalid latency"
            )
        if metrics.draft_token_metrics is not None:
            raise RuntimeError(
                f"{policy} {workload} {definition.role} unexpectedly used a draft model"
            )

    @classmethod
    def _validate_matching_input_counts(
        cls,
        results_by_policy: dict[str, SchedulingPolicyResult],
    ) -> None:
        expected = cls._input_counts(results_by_policy[POLICIES[0]])
        for policy in POLICIES[1:]:
            if cls._input_counts(results_by_policy[policy]) != expected:
                raise RuntimeError(f"input token counts differ for {policy}")

    @staticmethod
    def _input_counts(result: SchedulingPolicyResult) -> dict[tuple[str, str], int]:
        return {
            (workload.name, request.role): request.metrics.input_token_count
            for workload in result.workloads
            for request in workload.requests
        }

    @staticmethod
    def _validate_workload_shape(
        definition: WorkloadDefinition,
        result: SchedulingWorkloadResult,
    ) -> None:
        counts = {
            request.role: request.metrics.input_token_count
            for request in result.requests
        }
        long_counts = [
            counts[request.role]
            for request in definition.requests
            if request.prefill_length == "long"
        ]
        short_counts = [
            counts[request.role]
            for request in definition.requests
            if not request.is_anchor and request.prefill_length == "short"
        ]
        if short_counts and min(long_counts) < (
            MINIMUM_LONG_TO_SHORT_INPUT_RATIO * max(short_counts)
        ):
            raise RuntimeError(
                f"{definition.name} long prefill is not substantially longer"
            )
        if min(long_counts) <= MAX_BATCHED_TOKEN_COUNT:
            raise RuntimeError(
                f"{definition.name} long prefill fits in one scheduling decision"
            )
        if short_counts and max(short_counts) >= MAX_BATCHED_TOKEN_COUNT:
            raise RuntimeError(
                f"{definition.name} short prefill does not fit in one empty budget"
            )
        if definition.name == LONG_FAIRNESS_WORKLOAD.name and len(long_counts) < 3:
            raise RuntimeError("long-fairness requires at least three long followers")

    @classmethod
    def _request_metrics_table(
        cls,
        results_by_policy: dict[str, SchedulingPolicyResult],
    ) -> str:
        rows = []
        for policy in POLICIES:
            for workload in results_by_policy[policy].workloads:
                for request in workload.requests:
                    rows.append(
                        (
                            workload.name,
                            policy,
                            request.role,
                            request.metrics.input_token_count,
                            request.metrics.output_token_count,
                            f"{cls._required_ttft(request.metrics) / 1_000:.3f}",
                            f"{cls._required_e2e(request.metrics) / 1_000_000:.3f}",
                        )
                    )
        return tabulate(
            rows,
            headers=(
                "Workload",
                "Policy",
                "Role",
                "Input",
                "Output",
                "TTFT (ms)",
                "E2E (s)",
            ),
            tablefmt="simple",
        )

    @classmethod
    def _summary_table(
        cls,
        results_by_policy: dict[str, SchedulingPolicyResult],
    ) -> str:
        rows = []
        for workload_definition in WORKLOADS:
            for policy in POLICIES:
                workload = cls._result_for_workload(
                    results_by_policy[policy], workload_definition.name
                )
                metrics = cls._summary_metrics(workload_definition, workload)
                rows.append(
                    (
                        workload_definition.name,
                        policy,
                        *(f"{value:.3f}" for value in metrics[:-1]),
                        f"{metrics[-1]:.2f}",
                    )
                )
        return tabulate(
            rows,
            headers=(
                "Workload",
                "Policy",
                "Oldest TTFT (ms)",
                "Later mean TTFT (ms)",
                "Later max TTFT (ms)",
                "Follower max TTFT (ms)",
                "Follower spread (ms)",
                "Anchor E2E (s)",
                "Wall (s)",
                "Output tok/s",
            ),
            tablefmt="simple",
        )

    @classmethod
    def _comparison_table(
        cls,
        results_by_policy: dict[str, SchedulingPolicyResult],
    ) -> str:
        rows = []
        for workload in WORKLOADS:
            baseline = cls._summary_metrics(
                workload,
                cls._result_for_workload(
                    results_by_policy[POLICIES[0]], workload.name
                ),
            )
            for policy in POLICIES[1:]:
                metrics = cls._summary_metrics(
                    workload,
                    cls._result_for_workload(
                        results_by_policy[policy], workload.name
                    ),
                )
                comparable_metrics = (*metrics[:4], *metrics[5:])
                comparable_baseline = (*baseline[:4], *baseline[5:])
                rows.append(
                    (
                        workload.name,
                        policy,
                        *(
                            cls._percentage_change(value, reference)
                            for value, reference in zip(
                                comparable_metrics,
                                comparable_baseline,
                                strict=True,
                            )
                        ),
                    )
                )
        return tabulate(
            rows,
            headers=(
                "Workload",
                "Policy",
                "Oldest TTFT",
                "Later mean TTFT",
                "Later max TTFT",
                "Follower max TTFT",
                "Anchor E2E",
                "Wall",
                "Output throughput",
            ),
            tablefmt="simple",
        )

    @classmethod
    def _summary_metrics(
        cls,
        definition: WorkloadDefinition,
        result: SchedulingWorkloadResult,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        request_results = {request.role: request for request in result.requests}
        oldest_follower_ttft = (
            cls._required_ttft(request_results[definition.requests[1].role].metrics)
            / 1_000
        )
        later_follower_ttfts = [
            cls._required_ttft(request_results[request.role].metrics) / 1_000
            for request in definition.requests[2:]
        ]
        follower_ttfts = [oldest_follower_ttft, *later_follower_ttfts]
        anchor_e2e = cls._required_e2e(result.requests[0].metrics) / 1_000_000
        return (
            oldest_follower_ttft,
            fmean(later_follower_ttfts),
            max(later_follower_ttfts),
            max(follower_ttfts),
            max(follower_ttfts) - min(follower_ttfts),
            anchor_e2e,
            result.wall_time_seconds,
            cls._output_throughput(result),
        )

    @staticmethod
    def _result_for_workload(
        result: SchedulingPolicyResult,
        name: str,
    ) -> SchedulingWorkloadResult:
        return next(workload for workload in result.workloads if workload.name == name)

    @staticmethod
    def _output_throughput(result: SchedulingWorkloadResult) -> float:
        output_token_count = sum(
            request.metrics.output_token_count for request in result.requests
        )
        return output_token_count / result.wall_time_seconds

    @staticmethod
    def _required_ttft(metrics: GenerationMetrics) -> int:
        assert metrics.time_to_first_token_microseconds is not None
        return metrics.time_to_first_token_microseconds

    @staticmethod
    def _required_e2e(metrics: GenerationMetrics) -> int:
        assert metrics.end_to_end_latency_microseconds is not None
        return metrics.end_to_end_latency_microseconds

    @staticmethod
    def _percentage_change(value: float, baseline: float) -> str:
        return f"{(value / baseline - 1.0) * 100:+.1f}%"
