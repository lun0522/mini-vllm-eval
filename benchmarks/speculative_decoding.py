"""Compare speculative-decoding draft-token count policies."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import DraftTokenMetrics
from benchmarks.base import GenerationMetrics
from benchmarks.base import LLAMA_LARGE_MODEL
from benchmarks.base import LLAMA_SMALL_MODEL
from benchmarks.base import QWEN_LARGE_MODEL
from benchmarks.base import QWEN_SMALL_MODEL
from benchmarks.example_prompts import EXAMPLE_SHORT_PROMPT_1
from benchmarks.speculative_prompts import CODE_REFACTORING_PROMPT
from benchmarks.speculative_prompts import CREATIVE_WRITING_PROMPT
from proto_loader import ProtoModules


TARGET_ONLY = "Target only"
DRAFT_ONLY = "Draft model only"
FIXED = "Fixed 4"
ADAPTIVE_MAX_DRAFT_TOKEN_COUNT = 8
MAX_INPUT_TOKEN_COUNT_DIFFERENCE_FRACTION = 0.15
ACCEPTANCE_RATE = f"Acceptance rate 1-{ADAPTIVE_MAX_DRAFT_TOKEN_COUNT}"
ACCEPTED_LENGTH = f"Accepted length 1-{ADAPTIVE_MAX_DRAFT_TOKEN_COUNT}"
CASE_ORDER = (TARGET_ONLY, FIXED, ACCEPTANCE_RATE, ACCEPTED_LENGTH)
ALL_CASE_ORDER = (*CASE_ORDER, DRAFT_ONLY)
MAX_NEW_TOKENS = 512
TARGET_KV_CACHE_SIZE_BYTES = 1024 * 1024 * 1024
WORKLOADS = (
    ("Code refactoring", CODE_REFACTORING_PROMPT),
    ("Creative writing", CREATIVE_WRITING_PROMPT),
)
GPU_ENVIRONMENT = (
    ("MINI_VLLM_METAL_GEMV_MAX_ROWS", "4"),
)
MODEL_FAMILIES = {
    "qwen": (QWEN_LARGE_MODEL, QWEN_SMALL_MODEL),
    "llama": (LLAMA_LARGE_MODEL, LLAMA_SMALL_MODEL),
}


@dataclass(frozen=True)
class SpeculativeRequestResult:
    workload: str
    output: str
    metrics: GenerationMetrics


@dataclass(frozen=True)
class SpeculativeCaseResult:
    configuration: str
    requests: tuple[SpeculativeRequestResult, ...]


class SpeculativeDecodingBenchmark(Benchmark):
    def __init__(self, model_family: str) -> None:
        try:
            self.target_model, self.draft_model = MODEL_FAMILIES[model_family]
        except KeyError as error:
            supported = ", ".join(MODEL_FAMILIES)
            raise ValueError(
                f"unsupported model family {model_family!r}; expected one of: {supported}"
            ) from error
        self.model_family = model_family

    def server_flags(self) -> list[str]:
        return self._server_flags(self.target_model)

    @staticmethod
    def _server_flags(model: str) -> list[str]:
        return [
            "--model",
            model,
            "--kv-cache-type",
            "contiguous",
            "--inference-device",
            "gpu",
            "--activation-dtype",
            "f32",
            "--target-kv-cache-size-bytes",
            str(TARGET_KV_CACHE_SIZE_BYTES),
            "--max-batched-token-count",
            "1024",
        ]

    def cases(self) -> tuple[BenchmarkCase, ...]:
        common_flags = tuple(self.server_flags())
        return (
            BenchmarkCase(TARGET_ONLY, common_flags, GPU_ENVIRONMENT),
            BenchmarkCase(
                FIXED,
                (
                    *common_flags,
                    "--draft-model",
                    self._draft_model("fixed { draft_token_count: 4 }"),
                ),
                GPU_ENVIRONMENT,
            ),
            BenchmarkCase(
                ACCEPTANCE_RATE,
                (
                    *common_flags,
                    "--draft-model",
                    self._draft_model(
                        "acceptance_rate { initial_draft_token_count: 4 "
                        "decrease_threshold: 0.4 increase_threshold: 0.8 "
                        "minimum_draft_token_count: 1 maximum_draft_token_count: "
                        f"{ADAPTIVE_MAX_DRAFT_TOKEN_COUNT} }}"
                    ),
                ),
                GPU_ENVIRONMENT,
            ),
            BenchmarkCase(
                ACCEPTED_LENGTH,
                (
                    *common_flags,
                    "--draft-model",
                    self._draft_model(
                        "accepted_length { initial_draft_token_count: 4 "
                        "smoothing_factor: 0.2 minimum_draft_token_count: 1 "
                        "maximum_draft_token_count: "
                        f"{ADAPTIVE_MAX_DRAFT_TOKEN_COUNT} }}"
                    ),
                ),
                GPU_ENVIRONMENT,
            ),
            BenchmarkCase(
                DRAFT_ONLY,
                tuple(self._server_flags(self.draft_model)),
                GPU_ENVIRONMENT,
            ),
        )

    def _draft_model(self, policy: str) -> str:
        return f"model {{ {self.draft_model} }} token_count_policy {{ {policy} }}"

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> SpeculativeCaseResult:
        self._run_request(client, proto, "Warm-up", EXAMPLE_SHORT_PROMPT_1, 1)
        requests = tuple(
            self._run_request(
                client,
                proto,
                workload,
                prompt,
                MAX_NEW_TOKENS,
            )
            for workload, prompt in WORKLOADS
        )
        return SpeculativeCaseResult(case.name, requests)

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_case = self._validated_results(results)
        logger.info(
            "{} speculative-decoding performance:\n{}",
            self.model_family,
            self._performance_table(results_by_case),
        )
        logger.info(
            "{} standalone draft-model cost proxy:\n{}",
            self.model_family,
            self._draft_model_performance_table(results_by_case),
        )
        logger.info(
            "{} speculative-decoding policy behavior:\n{}",
            self.model_family,
            self._draft_stats_table(results_by_case),
        )

    def _run_request(
        self,
        client: Any,
        proto: ProtoModules,
        workload: str,
        prompt: str,
        max_new_tokens: int,
    ) -> SpeculativeRequestResult:
        request = proto.request_handler.GenerateText(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        logger.warning(
            "Sending {} request with max_new_tokens={}",
            workload,
            max_new_tokens,
        )
        output_parts: list[str] = []
        metrics = None
        for response in client.GenerateText(request):
            event = response.WhichOneof("event")
            if event == "text":
                output_parts.append(response.text)
            elif event == "stats":
                metrics = self.generation_metrics(response.stats)
        if metrics is None:
            raise RuntimeError(f"{workload} completed without statistics")
        return SpeculativeRequestResult(workload, "".join(output_parts), metrics)

    @classmethod
    def _validated_results(
        cls,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> dict[str, SpeculativeCaseResult]:
        results_by_case: dict[str, SpeculativeCaseResult] = {}
        for case, result in results:
            if not isinstance(result, SpeculativeCaseResult):
                raise RuntimeError(
                    f"case {case.name} did not return speculative-decoding results"
                )
            if result.configuration != case.name or case.name in results_by_case:
                raise RuntimeError(f"invalid or duplicate result for case {case.name}")
            results_by_case[case.name] = result
        if set(results_by_case) != set(ALL_CASE_ORDER):
            raise RuntimeError("speculative-decoding benchmark requires all five cases")

        expected_workloads = {name for name, _prompt in WORKLOADS}
        requests_by_case = {
            case_name: {request.workload: request for request in result.requests}
            for case_name, result in results_by_case.items()
        }
        for case_name, requests in requests_by_case.items():
            if set(requests) != expected_workloads:
                raise RuntimeError(f"case {case_name} has unexpected workloads")
            for request in requests.values():
                cls._validate_request(case_name, request)

        for workload in expected_workloads:
            input_counts = {
                requests[workload].metrics.input_token_count
                for requests in requests_by_case.values()
            }
            if len(input_counts) != 1:
                raise RuntimeError(f"input token counts differ for {workload}")
        workload_input_counts = [
            requests_by_case[TARGET_ONLY][workload].metrics.input_token_count
            for workload in expected_workloads
        ]
        if (
            max(workload_input_counts) - min(workload_input_counts)
        ) / max(workload_input_counts) > MAX_INPUT_TOKEN_COUNT_DIFFERENCE_FRACTION:
            raise RuntimeError("workload input token counts differ by more than 15%")
        return results_by_case

    @classmethod
    def _validate_request(
        cls,
        case_name: str,
        request: SpeculativeRequestResult,
    ) -> None:
        metrics = request.metrics
        if not request.output:
            raise RuntimeError(f"{case_name} {request.workload} produced no text")
        if metrics.output_token_count != MAX_NEW_TOKENS:
            raise RuntimeError(
                f"{case_name} {request.workload} produced "
                f"{metrics.output_token_count} tokens instead of {MAX_NEW_TOKENS}"
            )
        cls._required_latencies(metrics)
        draft = metrics.draft_token_metrics
        if case_name in (TARGET_ONLY, DRAFT_ONLY):
            if draft is not None:
                raise RuntimeError(
                    f"non-speculative case {case_name} returned draft-token statistics"
                )
            return
        if draft is None:
            raise RuntimeError(f"{case_name} did not return draft-token statistics")
        cls._validate_draft_metrics(case_name, draft)

    @staticmethod
    def _validate_draft_metrics(case_name: str, metrics: DraftTokenMetrics) -> None:
        if metrics.proposed_token_count <= 0:
            raise RuntimeError(f"{case_name} proposed no draft tokens")
        if not 0 <= metrics.accepted_token_count <= metrics.proposed_token_count:
            raise RuntimeError(f"{case_name} returned invalid draft-token totals")
        histogram = metrics.selected_token_count_histogram
        if not histogram or tuple(sorted(histogram)) != histogram:
            raise RuntimeError(f"{case_name} returned an invalid draft-token histogram")
        if len({count for count, _uses in histogram}) != len(histogram):
            raise RuntimeError(f"{case_name} returned duplicate histogram buckets")
        if any(uses <= 0 for _count, uses in histogram):
            raise RuntimeError(f"{case_name} returned an empty histogram bucket")
        counts = {count for count, _uses in histogram}
        if case_name == FIXED and counts != {4}:
            raise RuntimeError("fixed-4 policy selected a draft-token count other than 4")
        if case_name in (ACCEPTANCE_RATE, ACCEPTED_LENGTH) and any(
            count < 1 or count > ADAPTIVE_MAX_DRAFT_TOKEN_COUNT for count in counts
        ):
            raise RuntimeError(
                f"{case_name} policy selected a count outside "
                f"[1, {ADAPTIVE_MAX_DRAFT_TOKEN_COUNT}]"
            )

    @classmethod
    def _performance_table(
        cls,
        results_by_case: dict[str, SpeculativeCaseResult],
    ) -> str:
        requests_by_case = {
            case_name: {request.workload: request for request in result.requests}
            for case_name, result in results_by_case.items()
        }
        rows = []
        for workload, _prompt in WORKLOADS:
            rates = {
                case_name: cls._request_metrics(requests[workload])
                for case_name, requests in requests_by_case.items()
            }
            target_rate = rates[TARGET_ONLY][4]
            fixed_rate = rates[FIXED][4]
            for case_name in CASE_ORDER:
                input_tokens, output_tokens, ttft, e2e, decode_rate = rates[case_name]
                rows.append(
                    (
                        workload,
                        case_name,
                        str(input_tokens),
                        str(output_tokens),
                        f"{decode_rate:.2f}",
                        f"{decode_rate / target_rate:.2f}x",
                        f"{decode_rate / fixed_rate:.2f}x",
                        f"{e2e / 1_000_000:.3f}",
                        f"{ttft / 1_000:.3f}",
                    )
                )
        return tabulate(
            rows,
            headers=(
                "Workload",
                "Configuration",
                "Input",
                "Output",
                "Decode (tok/s)",
                "vs target",
                "vs fixed",
                "E2E (s)",
                "TTFT (ms)",
            ),
            tablefmt="simple",
        )

    @classmethod
    def _request_metrics(
        cls,
        request: SpeculativeRequestResult,
    ) -> tuple[int, int, int, int, float]:
        metrics = request.metrics
        ttft, e2e = cls._required_latencies(metrics)
        decode_microseconds = e2e - ttft
        decode_tokens = metrics.output_token_count - 1
        return (
            metrics.input_token_count,
            metrics.output_token_count,
            ttft,
            e2e,
            decode_tokens * 1_000_000 / decode_microseconds,
        )

    @classmethod
    def _draft_model_performance_table(
        cls,
        results_by_case: dict[str, SpeculativeCaseResult],
    ) -> str:
        requests_by_case = {
            case_name: {request.workload: request for request in result.requests}
            for case_name, result in results_by_case.items()
        }
        rows = []
        for workload, _prompt in WORKLOADS:
            target = cls._request_metrics(requests_by_case[TARGET_ONLY][workload])
            draft = cls._request_metrics(requests_by_case[DRAFT_ONLY][workload])
            input_tokens, output_tokens, ttft, e2e, decode_rate = draft
            rows.append(
                (
                    workload,
                    str(input_tokens),
                    str(output_tokens),
                    f"{decode_rate:.2f}",
                    f"{decode_rate / target[4]:.2f}x",
                    f"{e2e / 1_000_000:.3f}",
                    f"{ttft / 1_000:.3f}",
                )
            )
        return tabulate(
            rows,
            headers=(
                "Workload",
                "Input",
                "Output",
                "Decode (tok/s)",
                "vs target",
                "E2E (s)",
                "TTFT (ms)",
            ),
            tablefmt="simple",
        )

    @classmethod
    def _draft_stats_table(
        cls,
        results_by_case: dict[str, SpeculativeCaseResult],
    ) -> str:
        rows = []
        for case_name in (FIXED, ACCEPTANCE_RATE, ACCEPTED_LENGTH):
            requests = results_by_case[case_name].requests
            for request in requests:
                draft = request.metrics.draft_token_metrics
                if draft is None:
                    raise RuntimeError("validated speculative result lost draft metrics")
                rows.append(
                    (
                        request.workload,
                        case_name,
                        f"{draft.accepted_token_count / draft.proposed_token_count * 100:.1f}%",
                        f"{draft.accepted_token_count}/{draft.proposed_token_count}",
                        ", ".join(
                            f"{token_count}:{usage_count}"
                            for token_count, usage_count in draft.selected_token_count_histogram
                        ),
                    )
                )
        return tabulate(
            rows,
            headers=(
                "Workload",
                "Configuration",
                "Acceptance",
                "Accepted/proposed",
                "Selected count:uses",
            ),
            tablefmt="simple",
        )

    @staticmethod
    def _required_latencies(metrics: GenerationMetrics) -> tuple[int, int]:
        ttft = metrics.time_to_first_token_microseconds
        e2e = metrics.end_to_end_latency_microseconds
        if ttft is None or e2e is None or ttft <= 0 or e2e <= ttft:
            raise RuntimeError(f"invalid generation latencies: TTFT={ttft}, E2E={e2e}")
        return ttft, e2e
