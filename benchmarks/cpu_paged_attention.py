"""Compare CPU attention implementations and query-head layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_SMALL_MODEL
from benchmarks.example_prompts import EXAMPLE_SHORT_PROMPT_1
from proto_loader import ProtoModules


PAGED_ATTENTION_ENVIRONMENT_VARIABLE = "MINI_VLLM_ENABLE_CPU_PAGED_ATTENTION"
CPU_GROUPED_QUERY_MATMUL_ENVIRONMENT_VARIABLE = (
    "MINI_VLLM_ENABLE_CPU_GROUPED_QUERY_MATMUL"
)
MAX_NEW_TOKEN_COUNTS = (1, 512, 2048)


@dataclass(frozen=True)
class CpuPagedAttentionRequestResult:
    max_new_tokens: int
    metrics: GenerationMetrics


@dataclass(frozen=True)
class CpuPagedAttentionCaseResult:
    paged_attention_enabled: bool
    grouped_query_matmul_enabled: bool
    requests: tuple[CpuPagedAttentionRequestResult, ...]


class CpuPagedAttentionBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return [
            "--model",
            QWEN_SMALL_MODEL,
            "--kv-cache-type",
            "paged:16",
            "--inference-device",
            "cpu",
        ]

    def cases(self) -> tuple[BenchmarkCase, ...]:
        server_flags = tuple(self.server_flags())
        return tuple(
            BenchmarkCase(
                f"{attention_name}, {matmul_name}",
                server_flags,
                (
                    (
                        PAGED_ATTENTION_ENVIRONMENT_VARIABLE,
                        str(paged_attention_enabled).lower(),
                    ),
                    (
                        CPU_GROUPED_QUERY_MATMUL_ENVIRONMENT_VARIABLE,
                        str(grouped_query_matmul_enabled).lower(),
                    ),
                ),
            )
            for paged_attention_enabled, attention_name in (
                (False, "Contiguous attention"),
                (True, "Paged attention"),
            )
            for grouped_query_matmul_enabled, matmul_name in (
                (False, "repeated KV"),
                (True, "grouped Q"),
            )
        )

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
    ) -> CpuPagedAttentionCaseResult:
        paged_attention_enabled, grouped_query_matmul_enabled = self._case_mode(case)
        requests = tuple(
            self._run_request(client, proto, max_new_tokens)
            for max_new_tokens in MAX_NEW_TOKEN_COUNTS
        )
        return CpuPagedAttentionCaseResult(
            paged_attention_enabled=paged_attention_enabled,
            grouped_query_matmul_enabled=grouped_query_matmul_enabled,
            requests=requests,
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_mode: dict[tuple[bool, bool], CpuPagedAttentionCaseResult] = {}
        for case, result in results:
            if not isinstance(result, CpuPagedAttentionCaseResult):
                raise RuntimeError(
                    f"case {case.name} did not return CPU paged-attention results"
                )
            mode = (
                result.paged_attention_enabled,
                result.grouped_query_matmul_enabled,
            )
            if mode in results_by_mode:
                raise RuntimeError(
                    "received duplicate CPU paged-attention benchmark results"
                )
            results_by_mode[mode] = result

        expected_modes = {
            (paged_attention_enabled, grouped_query_matmul_enabled)
            for paged_attention_enabled in (False, True)
            for grouped_query_matmul_enabled in (False, True)
        }
        if set(results_by_mode) != expected_modes:
            raise RuntimeError(
                "CPU paged-attention benchmark requires all four attention/layout results"
            )

        logger.info(
            "CPU attention comparison:\n{}",
            self._format_comparison(results_by_mode),
        )

    def _run_request(
        self,
        client: Any,
        proto: ProtoModules,
        max_new_tokens: int,
    ) -> CpuPagedAttentionRequestResult:
        request = proto.request_handler.GenerateText(
            prompt=EXAMPLE_SHORT_PROMPT_1,
            max_new_tokens=max_new_tokens,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        logger.warning(
            "Sending request with max_new_tokens={}",
            max_new_tokens,
        )
        result = None
        for response in client.GenerateText(request):
            if response.WhichOneof("event") == "stats":
                result = CpuPagedAttentionRequestResult(
                    max_new_tokens=max_new_tokens,
                    metrics=self.generation_metrics(response.stats),
                )
        if result is None:
            raise RuntimeError(
                f"{max_new_tokens}-token request completed without statistics"
            )
        return result

    @staticmethod
    def _case_mode(case: BenchmarkCase) -> tuple[bool, bool]:
        environment = dict(case.environment)
        return (
            CpuPagedAttentionBenchmark._environment_bool(
                case,
                environment,
                PAGED_ATTENTION_ENVIRONMENT_VARIABLE,
            ),
            CpuPagedAttentionBenchmark._environment_bool(
                case,
                environment,
                CPU_GROUPED_QUERY_MATMUL_ENVIRONMENT_VARIABLE,
            ),
        )

    @staticmethod
    def _environment_bool(
        case: BenchmarkCase,
        environment: dict[str, str],
        name: str,
    ) -> bool:
        value = environment.get(name)
        if value not in ("false", "true"):
            raise RuntimeError(f"case {case.name} has invalid {name}={value}")
        return value == "true"

    @classmethod
    def _format_comparison(
        cls,
        results_by_mode: dict[tuple[bool, bool], CpuPagedAttentionCaseResult],
    ) -> str:
        requests_by_mode = {
            mode: {
                request.max_new_tokens: request for request in result.requests
            }
            for mode, result in results_by_mode.items()
        }
        for mode, requests in requests_by_mode.items():
            if set(requests) != set(MAX_NEW_TOKEN_COUNTS):
                raise RuntimeError(
                    f"results for mode {mode} do not contain the expected token limits"
                )

        rows = []
        for max_new_tokens in MAX_NEW_TOKEN_COUNTS:
            metrics_by_mode = {
                mode: requests[max_new_tokens].metrics
                for mode, requests in requests_by_mode.items()
            }
            cls._validate_comparable_metrics(max_new_tokens, metrics_by_mode)
            baseline_metrics = metrics_by_mode[(False, False)]
            _, baseline_e2e = cls._required_latencies(baseline_metrics)
            baseline_decode_rate = cls._decode_tokens_per_second(
                baseline_metrics.output_token_count,
                *cls._required_latencies(baseline_metrics),
            )
            for mode in ((False, False), (False, True), (True, False), (True, True)):
                paged_attention_enabled, grouped_query_matmul_enabled = mode
                metrics = metrics_by_mode[mode]
                ttft, e2e = cls._required_latencies(metrics)
                total_rate = cls._tokens_per_second(metrics.output_token_count, e2e)
                decode_rate = cls._decode_tokens_per_second(
                    metrics.output_token_count,
                    ttft,
                    e2e,
                )
                rows.append(
                    (
                        str(max_new_tokens),
                        str(metrics.output_token_count),
                        "Paged" if paged_attention_enabled else "Contiguous",
                        "Grouped Q" if grouped_query_matmul_enabled else "Repeated KV",
                        f"{ttft / 1_000:.3f}",
                        f"{e2e / 1_000_000:.3f}",
                        cls._format_speedup(baseline_e2e, e2e),
                        f"{total_rate:.2f}",
                        cls._format_optional_rate(decode_rate),
                        cls._format_optional_speedup(
                            baseline_decode_rate,
                            decode_rate,
                        ),
                    )
                )

        return tabulate(
            rows,
            headers=(
                "Max new",
                "Output",
                "Attention",
                "Q/KV layout",
                "TTFT (ms)",
                "E2E (s)",
                "E2E vs baseline",
                "Total (tok/s)",
                "Decode (tok/s)",
                "Decode vs baseline",
            ),
            tablefmt="simple",
        )

    @staticmethod
    def _validate_comparable_metrics(
        max_new_tokens: int,
        metrics_by_mode: dict[tuple[bool, bool], GenerationMetrics],
    ) -> None:
        input_token_counts = {
            metrics.input_token_count for metrics in metrics_by_mode.values()
        }
        if len(input_token_counts) != 1:
            raise RuntimeError(
                f"{max_new_tokens}-token runs have different input token counts: "
                f"{sorted(input_token_counts)}"
            )
        output_token_counts = {
            metrics.output_token_count for metrics in metrics_by_mode.values()
        }
        if len(output_token_counts) != 1:
            raise RuntimeError(
                f"{max_new_tokens}-token runs have different output token counts: "
                f"{sorted(output_token_counts)}"
            )
        output_token_count = next(iter(output_token_counts))
        if output_token_count != max_new_tokens:
            raise RuntimeError(
                f"{max_new_tokens}-token runs produced {output_token_count} tokens"
            )

    @staticmethod
    def _required_latencies(metrics: GenerationMetrics) -> tuple[int, int]:
        ttft = metrics.time_to_first_token_microseconds
        e2e = metrics.end_to_end_latency_microseconds
        if ttft is None or e2e is None:
            raise RuntimeError("generation result does not contain latency metrics")
        if ttft <= 0 or e2e <= 0 or ttft > e2e:
            raise RuntimeError(
                f"generation result has invalid latency metrics: TTFT={ttft}, E2E={e2e}"
            )
        return ttft, e2e

    @staticmethod
    def _tokens_per_second(token_count: int, duration_microseconds: int) -> float:
        return token_count * 1_000_000 / duration_microseconds

    @staticmethod
    def _decode_tokens_per_second(
        output_token_count: int,
        ttft_microseconds: int,
        e2e_microseconds: int,
    ) -> float | None:
        if output_token_count <= 1:
            return None
        decode_duration_microseconds = e2e_microseconds - ttft_microseconds
        if decode_duration_microseconds <= 0:
            raise RuntimeError("generation result has no measurable decode duration")
        return (output_token_count - 1) * 1_000_000 / decode_duration_microseconds

    @staticmethod
    def _format_speedup(baseline_duration: int, duration: int) -> str:
        return f"{baseline_duration / duration:.2f}x"

    @staticmethod
    def _format_optional_rate(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    @staticmethod
    def _format_optional_speedup(
        baseline_rate: float | None,
        rate: float | None,
    ) -> str:
        if baseline_rate is None or rate is None:
            return "-"
        return f"{rate / baseline_rate:.2f}x"
