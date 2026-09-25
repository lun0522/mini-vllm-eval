"""Compare CPU attention implementations and query-head layouts."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_SMALL_MODEL
from benchmarks.example_prompts import EXAMPLE_LONG_PROMPT_1
from benchmarks.example_prompts import EXAMPLE_SHORT_PROMPT_1
from benchmarks.example_prompts import EXAMPLE_SHORT_PROMPT_2
from process_metrics import ProcessTreeRssSampler
from process_metrics import RssMetrics
from proto_loader import ProtoModules


PAGED_ATTENTION_ENVIRONMENT_VARIABLE = "MINI_VLLM_ENABLE_CPU_PAGED_ATTENTION"
CPU_GROUPED_QUERY_MATMUL_ENVIRONMENT_VARIABLE = (
    "MINI_VLLM_ENABLE_CPU_GROUPED_QUERY_MATMUL"
)
CPU_PAGEWISE_VALUE_MATMUL_ENVIRONMENT_VARIABLE = (
    "MINI_VLLM_ENABLE_CPU_PAGEWISE_VALUE_MATMUL"
)
CPU_F16_QMATMUL_ENVIRONMENT_VARIABLE = (
    "MINI_VLLM_ENABLE_CPU_F16_QMATMUL_VIA_F32"
)

WARMUP_REQUEST = ("Warm-Up", EXAMPLE_SHORT_PROMPT_1, 1)
SMALL_PREFILL_REQUEST = ("Small-Prefill", EXAMPLE_SHORT_PROMPT_2, 1)
LONG_REQUEST = ("Long-Request", EXAMPLE_LONG_PROMPT_1, 1024)
MEASURED_REQUESTS = (SMALL_PREFILL_REQUEST, LONG_REQUEST)


@dataclass(frozen=True)
class CpuPagedAttentionRequestResult:
    name: str
    max_new_tokens: int
    metrics: GenerationMetrics
    rss_metrics: RssMetrics | None


@dataclass(frozen=True)
class CpuPagedAttentionCaseResult:
    paged_attention_enabled: bool
    grouped_query_matmul_enabled: bool
    pagewise_value_matmul_enabled: bool
    requests: tuple[CpuPagedAttentionRequestResult, ...]


class CpuPagedAttentionBenchmark(Benchmark):
    def __init__(self, activation_dtype: str) -> None:
        if activation_dtype not in ("f32", "f16"):
            raise ValueError(f"unsupported activation dtype: {activation_dtype}")
        self.activation_dtype = activation_dtype

    def server_flags(self) -> list[str]:
        flags = [
            "--model",
            QWEN_SMALL_MODEL,
            "--kv-cache-type",
            "paged:16",
            "--inference-device",
            "cpu",
            "--max-batched-token-count",
            "1024",
        ]
        flags.extend(("--activation-dtype", self.activation_dtype))
        return flags

    def cases(self) -> tuple[BenchmarkCase, ...]:
        server_flags = tuple(self.server_flags())
        f16_environment = (
            ((CPU_F16_QMATMUL_ENVIRONMENT_VARIABLE, "true"),)
            if self.activation_dtype == "f16"
            else ()
        )
        return tuple(
            BenchmarkCase(
                f"{attention_name}, {query_matmul_name}, {value_matmul_name}",
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
                    (
                        CPU_PAGEWISE_VALUE_MATMUL_ENVIRONMENT_VARIABLE,
                        str(pagewise_value_matmul_enabled).lower(),
                    ),
                )
                + f16_environment,
            )
            for (
                paged_attention_enabled,
                grouped_query_matmul_enabled,
                pagewise_value_matmul_enabled,
                attention_name,
                query_matmul_name,
                value_matmul_name,
            ) in (
                (False, False, False, "Contiguous attention", "repeated KV", "full V"),
                (False, True, False, "Contiguous attention", "grouped Q", "full V"),
                (True, False, False, "Paged attention", "repeated KV", "concatenated V"),
                (True, False, True, "Paged attention", "repeated KV", "page-wise V"),
                (True, True, False, "Paged attention", "grouped Q", "concatenated V"),
                (True, True, True, "Paged attention", "grouped Q", "page-wise V"),
            )
        )

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        process: subprocess.Popen[bytes],
    ) -> CpuPagedAttentionCaseResult:
        (
            paged_attention_enabled,
            grouped_query_matmul_enabled,
            pagewise_value_matmul_enabled,
        ) = self._case_mode(case)
        warmup_result = self._run_request(
            client,
            proto,
            *WARMUP_REQUEST,
        )
        small_prefill_result = self._run_request(
            client,
            proto,
            *SMALL_PREFILL_REQUEST,
        )
        sampler = ProcessTreeRssSampler(process.pid)
        sampler.start()
        try:
            long_request_result = self._run_request(
                client,
                proto,
                *LONG_REQUEST,
            )
        finally:
            rss_metrics = sampler.stop()
        long_request_result = replace(long_request_result, rss_metrics=rss_metrics)
        return CpuPagedAttentionCaseResult(
            paged_attention_enabled=paged_attention_enabled,
            grouped_query_matmul_enabled=grouped_query_matmul_enabled,
            pagewise_value_matmul_enabled=pagewise_value_matmul_enabled,
            requests=(warmup_result, small_prefill_result, long_request_result),
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_mode: dict[tuple[bool, bool, bool], CpuPagedAttentionCaseResult] = {}
        for case, result in results:
            if not isinstance(result, CpuPagedAttentionCaseResult):
                raise RuntimeError(
                    f"case {case.name} did not return CPU paged-attention results"
                )
            mode = (
                result.paged_attention_enabled,
                result.grouped_query_matmul_enabled,
                result.pagewise_value_matmul_enabled,
            )
            if mode in results_by_mode:
                raise RuntimeError(
                    "received duplicate CPU paged-attention benchmark results"
                )
            results_by_mode[mode] = result

        expected_modes = {
            (False, False, False),
            (False, True, False),
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (True, True, True),
        }
        if set(results_by_mode) != expected_modes:
            raise RuntimeError(
                "CPU paged-attention benchmark requires all six attention/layout results"
            )

        for request_name, table in self._format_comparisons(results_by_mode):
            logger.info(
                "CPU attention comparison for {}:\n{}",
                request_name,
                table,
            )
        logger.info(
            "Peak RSS comparison during the large request:\n{}",
            self._format_rss_comparison(results_by_mode),
        )

    def _run_request(
        self,
        client: Any,
        proto: ProtoModules,
        name: str,
        prompt: str,
        max_new_tokens: int,
    ) -> CpuPagedAttentionRequestResult:
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
            name,
            max_new_tokens,
        )
        metrics = None
        for response in client.GenerateText(request):
            if response.WhichOneof("event") == "stats":
                metrics = self.generation_metrics(response.stats)
        if metrics is None:
            raise RuntimeError(
                f"{max_new_tokens}-token request completed without statistics"
            )
        return CpuPagedAttentionRequestResult(
            name=name,
            max_new_tokens=max_new_tokens,
            metrics=metrics,
            rss_metrics=None,
        )

    @staticmethod
    def _case_mode(case: BenchmarkCase) -> tuple[bool, bool, bool]:
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
            CpuPagedAttentionBenchmark._environment_bool(
                case,
                environment,
                CPU_PAGEWISE_VALUE_MATMUL_ENVIRONMENT_VARIABLE,
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
    def _format_comparisons(
        cls,
        results_by_mode: dict[tuple[bool, bool, bool], CpuPagedAttentionCaseResult],
    ) -> tuple[tuple[str, str], ...]:
        requests_by_mode = {
            mode: {request.name: request for request in result.requests}
            for mode, result in results_by_mode.items()
        }
        expected_requests = {
            request_name
            for request_name, _prompt, _max_new_tokens in (
                WARMUP_REQUEST,
                *MEASURED_REQUESTS,
            )
        }
        for mode, requests in requests_by_mode.items():
            if set(requests) != expected_requests:
                raise RuntimeError(
                    f"results for mode {mode} do not contain the expected requests"
                )

        tables = []
        for request_name, _prompt, max_new_tokens in MEASURED_REQUESTS:
            rows = []
            metrics_by_mode = {
                mode: requests[request_name].metrics
                for mode, requests in requests_by_mode.items()
            }
            cls._validate_comparable_metrics(max_new_tokens, metrics_by_mode)
            baseline_metrics = metrics_by_mode[(False, False, False)]
            _, baseline_e2e = cls._required_latencies(baseline_metrics)
            baseline_decode_rate = cls._decode_tokens_per_second(
                baseline_metrics.output_token_count,
                *cls._required_latencies(baseline_metrics),
            )
            for mode in (
                (False, False, False),
                (False, True, False),
                (True, False, False),
                (True, False, True),
                (True, True, False),
                (True, True, True),
            ):
                (
                    paged_attention_enabled,
                    grouped_query_matmul_enabled,
                    pagewise_value_matmul_enabled,
                ) = mode
                metrics = metrics_by_mode[mode]
                ttft, e2e = cls._required_latencies(metrics)
                total_rate = cls._tokens_per_second(metrics.output_token_count, e2e)
                decode_rate = cls._decode_tokens_per_second(
                    metrics.output_token_count,
                    ttft,
                    e2e,
                )
                if not paged_attention_enabled:
                    value_matmul_name = "Full V"
                elif pagewise_value_matmul_enabled:
                    value_matmul_name = "Page-wise V"
                else:
                    value_matmul_name = "Concatenated V"
                rows.append(
                    (
                        "Paged" if paged_attention_enabled else "Contiguous",
                        "Grouped Q" if grouped_query_matmul_enabled else "Repeated KV",
                        value_matmul_name,
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

            tables.append(
                (
                    request_name,
                    tabulate(
                        rows,
                        headers=(
                            "Attention",
                            "Q/KV layout",
                            "V matmul",
                            "TTFT (ms)",
                            "E2E (s)",
                            "E2E vs baseline",
                            "Total (tok/s)",
                            "Decode (tok/s)",
                            "Decode vs baseline",
                        ),
                        tablefmt="simple",
                    ),
                )
            )
        return tuple(tables)

    @classmethod
    def _format_rss_comparison(
        cls,
        results_by_mode: dict[tuple[bool, bool, bool], CpuPagedAttentionCaseResult],
    ) -> str:
        rows = []
        for mode in (
            (False, False, False),
            (False, True, False),
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (True, True, True),
        ):
            result = results_by_mode[mode]
            requests = {request.name: request for request in result.requests}
            request = requests.get(LONG_REQUEST[0])
            if request is None:
                raise RuntimeError(
                    f"results for mode {mode} do not contain the large request"
                )
            if request.rss_metrics is None:
                raise RuntimeError(
                    f"large-request result for mode {mode} has no RSS metrics"
                )

            (
                paged_attention_enabled,
                grouped_query_matmul_enabled,
                pagewise_enabled,
            ) = mode
            if not paged_attention_enabled:
                value_matmul_name = "Full V"
            elif pagewise_enabled:
                value_matmul_name = "Page-wise V"
            else:
                value_matmul_name = "Concatenated V"
            rss_metrics = request.rss_metrics
            rows.append(
                (
                    "Paged" if paged_attention_enabled else "Contiguous",
                    "Grouped Q" if grouped_query_matmul_enabled else "Repeated KV",
                    value_matmul_name,
                    cls._format_kib_as_mib(rss_metrics.starting_rss_kib),
                    cls._format_kib_as_mib(rss_metrics.peak_rss_kib),
                    cls._format_kib_as_mib(rss_metrics.peak_increase_kib),
                )
            )

        return tabulate(
            rows,
            headers=(
                "Attention",
                "Q/KV layout",
                "V matmul",
                "Start RSS (MiB)",
                "Peak RSS (MiB)",
                "Peak increase (MiB)",
            ),
            tablefmt="simple",
        )

    @staticmethod
    def _format_kib_as_mib(value: int) -> str:
        return f"{value / 1024:.2f}"

    @staticmethod
    def _validate_comparable_metrics(
        max_new_tokens: int,
        metrics_by_mode: dict[tuple[bool, bool, bool], GenerationMetrics],
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
