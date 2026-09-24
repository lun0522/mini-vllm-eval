"""Compare F32 and F16 CPU activation performance and memory use."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
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


WARMUP_MAX_NEW_TOKENS = 1
SMALL_PREFILL_MAX_NEW_TOKENS = 1
MEASURED_MAX_NEW_TOKENS = 1024
F32_KV_CACHE_SIZE_BYTES = 128 * 1024 * 1024
F16_KV_CACHE_SIZE_BYTES = F32_KV_CACHE_SIZE_BYTES // 2
ATTENTION_ENVIRONMENT = (
    ("MINI_VLLM_ENABLE_CPU_GROUPED_QUERY_MATMUL", "true"),
    ("MINI_VLLM_ENABLE_CPU_PAGED_ATTENTION", "false"),
    ("MINI_VLLM_ENABLE_CPU_PAGEWISE_VALUE_MATMUL", "false"),
)
F16_QMATMUL_ENVIRONMENT_VARIABLE = "MINI_VLLM_ENABLE_CPU_F16_QMATMUL_VIA_F32"
CONFIGURATIONS = (
    "F32",
    "F16",
    "F16-QMatMul",
)


@dataclass(frozen=True)
class CpuActivationDtypeCaseResult:
    configuration: str
    warmup_metrics: GenerationMetrics
    small_prefill_metrics: GenerationMetrics
    long_request_metrics: GenerationMetrics
    rss_metrics: RssMetrics


class CpuActivationDtypeBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return [
            "--model",
            QWEN_SMALL_MODEL,
            "--kv-cache-type",
            "paged:16",
            "--inference-device",
            "cpu",
            "--max-batched-token-count",
            "1024",
        ]

    def cases(self) -> tuple[BenchmarkCase, ...]:
        server_flags = tuple(self.server_flags())
        return tuple(
            self._case(
                name,
                server_flags,
                activation_dtype,
                qmatmul_via_f32,
            )
            for (name, activation_dtype, qmatmul_via_f32) in (
                ("F32", "f32", False),
                ("F16", "f16", False),
                ("F16-QMatMul", "f16", True),
            )
        )

    @staticmethod
    def _case(
        name: str,
        server_flags: tuple[str, ...],
        activation_dtype: str,
        qmatmul_via_f32: bool,
    ) -> BenchmarkCase:
        kv_cache_size_bytes = (
            F32_KV_CACHE_SIZE_BYTES
            if activation_dtype == "f32"
            else F16_KV_CACHE_SIZE_BYTES
        )
        return BenchmarkCase(
            name,
            (
                *server_flags,
                "--activation-dtype",
                activation_dtype,
                "--target-kv-cache-size-bytes",
                str(kv_cache_size_bytes),
            ),
            (
                *ATTENTION_ENVIRONMENT,
                (F16_QMATMUL_ENVIRONMENT_VARIABLE, str(qmatmul_via_f32).lower()),
            ),
        )

    def build_server_command(
        self,
        case: BenchmarkCase,
        trace_directory: Path | None = None,
    ) -> list[str]:
        case_trace_directory = (
            trace_directory / case.name.lower()
            if trace_directory is not None
            else None
        )
        return super().build_server_command(case, case_trace_directory)

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        process: subprocess.Popen[bytes],
    ) -> CpuActivationDtypeCaseResult:
        warmup_metrics = self._run_request(
            client,
            proto,
            EXAMPLE_SHORT_PROMPT_1,
            WARMUP_MAX_NEW_TOKENS,
        )
        small_prefill_metrics = self._run_request(
            client,
            proto,
            EXAMPLE_SHORT_PROMPT_2,
            SMALL_PREFILL_MAX_NEW_TOKENS,
        )
        sampler = ProcessTreeRssSampler(process.pid)
        sampler.start()
        try:
            long_request_metrics = self._run_request(
                client,
                proto,
                EXAMPLE_LONG_PROMPT_1,
                MEASURED_MAX_NEW_TOKENS,
            )
        finally:
            rss_metrics = sampler.stop()
        return CpuActivationDtypeCaseResult(
            configuration=case.name,
            warmup_metrics=warmup_metrics,
            small_prefill_metrics=small_prefill_metrics,
            long_request_metrics=long_request_metrics,
            rss_metrics=rss_metrics,
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_configuration: dict[str, CpuActivationDtypeCaseResult] = {}
        for case, result in results:
            if not isinstance(result, CpuActivationDtypeCaseResult):
                raise RuntimeError(
                    f"case {case.name} did not return CPU activation-dtype results"
                )
            if result.configuration in results_by_configuration:
                raise RuntimeError(
                    f"received duplicate {result.configuration} benchmark results"
                )
            results_by_configuration[result.configuration] = result

        if set(results_by_configuration) != set(CONFIGURATIONS):
            raise RuntimeError(
                "CPU activation-dtype benchmark requires all three configurations"
            )

        for result in results_by_configuration.values():
            self._validate_result(result)
        f32 = results_by_configuration["F32"]
        f32_small_ttft, _ = self._required_latencies(f32.small_prefill_metrics)
        f32_ttft, f32_e2e = self._required_latencies(f32.long_request_metrics)
        f32_decode_rate = self._decode_tokens_per_second(
            f32.long_request_metrics.output_token_count,
            f32_ttft,
            f32_e2e,
        )

        rows = []
        for configuration in CONFIGURATIONS:
            result = results_by_configuration[configuration]
            metrics = result.long_request_metrics
            small_ttft, _ = self._required_latencies(result.small_prefill_metrics)
            ttft, e2e = self._required_latencies(metrics)
            decode_rate = self._decode_tokens_per_second(
                metrics.output_token_count,
                ttft,
                e2e,
            )
            rows.append(
                (
                    configuration,
                    f"{small_ttft / 1_000:.3f}",
                    f"{small_ttft / f32_small_ttft:.2f}x",
                    f"{ttft / 1_000:.3f}",
                    f"{e2e / 1_000_000:.3f}",
                    f"{f32_e2e / e2e:.2f}x",
                    f"{decode_rate:.2f}",
                    f"{decode_rate / f32_decode_rate:.2f}x",
                    self._format_kib_as_mib(result.rss_metrics.starting_rss_kib),
                    self._format_kib_as_mib(result.rss_metrics.peak_rss_kib),
                    self._format_kib_as_mib(result.rss_metrics.peak_increase_kib),
                )
            )

        logger.info(
            "CPU activation dtype comparison with a {}-token small prefill and a "
            "{}-token input / {}-token output long request:\n{}",
            f32.small_prefill_metrics.input_token_count,
            f32.long_request_metrics.input_token_count,
            MEASURED_MAX_NEW_TOKENS,
            tabulate(
                rows,
                headers=(
                    "Configuration",
                    "Small TTFT (ms)",
                    "Small vs F32",
                    "Large TTFT (ms)",
                    "E2E (s)",
                    "E2E vs F32",
                    "Decode (tok/s)",
                    "Decode vs F32",
                    "Start RSS (MiB)",
                    "Peak RSS (MiB)",
                    "Peak increase (MiB)",
                ),
                tablefmt="simple",
            ),
        )

    def _run_request(
        self,
        client: Any,
        proto: ProtoModules,
        prompt: str,
        max_new_tokens: int,
    ) -> GenerationMetrics:
        request = proto.request_handler.GenerateText(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        logger.warning("Sending request with max_new_tokens={}", max_new_tokens)
        metrics = None
        for response in client.GenerateText(request):
            if response.WhichOneof("event") == "stats":
                metrics = self.generation_metrics(response.stats)
        if metrics is None:
            raise RuntimeError(
                f"{max_new_tokens}-token request completed without statistics"
            )
        return metrics

    @staticmethod
    def _validate_result(result: CpuActivationDtypeCaseResult) -> None:
        if result.warmup_metrics.output_token_count != WARMUP_MAX_NEW_TOKENS:
            raise RuntimeError(
                f"{result.configuration} warm-up produced "
                f"{result.warmup_metrics.output_token_count} tokens"
            )
        if (
            result.small_prefill_metrics.output_token_count
            != SMALL_PREFILL_MAX_NEW_TOKENS
        ):
            raise RuntimeError(
                f"{result.configuration} small-prefill request produced "
                f"{result.small_prefill_metrics.output_token_count} tokens"
            )
        if result.long_request_metrics.output_token_count != MEASURED_MAX_NEW_TOKENS:
            raise RuntimeError(
                f"{result.configuration} long request produced "
                f"{result.long_request_metrics.output_token_count} tokens"
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
    def _decode_tokens_per_second(
        output_token_count: int,
        ttft_microseconds: int,
        e2e_microseconds: int,
    ) -> float:
        decode_duration_microseconds = e2e_microseconds - ttft_microseconds
        if output_token_count <= 1 or decode_duration_microseconds <= 0:
            raise RuntimeError("generation result has no measurable decode duration")
        return (output_token_count - 1) * 1_000_000 / decode_duration_microseconds

    @staticmethod
    def _format_kib_as_mib(value: int) -> str:
        return f"{value / 1024:.2f}"
