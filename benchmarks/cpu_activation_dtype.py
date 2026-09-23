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
from benchmarks.example_prompts import EXAMPLE_SHORT_PROMPT_1
from process_metrics import ProcessTreeRssSampler
from process_metrics import RssMetrics
from proto_loader import ProtoModules


WARMUP_MAX_NEW_TOKENS = 1
MEASURED_MAX_NEW_TOKENS = 1024
F32_KV_CACHE_SIZE_BYTES = 128 * 1024 * 1024
F16_KV_CACHE_SIZE_BYTES = F32_KV_CACHE_SIZE_BYTES // 2
ATTENTION_ENVIRONMENT = (
    ("MINI_VLLM_ENABLE_CPU_GROUPED_QUERY_MATMUL", "true"),
    ("MINI_VLLM_ENABLE_CPU_PAGED_ATTENTION", "false"),
    ("MINI_VLLM_ENABLE_CPU_PAGEWISE_VALUE_MATMUL", "false"),
)


@dataclass(frozen=True)
class CpuActivationDtypeCaseResult:
    activation_dtype: str
    warmup_metrics: GenerationMetrics
    measured_metrics: GenerationMetrics
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
        ]

    def cases(self) -> tuple[BenchmarkCase, ...]:
        server_flags = tuple(self.server_flags())
        return (
            BenchmarkCase(
                "F32",
                (
                    *server_flags,
                    "--activation-dtype",
                    "f32",
                    "--target-kv-cache-size-bytes",
                    str(F32_KV_CACHE_SIZE_BYTES),
                ),
                ATTENTION_ENVIRONMENT,
            ),
            BenchmarkCase(
                "F16",
                (
                    *server_flags,
                    "--activation-dtype",
                    "f16",
                    "--target-kv-cache-size-bytes",
                    str(F16_KV_CACHE_SIZE_BYTES),
                ),
                ATTENTION_ENVIRONMENT,
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
            WARMUP_MAX_NEW_TOKENS,
        )
        sampler = ProcessTreeRssSampler(process.pid)
        sampler.start()
        try:
            measured_metrics = self._run_request(
                client,
                proto,
                MEASURED_MAX_NEW_TOKENS,
            )
        finally:
            rss_metrics = sampler.stop()
        return CpuActivationDtypeCaseResult(
            activation_dtype=case.name,
            warmup_metrics=warmup_metrics,
            measured_metrics=measured_metrics,
            rss_metrics=rss_metrics,
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_dtype: dict[str, CpuActivationDtypeCaseResult] = {}
        for case, result in results:
            if not isinstance(result, CpuActivationDtypeCaseResult):
                raise RuntimeError(
                    f"case {case.name} did not return CPU activation-dtype results"
                )
            if result.activation_dtype in results_by_dtype:
                raise RuntimeError(
                    f"received duplicate {result.activation_dtype} benchmark results"
                )
            results_by_dtype[result.activation_dtype] = result

        if set(results_by_dtype) != {"F32", "F16"}:
            raise RuntimeError("CPU activation-dtype benchmark requires F32 and F16 results")

        f32 = results_by_dtype["F32"]
        self._validate_result(f32)
        self._validate_result(results_by_dtype["F16"])
        f32_ttft, f32_e2e = self._required_latencies(f32.measured_metrics)
        f32_decode_rate = self._decode_tokens_per_second(
            f32.measured_metrics.output_token_count,
            f32_ttft,
            f32_e2e,
        )

        rows = []
        for activation_dtype in ("F32", "F16"):
            result = results_by_dtype[activation_dtype]
            metrics = result.measured_metrics
            ttft, e2e = self._required_latencies(metrics)
            decode_rate = self._decode_tokens_per_second(
                metrics.output_token_count,
                ttft,
                e2e,
            )
            rows.append(
                (
                    activation_dtype,
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
            "CPU activation dtype comparison with a {}-token measured request:\n{}",
            MEASURED_MAX_NEW_TOKENS,
            tabulate(
                rows,
                headers=(
                    "Activation",
                    "TTFT (ms)",
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
        max_new_tokens: int,
    ) -> GenerationMetrics:
        request = proto.request_handler.GenerateText(
            prompt=EXAMPLE_SHORT_PROMPT_1,
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
                f"{result.activation_dtype} warm-up produced "
                f"{result.warmup_metrics.output_token_count} tokens"
            )
        if result.measured_metrics.output_token_count != MEASURED_MAX_NEW_TOKENS:
            raise RuntimeError(
                f"{result.activation_dtype} measured request produced "
                f"{result.measured_metrics.output_token_count} tokens"
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
