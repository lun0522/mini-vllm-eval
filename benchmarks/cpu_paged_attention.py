"""Compare CPU generation with and without paged attention."""

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
MAX_NEW_TOKEN_COUNTS = (1, 512, 2048)


@dataclass(frozen=True)
class CpuPagedAttentionRequestResult:
    max_new_tokens: int
    metrics: GenerationMetrics


@dataclass(frozen=True)
class CpuPagedAttentionCaseResult:
    paged_attention_enabled: bool
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
        return (
            BenchmarkCase(
                "Paged attention disabled",
                server_flags,
                ((PAGED_ATTENTION_ENVIRONMENT_VARIABLE, "false"),),
            ),
            BenchmarkCase(
                "Paged attention enabled",
                server_flags,
                ((PAGED_ATTENTION_ENVIRONMENT_VARIABLE, "true"),),
            ),
        )

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
    ) -> CpuPagedAttentionCaseResult:
        paged_attention_enabled = self._paged_attention_enabled(case)
        requests = tuple(
            self._run_request(client, proto, max_new_tokens)
            for max_new_tokens in MAX_NEW_TOKEN_COUNTS
        )
        return CpuPagedAttentionCaseResult(
            paged_attention_enabled=paged_attention_enabled,
            requests=requests,
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_mode: dict[bool, CpuPagedAttentionCaseResult] = {}
        for case, result in results:
            if not isinstance(result, CpuPagedAttentionCaseResult):
                raise RuntimeError(
                    f"case {case.name} did not return CPU paged-attention results"
                )
            if result.paged_attention_enabled in results_by_mode:
                raise RuntimeError(
                    "received duplicate CPU paged-attention benchmark results"
                )
            results_by_mode[result.paged_attention_enabled] = result

        try:
            disabled = results_by_mode[False]
            enabled = results_by_mode[True]
        except KeyError as error:
            raise RuntimeError(
                "CPU paged-attention benchmark requires disabled and enabled results"
            ) from error

        logger.info(
            "CPU paged-attention comparison:\n{}",
            self._format_comparison(disabled, enabled),
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
    def _paged_attention_enabled(case: BenchmarkCase) -> bool:
        environment = dict(case.environment)
        value = environment.get(PAGED_ATTENTION_ENVIRONMENT_VARIABLE)
        if value not in ("false", "true"):
            raise RuntimeError(
                f"case {case.name} has invalid {PAGED_ATTENTION_ENVIRONMENT_VARIABLE}={value}"
            )
        return value == "true"

    @classmethod
    def _format_comparison(
        cls,
        disabled: CpuPagedAttentionCaseResult,
        enabled: CpuPagedAttentionCaseResult,
    ) -> str:
        disabled_requests = {
            request.max_new_tokens: request for request in disabled.requests
        }
        enabled_requests = {
            request.max_new_tokens: request for request in enabled.requests
        }
        if set(disabled_requests) != set(MAX_NEW_TOKEN_COUNTS):
            raise RuntimeError("disabled results do not contain the expected token limits")
        if set(enabled_requests) != set(MAX_NEW_TOKEN_COUNTS):
            raise RuntimeError("enabled results do not contain the expected token limits")

        rows = []
        for max_new_tokens in MAX_NEW_TOKEN_COUNTS:
            disabled_metrics = disabled_requests[max_new_tokens].metrics
            enabled_metrics = enabled_requests[max_new_tokens].metrics
            cls._validate_comparable_metrics(
                max_new_tokens,
                disabled_metrics,
                enabled_metrics,
            )
            disabled_ttft, disabled_e2e = cls._required_latencies(disabled_metrics)
            enabled_ttft, enabled_e2e = cls._required_latencies(enabled_metrics)
            disabled_total_rate = cls._tokens_per_second(
                disabled_metrics.output_token_count,
                disabled_e2e,
            )
            enabled_total_rate = cls._tokens_per_second(
                enabled_metrics.output_token_count,
                enabled_e2e,
            )
            disabled_decode_rate = cls._decode_tokens_per_second(
                disabled_metrics.output_token_count,
                disabled_ttft,
                disabled_e2e,
            )
            enabled_decode_rate = cls._decode_tokens_per_second(
                enabled_metrics.output_token_count,
                enabled_ttft,
                enabled_e2e,
            )
            rows.append(
                (
                    str(max_new_tokens),
                    str(disabled_metrics.output_token_count),
                    f"{disabled_ttft / 1_000:.3f}",
                    f"{enabled_ttft / 1_000:.3f}",
                    f"{disabled_e2e / 1_000_000:.3f}",
                    f"{enabled_e2e / 1_000_000:.3f}",
                    cls._format_speedup(disabled_e2e, enabled_e2e),
                    f"{disabled_total_rate:.2f}",
                    f"{enabled_total_rate:.2f}",
                    cls._format_optional_rate(disabled_decode_rate),
                    cls._format_optional_rate(enabled_decode_rate),
                    cls._format_optional_speedup(
                        disabled_decode_rate,
                        enabled_decode_rate,
                    ),
                )
            )

        return tabulate(
            rows,
            headers=(
                "Max new",
                "Output",
                "TTFT off (ms)",
                "TTFT on (ms)",
                "E2E off (s)",
                "E2E on (s)",
                "E2E speedup",
                "Total off (tok/s)",
                "Total on (tok/s)",
                "Decode off (tok/s)",
                "Decode on (tok/s)",
                "Decode speedup",
            ),
            tablefmt="simple",
        )

    @staticmethod
    def _validate_comparable_metrics(
        max_new_tokens: int,
        disabled: GenerationMetrics,
        enabled: GenerationMetrics,
    ) -> None:
        if disabled.input_token_count != enabled.input_token_count:
            raise RuntimeError(
                f"{max_new_tokens}-token runs have different input token counts"
            )
        if disabled.output_token_count != enabled.output_token_count:
            raise RuntimeError(
                f"{max_new_tokens}-token runs have different output token counts"
            )
        if disabled.output_token_count != max_new_tokens:
            raise RuntimeError(
                f"{max_new_tokens}-token runs produced {disabled.output_token_count} tokens"
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
    def _format_speedup(disabled_duration: int, enabled_duration: int) -> str:
        return f"{disabled_duration / enabled_duration:.2f}x"

    @staticmethod
    def _format_optional_rate(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    @staticmethod
    def _format_optional_speedup(
        disabled_rate: float | None,
        enabled_rate: float | None,
    ) -> str:
        if disabled_rate is None or enabled_rate is None:
            return "-"
        return f"{enabled_rate / disabled_rate:.2f}x"
