"""Measure speculative sampling's speed-versus-fidelity tradeoff."""

from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import DraftTokenMetrics
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_0_5B_MODEL
from benchmarks.base import QWEN_7B_MODEL
from benchmarks.speculative_prompts import CREATIVE_WRITING_PROMPT
from proto_loader import ProtoModules


TARGET_ONLY = "Target only"
FIXED_4 = "Fixed 4"
CASE_ORDER = (TARGET_ONLY, FIXED_4)
TEMPERATURES = (0.2, 0.7, 1.0, 1.6)
TOTAL_VARIATION_BUDGETS = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
REPRESENTATIVE_TEMPERATURE = 0.7
SEEDS = (17, 101, 20261022)
EXCERPT_SEED = 101
MAX_NEW_TOKENS = 256
DRAFT_TOKEN_COUNT = 4
TARGET_KV_CACHE_SIZE_BYTES = 1024 * 1024 * 1024
GPU_ENVIRONMENT = (("MINI_VLLM_METAL_GEMV_MAX_ROWS", "4"),)


@dataclass(frozen=True)
class SamplingSetting:
    temperature: float
    total_variation_budget: float


@dataclass(frozen=True)
class SamplingRequestResult:
    setting: SamplingSetting
    seed: int
    output: str
    metrics: GenerationMetrics


@dataclass(frozen=True)
class SamplingCaseResult:
    configuration: str
    requests: tuple[SamplingRequestResult, ...]


@dataclass(frozen=True)
class SamplingSummary:
    median_e2e_seconds: float
    minimum_e2e_seconds: float
    maximum_e2e_seconds: float
    median_decode_tokens_per_second: float
    accepted_token_count: int = 0
    proposed_token_count: int = 0
    accepted_tokens_per_round: float | None = None
    output_tokens_per_round: float | None = None


class SpeculativeSamplingTradeoffBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return self._server_flags()

    @staticmethod
    def _server_flags() -> list[str]:
        return [
            "--model",
            QWEN_7B_MODEL,
            "--kv-cache-config",
            (
                "kv_cache_type: KV_CACHE_TYPE_CONTIGUOUS "
                f"target_kv_cache_size_bytes: {TARGET_KV_CACHE_SIZE_BYTES}"
            ),
            "--inference-device",
            "gpu",
            "--activation-dtype",
            "f32",
            "--scheduler-config",
            "max_batched_token_count: 1024",
        ]

    def cases(self) -> tuple[BenchmarkCase, ...]:
        common_flags = tuple(self.server_flags())
        draft_model = (
            f"model {{ {QWEN_0_5B_MODEL} }} token_count_policy {{ fixed {{ "
            f"draft_token_count: {DRAFT_TOKEN_COUNT} }} }}"
        )
        return (
            BenchmarkCase(TARGET_ONLY, common_flags, GPU_ENVIRONMENT),
            BenchmarkCase(
                FIXED_4,
                (*common_flags, "--draft-model", draft_model),
                GPU_ENVIRONMENT,
            ),
        )

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> SamplingCaseResult:
        self._run_request(
            client,
            proto,
            SamplingSetting(REPRESENTATIVE_TEMPERATURE, 0.0),
            SEEDS[0],
            1,
        )
        settings = (
            (SamplingSetting(REPRESENTATIVE_TEMPERATURE, 0.0),)
            if case.name == TARGET_ONLY
            else self._speculative_settings()
        )
        requests = tuple(
            self._run_request(client, proto, setting, seed, MAX_NEW_TOKENS)
            for seed in SEEDS
            for setting in settings
        )
        return SamplingCaseResult(case.name, requests)

    @staticmethod
    def _speculative_settings() -> tuple[SamplingSetting, ...]:
        temperature_settings = tuple(
            SamplingSetting(temperature, 0.0) for temperature in TEMPERATURES
        )
        budget_settings = tuple(
            SamplingSetting(REPRESENTATIVE_TEMPERATURE, budget)
            for budget in TOTAL_VARIATION_BUDGETS
            if budget != 0.0
        )
        return (*temperature_settings, *budget_settings)

    def _run_request(
        self,
        client: Any,
        proto: ProtoModules,
        setting: SamplingSetting,
        seed: int,
        max_new_tokens: int,
    ) -> SamplingRequestResult:
        request = proto.request_handler.GenerateText(
            prompt=CREATIVE_WRITING_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=setting.temperature,
            total_variation_budget=setting.total_variation_budget,
            seed=seed,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        logger.warning(
            "Sending creative-writing request with temperature={}, "
            "total_variation_budget={}, seed={}, and max_new_tokens={}",
            setting.temperature,
            setting.total_variation_budget,
            seed,
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
            raise RuntimeError("creative-writing request completed without statistics")
        return SamplingRequestResult(
            setting,
            seed,
            "".join(output_parts),
            metrics,
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_case = self._validated_results(results)
        target_requests = results_by_case[TARGET_ONLY].requests
        speculative_requests = results_by_case[FIXED_4].requests

        logger.info(
            "Temperature effect on lossless fixed-4 speculative decoding:\n{}",
            self._temperature_table(target_requests, speculative_requests),
        )
        logger.info(
            "Total-variation budget tradeoff at temperature {}:\n{}",
            REPRESENTATIVE_TEMPERATURE,
            self._budget_table(target_requests, speculative_requests),
        )
        logger.info(
            "Creative-writing excerpts for seed {} at temperature {}:\n{}",
            EXCERPT_SEED,
            REPRESENTATIVE_TEMPERATURE,
            self._excerpt_report(speculative_requests),
        )
        logger.info(
            "Machine-readable speculative-sampling results:\n{}",
            self._machine_readable_results(target_requests, speculative_requests),
        )

    @classmethod
    def _validated_results(
        cls,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> dict[str, SamplingCaseResult]:
        results_by_case: dict[str, SamplingCaseResult] = {}
        for case, result in results:
            if not isinstance(result, SamplingCaseResult):
                raise RuntimeError(
                    f"case {case.name} did not return sampling-tradeoff results"
                )
            if result.configuration != case.name or case.name in results_by_case:
                raise RuntimeError(f"invalid or duplicate result for case {case.name}")
            results_by_case[case.name] = result
        if tuple(results_by_case) != CASE_ORDER:
            raise RuntimeError(
                "speculative-sampling tradeoff benchmark requires target-only and "
                "fixed-4 cases"
            )

        target_requests = results_by_case[TARGET_ONLY].requests
        speculative_requests = results_by_case[FIXED_4].requests
        target_setting = SamplingSetting(REPRESENTATIVE_TEMPERATURE, 0.0)
        cls._validate_request_matrix(
            TARGET_ONLY,
            target_requests,
            (target_setting,),
            speculative=False,
        )
        cls._validate_request_matrix(
            FIXED_4,
            speculative_requests,
            cls._speculative_settings(),
            speculative=True,
        )

        input_token_counts = {
            request.metrics.input_token_count
            for request in (*target_requests, *speculative_requests)
        }
        if len(input_token_counts) != 1:
            raise RuntimeError("creative-writing input token counts differ across requests")
        return results_by_case

    @classmethod
    def _validate_request_matrix(
        cls,
        case_name: str,
        requests: tuple[SamplingRequestResult, ...],
        settings: tuple[SamplingSetting, ...],
        *,
        speculative: bool,
    ) -> None:
        expected = {(setting, seed) for setting in settings for seed in SEEDS}
        actual = {(request.setting, request.seed) for request in requests}
        if len(requests) != len(expected) or actual != expected:
            raise RuntimeError(f"case {case_name} returned an unexpected request matrix")

        for request in requests:
            if not request.output:
                raise RuntimeError(f"case {case_name} produced no text")
            if request.metrics.output_token_count != MAX_NEW_TOKENS:
                raise RuntimeError(
                    f"case {case_name} produced {request.metrics.output_token_count} "
                    f"tokens instead of {MAX_NEW_TOKENS}"
                )
            cls._required_latencies(request.metrics)
            draft = request.metrics.draft_token_metrics
            if not speculative:
                if draft is not None:
                    raise RuntimeError("target-only request returned draft-token statistics")
                continue
            if draft is None:
                raise RuntimeError("fixed-4 request omitted draft-token statistics")
            cls._validate_draft_metrics(draft)
            if (
                request.setting.total_variation_budget == 1.0
                and draft.accepted_token_count != draft.proposed_token_count
            ):
                raise RuntimeError(
                    "a total-variation budget of 1 did not accept every draft token"
                )

    @staticmethod
    def _validate_draft_metrics(metrics: DraftTokenMetrics) -> None:
        if metrics.proposed_token_count <= 0:
            raise RuntimeError("fixed-4 request proposed no draft tokens")
        if not 0 <= metrics.accepted_token_count <= metrics.proposed_token_count:
            raise RuntimeError("fixed-4 request returned invalid draft-token totals")
        histogram = metrics.selected_token_count_histogram
        if len(histogram) != 1 or histogram[0][0] != DRAFT_TOKEN_COUNT:
            raise RuntimeError("fixed-4 policy selected a draft-token count other than 4")
        if histogram[0][1] <= 0:
            raise RuntimeError("fixed-4 request returned an empty histogram bucket")

    @classmethod
    def _temperature_table(
        cls,
        target_requests: tuple[SamplingRequestResult, ...],
        speculative_requests: tuple[SamplingRequestResult, ...],
    ) -> str:
        target_summary = cls._summarize(target_requests)
        rows = [
            cls._summary_row(
                TARGET_ONLY,
                REPRESENTATIVE_TEMPERATURE,
                None,
                target_summary,
                None,
            )
        ]
        for temperature in TEMPERATURES:
            matching = cls._matching_requests(
                speculative_requests,
                SamplingSetting(temperature, 0.0),
            )
            summary = cls._summarize(matching)
            baseline = target_summary if temperature == REPRESENTATIVE_TEMPERATURE else None
            rows.append(
                cls._summary_row(FIXED_4, temperature, 0.0, summary, baseline)
            )
        return cls._summary_table(rows)

    @classmethod
    def _budget_table(
        cls,
        target_requests: tuple[SamplingRequestResult, ...],
        speculative_requests: tuple[SamplingRequestResult, ...],
    ) -> str:
        target_summary = cls._summarize(target_requests)
        rows = [
            cls._summary_row(
                TARGET_ONLY,
                REPRESENTATIVE_TEMPERATURE,
                None,
                target_summary,
                target_summary,
            )
        ]
        for budget in TOTAL_VARIATION_BUDGETS:
            matching = cls._matching_requests(
                speculative_requests,
                SamplingSetting(REPRESENTATIVE_TEMPERATURE, budget),
            )
            rows.append(
                cls._summary_row(
                    FIXED_4,
                    REPRESENTATIVE_TEMPERATURE,
                    budget,
                    cls._summarize(matching),
                    target_summary,
                )
            )
        return cls._summary_table(rows)

    @staticmethod
    def _matching_requests(
        requests: tuple[SamplingRequestResult, ...],
        setting: SamplingSetting,
    ) -> tuple[SamplingRequestResult, ...]:
        matching = tuple(request for request in requests if request.setting == setting)
        if len(matching) != len(SEEDS):
            raise RuntimeError(f"setting {setting} does not have all expected seeds")
        return matching

    @classmethod
    def _summarize(
        cls,
        requests: tuple[SamplingRequestResult, ...],
    ) -> SamplingSummary:
        latencies = []
        decode_rates = []
        for request in requests:
            ttft, e2e = cls._required_latencies(request.metrics)
            latencies.append(e2e / 1_000_000)
            decode_rates.append(
                (request.metrics.output_token_count - 1) * 1_000_000 / (e2e - ttft)
            )

        drafts = [
            request.metrics.draft_token_metrics
            for request in requests
            if request.metrics.draft_token_metrics is not None
        ]
        if not drafts:
            return SamplingSummary(
                median_e2e_seconds=statistics.median(latencies),
                minimum_e2e_seconds=min(latencies),
                maximum_e2e_seconds=max(latencies),
                median_decode_tokens_per_second=statistics.median(decode_rates),
            )
        accepted = sum(draft.accepted_token_count for draft in drafts)
        proposed = sum(draft.proposed_token_count for draft in drafts)
        verification_rounds = sum(
            sum(uses for _count, uses in draft.selected_token_count_histogram)
            for draft in drafts
        )
        output_tokens = sum(request.metrics.output_token_count for request in requests)
        return SamplingSummary(
            median_e2e_seconds=statistics.median(latencies),
            minimum_e2e_seconds=min(latencies),
            maximum_e2e_seconds=max(latencies),
            median_decode_tokens_per_second=statistics.median(decode_rates),
            accepted_token_count=accepted,
            proposed_token_count=proposed,
            accepted_tokens_per_round=accepted / verification_rounds,
            output_tokens_per_round=output_tokens / verification_rounds,
        )

    @staticmethod
    def _summary_row(
        configuration: str,
        temperature: float,
        budget: float | None,
        summary: SamplingSummary,
        target_summary: SamplingSummary | None,
    ) -> tuple[str, ...]:
        speedup = (
            f"{target_summary.median_e2e_seconds / summary.median_e2e_seconds:.2f}x"
            if target_summary is not None
            else "-"
        )
        return (
            configuration,
            f"{temperature:g}",
            "-" if budget is None else f"{budget:g}",
            f"{summary.median_e2e_seconds:.3f} "
            f"[{summary.minimum_e2e_seconds:.3f}, "
            f"{summary.maximum_e2e_seconds:.3f}]",
            f"{summary.median_decode_tokens_per_second:.2f}",
            speedup,
            (
                "-"
                if not summary.proposed_token_count
                else f"{summary.accepted_token_count / summary.proposed_token_count * 100:.1f}%"
            ),
            (
                "-"
                if not summary.proposed_token_count
                else f"{summary.accepted_token_count}/{summary.proposed_token_count}"
            ),
            (
                "-"
                if summary.accepted_tokens_per_round is None
                else f"{summary.accepted_tokens_per_round:.2f}"
            ),
            (
                "-"
                if summary.output_tokens_per_round is None
                else f"{summary.output_tokens_per_round:.2f}"
            ),
        )

    @staticmethod
    def _summary_table(rows: list[tuple[str, ...]]) -> str:
        return tabulate(
            rows,
            headers=(
                "Configuration",
                "Temperature",
                "TV budget",
                "Median E2E [min, max] (s)",
                "Median decode (tok/s)",
                "vs target",
                "Acceptance",
                "Accepted/proposed",
                "Accepted/round",
                "Output/round",
            ),
            tablefmt="simple",
        )

    @staticmethod
    def _excerpt_report(
        requests: tuple[SamplingRequestResult, ...],
    ) -> str:
        excerpts = []
        for budget in TOTAL_VARIATION_BUDGETS:
            setting = SamplingSetting(REPRESENTATIVE_TEMPERATURE, budget)
            request = next(
                request
                for request in requests
                if request.setting == setting and request.seed == EXCERPT_SEED
            )
            normalized_output = " ".join(request.output.split())
            excerpts.append(f"TV budget {budget:g}: {normalized_output[:500]}")
        return "\n\n".join(excerpts)

    @classmethod
    def _machine_readable_results(
        cls,
        target_requests: tuple[SamplingRequestResult, ...],
        speculative_requests: tuple[SamplingRequestResult, ...],
    ) -> str:
        rows = [
            cls._machine_readable_result(
                "target_only",
                SamplingSetting(REPRESENTATIVE_TEMPERATURE, 0.0),
                cls._summarize(target_requests),
                total_variation_budget="none",
            )
        ]
        rows.extend(
            cls._machine_readable_result(
                "fixed_4",
                setting,
                cls._summarize(cls._matching_requests(speculative_requests, setting)),
                total_variation_budget=f"{setting.total_variation_budget:g}",
            )
            for setting in cls._speculative_settings()
        )
        return "\n".join(rows)

    @staticmethod
    def _machine_readable_result(
        configuration: str,
        setting: SamplingSetting,
        summary: SamplingSummary,
        *,
        total_variation_budget: str,
    ) -> str:
        accepted_per_round = (
            "none"
            if summary.accepted_tokens_per_round is None
            else f"{summary.accepted_tokens_per_round:.6f}"
        )
        output_per_round = (
            "none"
            if summary.output_tokens_per_round is None
            else f"{summary.output_tokens_per_round:.6f}"
        )
        return (
            "SPECULATIVE_SAMPLING_TRADEOFF_RESULT "
            f"configuration={configuration} "
            f"temperature={setting.temperature:g} "
            f"total_variation_budget={total_variation_budget} "
            f"seed_count={len(SEEDS)} "
            f"median_e2e_seconds={summary.median_e2e_seconds:.6f} "
            f"minimum_e2e_seconds={summary.minimum_e2e_seconds:.6f} "
            f"maximum_e2e_seconds={summary.maximum_e2e_seconds:.6f} "
            "median_decode_tokens_per_second="
            f"{summary.median_decode_tokens_per_second:.6f} "
            f"accepted_token_count={summary.accepted_token_count} "
            f"proposed_token_count={summary.proposed_token_count} "
            f"accepted_tokens_per_round={accepted_per_round} "
            f"output_tokens_per_round={output_per_round}"
        )

    @staticmethod
    def _required_latencies(metrics: GenerationMetrics) -> tuple[int, int]:
        ttft = metrics.time_to_first_token_microseconds
        e2e = metrics.end_to_end_latency_microseconds
        if ttft is None or ttft <= 0 or e2e <= ttft:
            raise RuntimeError(f"invalid generation latencies: TTFT={ttft}, E2E={e2e}")
        return ttft, e2e
