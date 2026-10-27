"""Focused follow-up for speculative-sampling proposal length and TV budget."""

from __future__ import annotations

import json
import os
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


TEMPERATURE = 0.7
SEEDS = (17, 101, 20261022)
MAX_NEW_TOKENS = 256
TARGET_KV_CACHE_SIZE_BYTES = 1024 * 1024 * 1024
GPU_ENVIRONMENT = (("MINI_VLLM_METAL_GEMV_MAX_ROWS", "4"),)
CONFIG_ENVIRONMENT = "MINI_VLLM_SAMPLING_PROPOSAL_BREAKEVEN_CASES"
DEFAULT_CONFIG = "target,draft,4:1,6:1,8:1"


@dataclass(frozen=True)
class Configuration:
    name: str
    proposal_length: int | None
    total_variation_budget: float | None
    model: str


@dataclass(frozen=True)
class RequestResult:
    seed: int
    output: str
    metrics: GenerationMetrics


@dataclass(frozen=True)
class CaseResult:
    configuration: Configuration
    requests: tuple[RequestResult, ...]


def _parse_configurations(value: str) -> tuple[Configuration, ...]:
    configurations = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if item == "target":
            configuration = Configuration("Target only", None, None, QWEN_7B_MODEL)
        elif item == "draft":
            configuration = Configuration("Draft only", None, None, QWEN_0_5B_MODEL)
        else:
            length_text, separator, budget_text = item.partition(":")
            if not separator:
                raise ValueError(f"invalid follow-up configuration: {raw_item!r}")
            length = int(length_text)
            budget = float(budget_text)
            if length <= 0 or not 0.0 <= budget <= 1.0:
                raise ValueError(f"invalid follow-up configuration: {raw_item!r}")
            configuration = Configuration(
                f"Fixed {length} / TV={budget:g}",
                length,
                budget,
                QWEN_7B_MODEL,
            )
        if any(existing.name == configuration.name for existing in configurations):
            raise ValueError(f"duplicate follow-up configuration: {configuration.name}")
        configurations.append(configuration)
    if not configurations:
        raise ValueError("the focused follow-up requires at least one configuration")
    return tuple(configurations)


class SpeculativeSamplingProposalBreakevenBenchmark(Benchmark):
    def __init__(self) -> None:
        self.configurations = _parse_configurations(
            os.environ.get(CONFIG_ENVIRONMENT, DEFAULT_CONFIG)
        )

    @staticmethod
    def _server_flags(model: str) -> list[str]:
        return [
            "--model",
            model,
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

    def server_flags(self) -> list[str]:
        return self._server_flags(QWEN_7B_MODEL)

    def cases(self) -> tuple[BenchmarkCase, ...]:
        cases = []
        for configuration in self.configurations:
            flags = self._server_flags(configuration.model)
            if configuration.proposal_length is not None:
                draft_model = (
                    f"model {{ {QWEN_0_5B_MODEL} }} token_count_policy {{ fixed {{ "
                    f"draft_token_count: {configuration.proposal_length} }} }}"
                )
                flags.extend(("--draft-model", draft_model))
            cases.append(
                BenchmarkCase(configuration.name, tuple(flags), GPU_ENVIRONMENT)
            )
        return tuple(cases)

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> CaseResult:
        configuration = next(c for c in self.configurations if c.name == case.name)
        self._run_request(client, proto, configuration, SEEDS[0], 1)
        requests = tuple(
            self._run_request(client, proto, configuration, seed, MAX_NEW_TOKENS)
            for seed in SEEDS
        )
        return CaseResult(configuration, requests)

    @staticmethod
    def _run_request(
        client: Any,
        proto: ProtoModules,
        configuration: Configuration,
        seed: int,
        max_new_tokens: int,
    ) -> RequestResult:
        budget = configuration.total_variation_budget or 0.0
        request = proto.request_handler.GenerateText(
            prompt=CREATIVE_WRITING_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            total_variation_budget=budget,
            seed=seed,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        logger.warning(
            "Sending {} request with temperature={}, TV={}, seed={}, max_new_tokens={}",
            configuration.name,
            TEMPERATURE,
            budget,
            seed,
            max_new_tokens,
        )
        metrics = None
        output_parts = []
        for response in client.GenerateText(request):
            event = response.WhichOneof("event")
            if event == "text":
                output_parts.append(response.text)
            elif event == "stats":
                metrics = Benchmark.generation_metrics(response.stats)
        if metrics is None:
            raise RuntimeError(f"{configuration.name} completed without statistics")
        return RequestResult(seed, "".join(output_parts), metrics)

    def report_results(self, results: list[tuple[BenchmarkCase, Any]]) -> None:
        if len(results) != len(self.configurations):
            raise RuntimeError("focused follow-up returned an unexpected case count")
        rows = []
        machine_rows = []
        for (case, result), expected in zip(results, self.configurations, strict=True):
            if not isinstance(result, CaseResult) or result.configuration != expected:
                raise RuntimeError(f"invalid result for {case.name}")
            summary = self._summarize(result)
            rows.append(summary[:6])
            machine_rows.append(summary[6])
        logger.info(
            "Focused speculative-sampling follow-up:\n{}",
            tabulate(
                rows,
                headers=(
                    "Configuration",
                    "Median E2E [min, max] (s)",
                    "Median decode (tok/s)",
                    "Acceptance",
                    "Accepted/round",
                    "Output/round",
                ),
                tablefmt="simple",
            ),
        )
        logger.info("Machine-readable focused results:\n{}", "\n".join(machine_rows))
        output_rows = [
            json.dumps(
                {
                    "configuration": result.configuration.name,
                    "seed": request.seed,
                    "output": request.output,
                },
                ensure_ascii=False,
            )
            for _case, result in results
            for request in result.requests
        ]
        logger.info("Complete focused outputs:\n{}", "\n".join(output_rows))

    @staticmethod
    def _summarize(result: CaseResult) -> tuple[str, ...]:
        latencies = []
        decode_rates = []
        drafts: list[DraftTokenMetrics] = []
        for request in result.requests:
            metrics = request.metrics
            ttft = metrics.time_to_first_token_microseconds
            e2e = metrics.end_to_end_latency_microseconds
            if ttft is None or ttft <= 0 or e2e <= ttft:
                raise RuntimeError(f"invalid latency for {result.configuration.name}")
            if metrics.output_token_count != MAX_NEW_TOKENS:
                raise RuntimeError(
                    f"unexpected output length for {result.configuration.name}"
                )
            latencies.append(e2e / 1_000_000)
            decode_rates.append(
                (metrics.output_token_count - 1) * 1_000_000 / (e2e - ttft)
            )
            if metrics.draft_token_metrics is not None:
                drafts.append(metrics.draft_token_metrics)
        median = statistics.median(latencies)
        accepted = sum(d.accepted_token_count for d in drafts)
        proposed = sum(d.proposed_token_count for d in drafts)
        rounds = sum(
            sum(uses for _, uses in d.selected_token_count_histogram) for d in drafts
        )
        output_tokens = sum(r.metrics.output_token_count for r in result.requests)
        acceptance = "-" if not proposed else f"{accepted / proposed * 100:.1f}%"
        accepted_per_round = "-" if not rounds else f"{accepted / rounds:.2f}"
        output_per_round = "-" if not rounds else f"{output_tokens / rounds:.2f}"
        configuration = result.configuration
        budget = configuration.total_variation_budget
        budget_value = budget if budget is not None else "none"
        machine = (
            "SPECULATIVE_SAMPLING_FOLLOWUP_RESULT "
            f"configuration={configuration.name.replace(' ', '_')} "
            f"proposal_length={configuration.proposal_length or 'none'} "
            f"total_variation_budget={budget_value} "
            f"seed_count={len(SEEDS)} median_e2e_seconds={median:.6f} "
            f"minimum_e2e_seconds={min(latencies):.6f} "
            f"maximum_e2e_seconds={max(latencies):.6f} "
            f"median_decode_tokens_per_second={statistics.median(decode_rates):.6f} "
            f"accepted_token_count={accepted} proposed_token_count={proposed} "
            f"accepted_tokens_per_round={accepted / rounds if rounds else 'none'} "
            f"output_tokens_per_round={output_tokens / rounds if rounds else 'none'} "
            f"verification_round_count={rounds if rounds else 'none'}"
        )
        return (
            configuration.name,
            f"{median:.3f} [{min(latencies):.3f}, {max(latencies):.3f}]",
            f"{statistics.median(decode_rates):.2f}",
            acceptance,
            accepted_per_round,
            output_per_round,
            machine,
        )
