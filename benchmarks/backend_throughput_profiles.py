"""Calibrate CPU and GPU prefill and decode throughput for request routing."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from statistics import mean
from statistics import stdev
from typing import Any

from loguru import logger
from tabulate import tabulate

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import GenerationMetrics
from benchmarks.base import QWEN_0_5B_MODEL
from proto_loader import ProtoModules


INPUT_TOKEN_COUNTS = (128, 512, 2048)
OUTPUT_TOKEN_COUNTS = (16, 128, 512, 1024)
MEASURED_RUN_COUNT = 3
TARGET_KV_CACHE_SIZE_BYTES = 256 * 1024 * 1024
DEVICE_NAMES = ("cpu", "gpu")
MAXIMUM_PROMPT_CALIBRATION_ATTEMPTS = 8


@dataclass(frozen=True)
class Workload:
    input_token_count: int
    max_output_token_count: int

    @property
    def name(self) -> str:
        return f"input_{self.input_token_count}_output_{self.max_output_token_count}"


WORKLOADS = tuple(
    Workload(input_token_count, output_token_count)
    for input_token_count in INPUT_TOKEN_COUNTS
    for output_token_count in OUTPUT_TOKEN_COUNTS
)


@dataclass(frozen=True)
class WorkloadResult:
    workload: Workload
    metrics: tuple[GenerationMetrics, ...]

    @property
    def mean_prefill_duration_microseconds(self) -> float:
        return mean(metric.prefill_duration_microseconds for metric in self.metrics)

    @property
    def mean_decode_duration_microseconds(self) -> float:
        return mean(metric.decode_duration_microseconds for metric in self.metrics)

    @property
    def prefill_duration_cv(self) -> float:
        durations = tuple(
            metric.prefill_duration_microseconds for metric in self.metrics
        )
        return stdev(durations) / mean(durations)

    @property
    def decode_duration_cv(self) -> float:
        durations = tuple(metric.decode_duration_microseconds for metric in self.metrics)
        return stdev(durations) / mean(durations)

    @property
    def prefill_tokens_per_second(self) -> int:
        return max(
            1,
            round(
                self.workload.input_token_count
                * 1_000_000
                / self.mean_prefill_duration_microseconds
            ),
        )

    @property
    def decode_tokens_per_second(self) -> int:
        return max(
            1,
            round(
                (self.workload.max_output_token_count - 1)
                * 1_000_000
                / self.mean_decode_duration_microseconds
            ),
        )


@dataclass(frozen=True)
class DeviceResult:
    device: str
    workloads: tuple[WorkloadResult, ...]


class BackendThroughputProfilesBenchmark(Benchmark):
    def __init__(self) -> None:
        self._prompts_by_input_token_count: dict[int, str] = {}

    def server_flags(self) -> list[str]:
        return [
            "--model",
            QWEN_0_5B_MODEL,
            "--activation-dtype",
            "f32",
            "--kv-cache-config",
            (
                "kv_cache_type: KV_CACHE_TYPE_PAGED per_page_token_count: 16 "
                f"target_kv_cache_size_bytes: {TARGET_KV_CACHE_SIZE_BYTES}"
            ),
            "--scheduler-config",
            "max_batched_token_count: 2048 max_active_request_count: 1",
        ]

    def cases(self) -> tuple[BenchmarkCase, ...]:
        server_flags = tuple(self.server_flags())
        return tuple(
            BenchmarkCase(
                device,
                (*server_flags, "--inference-device", device),
            )
            for device in DEVICE_NAMES
        )

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> DeviceResult:
        if not self._prompts_by_input_token_count:
            self._prompts_by_input_token_count = {
                input_token_count: self._calibrate_prompt(
                    client,
                    proto,
                    input_token_count,
                )
                for input_token_count in INPUT_TOKEN_COUNTS
            }

        workload_results = []
        for workload in WORKLOADS:
            prompt = self._prompts_by_input_token_count[workload.input_token_count]
            logger.info("Warming up {} on {}", workload.name, case.name)
            warmup_metrics = self._run_request(
                client,
                proto,
                prompt,
                workload.max_output_token_count,
            )
            self._validate_metrics(workload, warmup_metrics)

            measured_metrics = []
            for measured_run in range(MEASURED_RUN_COUNT):
                logger.info(
                    "Measuring {} on {}: run {}/{}",
                    workload.name,
                    case.name,
                    measured_run + 1,
                    MEASURED_RUN_COUNT,
                )
                metrics = self._run_request(
                    client,
                    proto,
                    prompt,
                    workload.max_output_token_count,
                )
                self._validate_metrics(workload, metrics)
                measured_metrics.append(metrics)
            workload_results.append(WorkloadResult(workload, tuple(measured_metrics)))

        return DeviceResult(
            device=case.name,
            workloads=tuple(workload_results),
        )

    def report_results(
        self,
        results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        results_by_device: dict[str, DeviceResult] = {}
        for case, result in results:
            if not isinstance(result, DeviceResult):
                raise RuntimeError(
                    f"case {case.name} did not return backend-throughput results"
                )
            if result.device in results_by_device:
                raise RuntimeError(f"received duplicate {result.device} results")
            results_by_device[result.device] = result
        if set(results_by_device) != set(DEVICE_NAMES):
            raise RuntimeError("backend-throughput benchmark requires CPU and GPU results")

        results_by_workload = {
            device: {result.workload: result for result in device_result.workloads}
            for device, device_result in results_by_device.items()
        }
        rows = []
        profiles = []
        for workload in WORKLOADS:
            cpu = results_by_workload["cpu"][workload]
            gpu = results_by_workload["gpu"][workload]
            rows.append(
                (
                    workload.input_token_count,
                    workload.max_output_token_count,
                    f"{cpu.mean_prefill_duration_microseconds / 1_000:.3f}",
                    f"{cpu.prefill_duration_cv:.1%}",
                    cpu.prefill_tokens_per_second,
                    f"{cpu.mean_decode_duration_microseconds / 1_000:.3f}",
                    f"{cpu.decode_duration_cv:.1%}",
                    cpu.decode_tokens_per_second,
                    f"{gpu.mean_prefill_duration_microseconds / 1_000:.3f}",
                    f"{gpu.prefill_duration_cv:.1%}",
                    gpu.prefill_tokens_per_second,
                    f"{gpu.mean_decode_duration_microseconds / 1_000:.3f}",
                    f"{gpu.decode_duration_cv:.1%}",
                    gpu.decode_tokens_per_second,
                )
            )
            profiles.append(self._format_profile(workload, cpu, gpu))
            logger.info(
                "BACKEND_THROUGHPUT_PROFILE_RESULT "
                "workload={} input_tokens={} max_output_tokens={} measured_runs={} "
                "cpu_prefill_us={} cpu_prefill_cv={:.6f} cpu_prefill_tokens_per_second={} "
                "cpu_decode_us={} cpu_decode_cv={:.6f} cpu_decode_tokens_per_second={} "
                "gpu_prefill_us={} gpu_prefill_cv={:.6f} gpu_prefill_tokens_per_second={} "
                "gpu_decode_us={} gpu_decode_cv={:.6f} gpu_decode_tokens_per_second={}",
                workload.name,
                workload.input_token_count,
                workload.max_output_token_count,
                MEASURED_RUN_COUNT,
                round(cpu.mean_prefill_duration_microseconds),
                cpu.prefill_duration_cv,
                cpu.prefill_tokens_per_second,
                round(cpu.mean_decode_duration_microseconds),
                cpu.decode_duration_cv,
                cpu.decode_tokens_per_second,
                round(gpu.mean_prefill_duration_microseconds),
                gpu.prefill_duration_cv,
                gpu.prefill_tokens_per_second,
                round(gpu.mean_decode_duration_microseconds),
                gpu.decode_duration_cv,
                gpu.decode_tokens_per_second,
            )

        logger.info(
            "Backend throughput profiles:\n{}",
            tabulate(
                rows,
                headers=(
                    "Input",
                    "Max output",
                    "CPU prefill ms",
                    "CPU prefill CV",
                    "CPU prefill tok/s",
                    "CPU decode ms",
                    "CPU decode CV",
                    "CPU decode tok/s",
                    "GPU prefill ms",
                    "GPU prefill CV",
                    "GPU prefill tok/s",
                    "GPU decode ms",
                    "GPU decode CV",
                    "GPU decode tok/s",
                ),
                tablefmt="rounded_outline",
            ),
        )
        logger.info(
            "Phase-specific backend throughput profiles:\n{}",
            "\n".join(profiles),
        )

    @classmethod
    def _calibrate_prompt(
        cls,
        client: Any,
        proto: ProtoModules,
        target_input_token_count: int,
    ) -> str:
        repeat_count = target_input_token_count
        attempted_repeat_counts = set()
        for _ in range(MAXIMUM_PROMPT_CALIBRATION_ATTEMPTS):
            if repeat_count in attempted_repeat_counts:
                break
            attempted_repeat_counts.add(repeat_count)
            prompt = ("a " * repeat_count).rstrip()
            observed_input_token_count = cls._run_request(
                client,
                proto,
                prompt,
                1,
            ).input_token_count
            if observed_input_token_count == target_input_token_count:
                logger.info(
                    "Calibrated prompt with {} input tokens using {} repeated words",
                    target_input_token_count,
                    repeat_count,
                )
                return prompt
            repeat_count = max(
                1,
                repeat_count
                + target_input_token_count
                - observed_input_token_count,
            )
        raise RuntimeError(
            f"could not construct a {target_input_token_count}-token calibration prompt"
        )

    @classmethod
    def _run_request(
        cls,
        client: Any,
        proto: ProtoModules,
        prompt: str,
        max_output_token_count: int,
    ) -> GenerationMetrics:
        request = proto.request_handler.GenerateText(
            prompt=prompt,
            max_new_tokens=max_output_token_count,
            repeat_penalty=1.0,
            repeat_last_n=0,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        stats = None
        for response in client.GenerateText(request):
            if response.WhichOneof("event") == "stats":
                stats = response.stats
        if stats is None:
            raise RuntimeError("generation completed without statistics")
        metrics = cls.generation_metrics(stats)
        if metrics.output_token_count != max_output_token_count:
            raise RuntimeError(
                "generation returned "
                f"{metrics.output_token_count} output tokens; expected "
                f"{max_output_token_count}"
            )
        if (
            metrics.prefill_duration_microseconds
            + metrics.decode_duration_microseconds
            <= 0
        ):
            raise RuntimeError("generation reported a non-positive execution duration")
        return metrics

    @staticmethod
    def _validate_metrics(workload: Workload, metrics: GenerationMetrics) -> None:
        if metrics.input_token_count != workload.input_token_count:
            raise RuntimeError(
                f"{workload.name} returned {metrics.input_token_count} input tokens"
            )
        if metrics.output_token_count != workload.max_output_token_count:
            raise RuntimeError(
                f"{workload.name} returned {metrics.output_token_count} output tokens"
            )

    @staticmethod
    def _format_profile(
        workload: Workload,
        cpu: WorkloadResult,
        gpu: WorkloadResult,
    ) -> str:
        return (
            "throughput_profile { "
            f"input_token_count: {workload.input_token_count} "
            f"max_output_token_count: {workload.max_output_token_count} "
            f"cpu_prefill_tokens_per_second: {cpu.prefill_tokens_per_second} "
            f"cpu_decode_tokens_per_second: {cpu.decode_tokens_per_second} "
            f"gpu_prefill_tokens_per_second: {gpu.prefill_tokens_per_second} "
            f"gpu_decode_tokens_per_second: {gpu.decode_tokens_per_second} "
            "}"
        )
