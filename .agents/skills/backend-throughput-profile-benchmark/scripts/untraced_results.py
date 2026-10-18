#!/usr/bin/env python3
"""Aggregate repeated backend-throughput profile benchmark logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from statistics import stdev

from tabulate import tabulate


WORKLOADS = tuple(
    f"input_{input_token_count}_output_{output_token_count}"
    for input_token_count in (128, 512, 2048)
    for output_token_count in (16, 128, 512, 1024)
)
RESULT_ROW = re.compile(
    r"BACKEND_THROUGHPUT_PROFILE_RESULT "
    r"workload=(\S+) input_tokens=(\d+) max_output_tokens=(\d+) measured_runs=(\d+) "
    r"cpu_prefill_us=(\d+) cpu_prefill_cv=([0-9.]+) "
    r"cpu_prefill_tokens_per_second=(\d+) cpu_decode_us=(\d+) "
    r"cpu_decode_cv=([0-9.]+) cpu_decode_tokens_per_second=(\d+) "
    r"gpu_prefill_us=(\d+) gpu_prefill_cv=([0-9.]+) "
    r"gpu_prefill_tokens_per_second=(\d+) gpu_decode_us=(\d+) "
    r"gpu_decode_cv=([0-9.]+) gpu_decode_tokens_per_second=(\d+)"
)
MAXIMUM_CV = 0.05


@dataclass(frozen=True)
class Measurement:
    input_token_count: int
    max_output_token_count: int
    cpu_prefill_microseconds: int
    cpu_decode_microseconds: int
    gpu_prefill_microseconds: int
    gpu_decode_microseconds: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate backend-throughput profile benchmark logs."
    )
    parser.add_argument(
        "logs",
        type=Path,
        nargs="+",
        help="three or five complete benchmark logs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_paths = [path.expanduser().resolve() for path in args.logs]
    if len(log_paths) not in (3, 5):
        raise SystemExit("provide exactly three or five benchmark logs")
    if len(set(log_paths)) != len(log_paths):
        raise SystemExit("benchmark log paths must be distinct")

    runs = [_parse_log(path) for path in log_paths]
    _validate_workload_shapes(runs)
    maximum_cv = _print_results(runs)
    if maximum_cv > MAXIMUM_CV:
        print(
            f"\nMaximum phase-duration CV is {maximum_cv:.1%}; "
            "collect five total runs."
        )
    else:
        print(f"\nMaximum phase-duration CV is {maximum_cv:.1%}; profiles are stable.")
    return 0


def _parse_log(path: Path) -> dict[str, Measurement]:
    if not path.is_file():
        raise SystemExit(f"benchmark log does not exist: {path}")
    measurements = {}
    for match in RESULT_ROW.finditer(path.read_text()):
        workload = match.group(1)
        if workload in measurements:
            raise SystemExit(f"duplicate {workload} result in {path}")
        if int(match.group(4)) != 3:
            raise SystemExit(f"{workload} in {path} did not contain three measured runs")
        measurements[workload] = Measurement(
            input_token_count=int(match.group(2)),
            max_output_token_count=int(match.group(3)),
            cpu_prefill_microseconds=int(match.group(5)),
            cpu_decode_microseconds=int(match.group(8)),
            gpu_prefill_microseconds=int(match.group(11)),
            gpu_decode_microseconds=int(match.group(14)),
        )
    if set(measurements) != set(WORKLOADS):
        missing = sorted(set(WORKLOADS) - set(measurements))
        raise SystemExit(f"{path} is missing workload results: {missing}")
    return measurements


def _validate_workload_shapes(runs: list[dict[str, Measurement]]) -> None:
    expected = {
        workload: (
            runs[0][workload].input_token_count,
            runs[0][workload].max_output_token_count,
        )
        for workload in WORKLOADS
    }
    for index, run in enumerate(runs[1:], start=2):
        observed = {
            workload: (
                run[workload].input_token_count,
                run[workload].max_output_token_count,
            )
            for workload in WORKLOADS
        }
        if observed != expected:
            raise SystemExit(f"run {index} has different workload token counts")


def _print_results(runs: list[dict[str, Measurement]]) -> float:
    rows = []
    profiles = []
    coefficients_of_variation = []
    for workload in WORKLOADS:
        values = [run[workload] for run in runs]
        first = values[0]
        cpu_prefill = [value.cpu_prefill_microseconds for value in values]
        cpu_decode = [value.cpu_decode_microseconds for value in values]
        gpu_prefill = [value.gpu_prefill_microseconds for value in values]
        gpu_decode = [value.gpu_decode_microseconds for value in values]
        phase_durations = (cpu_prefill, cpu_decode, gpu_prefill, gpu_decode)
        phase_means = tuple(mean(durations) for durations in phase_durations)
        phase_cvs = tuple(
            stdev(durations) / phase_mean
            for durations, phase_mean in zip(phase_durations, phase_means)
        )
        coefficients_of_variation.extend(phase_cvs)
        cpu_prefill_rate = _tokens_per_second(first.input_token_count, phase_means[0])
        cpu_decode_rate = _tokens_per_second(
            first.max_output_token_count - 1, phase_means[1]
        )
        gpu_prefill_rate = _tokens_per_second(first.input_token_count, phase_means[2])
        gpu_decode_rate = _tokens_per_second(
            first.max_output_token_count - 1, phase_means[3]
        )
        rows.append(
            (
                workload,
                first.input_token_count,
                first.max_output_token_count,
                _distribution(cpu_prefill),
                f"{phase_cvs[0]:.1%}",
                cpu_prefill_rate,
                _distribution(cpu_decode),
                f"{phase_cvs[1]:.1%}",
                cpu_decode_rate,
                _distribution(gpu_prefill),
                f"{phase_cvs[2]:.1%}",
                gpu_prefill_rate,
                _distribution(gpu_decode),
                f"{phase_cvs[3]:.1%}",
                gpu_decode_rate,
            )
        )
        profiles.append(
            "throughput_profile { "
            f"input_token_count: {first.input_token_count} "
            f"max_output_token_count: {first.max_output_token_count} "
            f"cpu_prefill_tokens_per_second: {cpu_prefill_rate} "
            f"cpu_decode_tokens_per_second: {cpu_decode_rate} "
            f"gpu_prefill_tokens_per_second: {gpu_prefill_rate} "
            f"gpu_decode_tokens_per_second: {gpu_decode_rate} "
            "}"
        )

    print(f"Backend-throughput profile aggregate: {len(runs)} runs")
    print(
        tabulate(
            rows,
            headers=(
                "Workload",
                "Input",
                "Max output",
                "CPU prefill us",
                "CPU prefill CV",
                "CPU prefill tok/s",
                "CPU decode us",
                "CPU decode CV",
                "CPU decode tok/s",
                "GPU prefill us",
                "GPU prefill CV",
                "GPU prefill tok/s",
                "GPU decode us",
                "GPU decode CV",
                "GPU decode tok/s",
            ),
            tablefmt="simple",
        )
    )
    print("\nPhase-specific backend throughput profiles")
    print("\n".join(profiles))
    return max(coefficients_of_variation)


def _distribution(values: list[int]) -> str:
    return f"{mean(values):.0f} ± {stdev(values):.0f}"


def _tokens_per_second(token_count: int, duration_microseconds: float) -> int:
    return max(1, round(token_count * 1_000_000 / duration_microseconds))


if __name__ == "__main__":
    raise SystemExit(main())
