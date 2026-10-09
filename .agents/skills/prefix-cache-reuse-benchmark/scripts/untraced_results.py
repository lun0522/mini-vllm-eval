#!/usr/bin/env python3
"""Aggregate repeated prefix-cache reuse benchmark logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from tabulate import tabulate


CASES = ("no_prefix_cache", "prefix_cache")
WORKLOAD_VARIANTS = {
    "overlap": ("0", "25", "50", "75", "100"),
    "fan_out": ("1", "2", "4", "8"),
    "burst": ("cold", "warm"),
    "churn": ("1", "2"),
}
RESULT_PATTERN = re.compile(
    r"PREFIX_CACHE_REUSE_RESULT "
    r"case=(\w+) workload=(\w+) variant=(\w+) requests=(\d+) "
    r"input_tokens=(\d+) output_tokens=(\d+) restored_tokens=(\d+) "
    r"indexed_tokens=(\d+) mean_prefill_us=(\d+) mean_ttft_us=(\d+) "
    r"max_e2e_us=(\d+)"
)
MAXIMUM_CV = 0.05
HEADLINES = (
    ("overlap", "100", "mean_ttft_us"),
    ("fan_out", "8", "max_e2e_us"),
    ("burst", "cold", "mean_ttft_us"),
    ("burst", "warm", "mean_ttft_us"),
    ("churn", "2", "mean_prefill_us"),
)


@dataclass(frozen=True)
class Measurements:
    request_count: int
    input_token_count: int
    output_token_count: int
    restored_token_count: int
    indexed_token_count: int
    mean_prefill_us: int
    mean_ttft_us: int
    max_e2e_us: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate untraced prefix-cache reuse benchmark logs."
    )
    parser.add_argument("logs", type=Path, nargs="+", help="three or five logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.logs]
    if len(paths) not in (3, 5):
        raise SystemExit("provide exactly three or five benchmark logs")
    if len(set(paths)) != len(paths):
        raise SystemExit("benchmark log paths must be distinct")
    runs = [_parse_log(path) for path in paths]
    _validate_consistency(runs)
    _print_summary(runs)
    _print_comparisons(runs)
    _print_variance_decision(runs)
    return 0


def _expected_keys() -> set[tuple[str, str, str]]:
    return {
        (case, workload, variant)
        for case in CASES
        for workload, variants in WORKLOAD_VARIANTS.items()
        for variant in variants
    }


def _parse_log(path: Path) -> dict[tuple[str, str, str], Measurements]:
    if not path.is_file():
        raise SystemExit(f"benchmark log does not exist: {path}")
    results: dict[tuple[str, str, str], Measurements] = {}
    for match in RESULT_PATTERN.finditer(path.read_text()):
        key = (match.group(1), match.group(2), match.group(3))
        if key in results:
            raise SystemExit(f"duplicate result for {key} in {path}")
        results[key] = Measurements(*(int(match.group(i)) for i in range(4, 12)))
    expected = _expected_keys()
    if set(results) != expected:
        missing = sorted(expected - set(results))
        unexpected = sorted(set(results) - expected)
        raise SystemExit(
            f"{path} has incomplete results; missing={missing}, unexpected={unexpected}"
        )
    return results


def _validate_consistency(
    runs: list[dict[tuple[str, str, str], Measurements]],
) -> None:
    expected = {
        key: (value.request_count, value.input_token_count, value.output_token_count)
        for key, value in runs[0].items()
    }
    for index, run in enumerate(runs[1:], start=2):
        observed = {
            key: (
                value.request_count,
                value.input_token_count,
                value.output_token_count,
            )
            for key, value in run.items()
        }
        if observed != expected:
            raise SystemExit(f"run {index} has different request or token counts")
    for run_index, run in enumerate(runs, start=1):
        for workload, variants in WORKLOAD_VARIANTS.items():
            for variant in variants:
                baseline = run[(CASES[0], workload, variant)]
                cached = run[(CASES[1], workload, variant)]
                baseline_shape = (
                    baseline.request_count,
                    baseline.input_token_count,
                    baseline.output_token_count,
                )
                cached_shape = (
                    cached.request_count,
                    cached.input_token_count,
                    cached.output_token_count,
                )
                if baseline_shape != cached_shape:
                    raise SystemExit(
                        f"run {run_index} cases differ for {workload}/{variant}"
                    )


def _print_summary(
    runs: list[dict[tuple[str, str, str], Measurements]],
) -> None:
    print(f"Prefix-cache reuse aggregate: {len(runs)} runs")
    rows = []
    for case in CASES:
        for workload, variants in WORKLOAD_VARIANTS.items():
            for variant in variants:
                values = [run[(case, workload, variant)] for run in runs]
                rows.append(
                    (
                        case,
                        workload,
                        variant,
                        _distribution([value.restored_token_count for value in values]),
                        _distribution([value.indexed_token_count for value in values]),
                        _milliseconds([value.mean_prefill_us for value in values]),
                        _milliseconds([value.mean_ttft_us for value in values]),
                        _milliseconds([value.max_e2e_us for value in values]),
                    )
                )
    print("\nWorkload summary")
    print(
        tabulate(
            rows,
            headers=(
                "Case",
                "Workload",
                "Variant",
                "Restored tokens",
                "Indexed tokens",
                "Mean prefill ms",
                "Mean TTFT ms",
                "Max E2E ms",
            ),
            tablefmt="simple",
        )
    )


def _print_comparisons(
    runs: list[dict[tuple[str, str, str], Measurements]],
) -> None:
    rows = []
    for workload, variants in WORKLOAD_VARIANTS.items():
        for variant in variants:
            reductions = {"prefill": [], "ttft": [], "e2e": []}
            for run in runs:
                baseline = run[(CASES[0], workload, variant)]
                cached = run[(CASES[1], workload, variant)]
                reductions["prefill"].append(
                    1.0 - cached.mean_prefill_us / baseline.mean_prefill_us
                )
                reductions["ttft"].append(
                    1.0 - cached.mean_ttft_us / baseline.mean_ttft_us
                )
                reductions["e2e"].append(
                    1.0 - cached.max_e2e_us / baseline.max_e2e_us
                )
            rows.append(
                (
                    workload,
                    variant,
                    _percent_distribution(reductions["prefill"]),
                    _percent_distribution(reductions["ttft"]),
                    _percent_distribution(reductions["e2e"]),
                )
            )
    print("\nPrefix-cache improvement over ordinary paged cache")
    print(
        tabulate(
            rows,
            headers=(
                "Workload",
                "Variant",
                "Prefill reduction",
                "TTFT reduction",
                "Max E2E reduction",
            ),
            tablefmt="simple",
        )
    )


def _print_variance_decision(
    runs: list[dict[tuple[str, str, str], Measurements]],
) -> None:
    violations = []
    for case in CASES:
        for workload, variant, field in HEADLINES:
            values = [getattr(run[(case, workload, variant)], field) for run in runs]
            cv = stdev(values) / mean(values)
            if cv > MAXIMUM_CV:
                violations.append((case, workload, variant, field, cv))
    print("\nVariance decision")
    if violations:
        print("Collect two additional runs; headline CV exceeded 5%:")
        for case, workload, variant, field, cv in violations:
            print(f"- {case} {workload}/{variant} {field}: {cv:.1%}")
    else:
        print("Three runs are sufficient; every headline CV is at most 5%.")


def _distribution(values: list[int | float]) -> str:
    return f"{mean(values):.1f} ± {stdev(values):.1f}"


def _milliseconds(values_microseconds: list[int]) -> str:
    return _distribution([value / 1000 for value in values_microseconds])


def _percent_distribution(values: list[float]) -> str:
    return f"{mean(values):.1%} ± {stdev(values):.1%}"


if __name__ == "__main__":
    raise SystemExit(main())
