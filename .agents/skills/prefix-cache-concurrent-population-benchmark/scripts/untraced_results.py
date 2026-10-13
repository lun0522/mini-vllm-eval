#!/usr/bin/env python3
"""Aggregate repeated prefix publication and single-flight benchmark logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from tabulate import tabulate


PUBLICATION_POLICIES = ("completion", "incremental")
COORDINATION_POLICIES = ("independent", "single_flight")
CASES = tuple(
    f"{publication}_{coordination}"
    for publication in PUBLICATION_POLICIES
    for coordination in COORDINATION_POLICIES
)
IDENTICAL_VARIANTS = tuple(
    ("identical", concurrency, output_tokens)
    for output_tokens in (16, 128)
    for concurrency in (2, 4, 8)
)
INFLIGHT_VARIANTS = tuple(
    ("inflight", concurrency, 128) for concurrency in (2, 4, 8)
)
VARIANTS = (*IDENTICAL_VARIANTS, *INFLIGHT_VARIANTS, ("distinct", 8, 16))
RESULT_PATTERN = re.compile(
    r"PREFIX_CACHE_CONCURRENT_POPULATION_RESULT "
    r"case=(\w+) workload=(\w+) concurrency=(\d+) requested_output_tokens=(\d+) "
    r"requests=(\d+) input_tokens=(\d+) generated_tokens=(\d+) restored_tokens=(\d+) "
    r"indexed_tokens=(\d+) sum_prefill_us=(\d+) mean_ttft_us=(\d+) "
    r"max_e2e_us=(\d+) wall_us=(\d+)"
)
MAXIMUM_CV = 0.05
RACE_SENSITIVE_CASES = {("incremental_independent", "identical")}


@dataclass(frozen=True)
class Measurements:
    request_count: int
    input_token_count: int
    generated_token_count: int
    restored_token_count: int
    indexed_token_count: int
    sum_prefill_us: int
    mean_ttft_us: int
    max_e2e_us: int
    wall_us: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate untraced prefix-cache publication and single-flight "
            "benchmark logs."
        )
    )
    parser.add_argument("logs", type=Path, nargs="+", help="three or five logs")
    return parser.parse_args()


def main() -> int:
    paths = [path.expanduser().resolve() for path in parse_args().logs]
    if len(paths) not in (3, 5):
        raise SystemExit("provide exactly three or five benchmark logs")
    if len(set(paths)) != len(paths):
        raise SystemExit("benchmark log paths must be distinct")
    runs = [_parse_log(path) for path in paths]
    _validate(runs)
    _print_summary(runs)
    _print_comparisons(runs)
    _print_variance_decision(runs)
    return 0


def _expected_keys() -> set[tuple[str, str, int, int]]:
    return {
        (case, workload, concurrency, output_tokens)
        for case in CASES
        for workload, concurrency, output_tokens in VARIANTS
    }


def _parse_log(path: Path) -> dict[tuple[str, str, int, int], Measurements]:
    if not path.is_file():
        raise SystemExit(f"benchmark log does not exist: {path}")
    results = {}
    for match in RESULT_PATTERN.finditer(path.read_text()):
        key = (
            match.group(1),
            match.group(2),
            int(match.group(3)),
            int(match.group(4)),
        )
        if key in results:
            raise SystemExit(f"duplicate result for {key} in {path}")
        results[key] = Measurements(*(int(match.group(i)) for i in range(5, 14)))
    expected = _expected_keys()
    if set(results) != expected:
        raise SystemExit(
            f"{path} has incomplete results; "
            f"missing={sorted(expected - set(results))}, "
            f"unexpected={sorted(set(results) - expected)}"
        )
    return results


def _validate(runs: list[dict[tuple[str, str, int, int], Measurements]]) -> None:
    expected_shapes = {
        key: (
            measurements.request_count,
            measurements.input_token_count,
            measurements.generated_token_count,
        )
        for key, measurements in runs[0].items()
    }
    for run_index, run in enumerate(runs, start=1):
        for key, expected_shape in expected_shapes.items():
            measurements = run[key]
            shape = (
                measurements.request_count,
                measurements.input_token_count,
                measurements.generated_token_count,
            )
            if shape != expected_shape:
                raise SystemExit(
                    f"run {run_index} has a different request/token shape for {key}"
                )
        for workload, concurrency, output_tokens in VARIANTS:
            shapes = {
                (
                    run[(case, workload, concurrency, output_tokens)].request_count,
                    run[(case, workload, concurrency, output_tokens)].input_token_count,
                    run[(case, workload, concurrency, output_tokens)].generated_token_count,
                )
                for case in CASES
            }
            if len(shapes) != 1:
                raise SystemExit(
                    f"run {run_index} cases differ for "
                    f"{workload}/c{concurrency}/o{output_tokens}"
                )
            if workload == "identical":
                for publication in PUBLICATION_POLICIES:
                    independent = run[
                        (
                            f"{publication}_independent",
                            workload,
                            concurrency,
                            output_tokens,
                        )
                    ]
                    coordinated = run[
                        (
                            f"{publication}_single_flight",
                            workload,
                            concurrency,
                            output_tokens,
                        )
                    ]
                    if coordinated.restored_token_count < independent.restored_token_count:
                        raise SystemExit(
                            f"run {run_index} {publication} single-flight restored fewer "
                            f"tokens than independent execution for "
                            f"c{concurrency}/o{output_tokens}"
                        )
        for concurrency in (2, 4, 8):
            completion = run[("completion_independent", "inflight", concurrency, 128)]
            incremental = run[("incremental_independent", "inflight", concurrency, 128)]
            if incremental.restored_token_count <= completion.restored_token_count:
                raise SystemExit(
                    f"run {run_index} incremental publication did not increase "
                    f"in-flight restored tokens for c{concurrency}"
                )


def _print_summary(
    runs: list[dict[tuple[str, str, int, int], Measurements]],
) -> None:
    print(f"Prefix-cache concurrent-population aggregate: {len(runs)} runs")
    rows = []
    for case in CASES:
        for workload, concurrency, output_tokens in VARIANTS:
            values = [
                run[(case, workload, concurrency, output_tokens)] for run in runs
            ]
            rows.append(
                (
                    case,
                    workload,
                    concurrency,
                    output_tokens,
                    _distribution([value.restored_token_count for value in values]),
                    _milliseconds([value.sum_prefill_us for value in values]),
                    _milliseconds([value.mean_ttft_us for value in values]),
                    _milliseconds([value.max_e2e_us for value in values]),
                    _milliseconds([value.wall_us for value in values]),
                )
            )
    print(
        tabulate(
            rows,
            headers=(
                "Case",
                "Inputs",
                "Concurrency",
                "Output",
                "Restored tokens",
                "Sum prefill ms",
                "Mean TTFT ms",
                "Max E2E ms",
                "Wall ms",
            ),
            tablefmt="simple",
        )
    )


def _print_comparisons(
    runs: list[dict[tuple[str, str, int, int], Measurements]],
) -> None:
    _print_policy_comparison(
        runs,
        "Single-flight reduction versus independent requests",
        tuple(
            (
                publication,
                f"{publication}_independent",
                f"{publication}_single_flight",
            )
            for publication in PUBLICATION_POLICIES
        ),
    )
    _print_policy_comparison(
        runs,
        "Incremental-publication reduction versus completion publication",
        tuple(
            (
                coordination,
                f"completion_{coordination}",
                f"incremental_{coordination}",
            )
            for coordination in COORDINATION_POLICIES
        ),
    )


def _print_policy_comparison(
    runs: list[dict[tuple[str, str, int, int], Measurements]],
    title: str,
    comparisons: tuple[tuple[str, str, str], ...],
) -> None:
    rows = []
    for comparison, baseline_case, changed_case in comparisons:
        for workload, concurrency, output_tokens in VARIANTS:
            changes = {"prefill": [], "ttft": [], "e2e": [], "wall": []}
            for run in runs:
                baseline = run[
                    (baseline_case, workload, concurrency, output_tokens)
                ]
                changed = run[(changed_case, workload, concurrency, output_tokens)]
                for name, field in (
                    ("prefill", "sum_prefill_us"),
                    ("ttft", "mean_ttft_us"),
                    ("e2e", "max_e2e_us"),
                    ("wall", "wall_us"),
                ):
                    changes[name].append(
                        1.0 - getattr(changed, field) / getattr(baseline, field)
                    )
            rows.append(
                (
                    comparison,
                    workload,
                    concurrency,
                    output_tokens,
                    _percent_distribution(changes["prefill"]),
                    _percent_distribution(changes["ttft"]),
                    _percent_distribution(changes["e2e"]),
                    _percent_distribution(changes["wall"]),
                )
            )
    print(f"\n{title}")
    print(
        tabulate(
            rows,
            headers=(
                "Held constant",
                "Inputs",
                "Concurrency",
                "Output",
                "Sum prefill",
                "Mean TTFT",
                "Max E2E",
                "Wall time",
            ),
            tablefmt="simple",
        )
    )


def _print_variance_decision(
    runs: list[dict[tuple[str, str, int, int], Measurements]],
) -> None:
    violations = []
    observed_race_variance = []
    for case in CASES:
        for workload, output_tokens in (
            ("identical", 16),
            ("identical", 128),
            ("inflight", 128),
        ):
            key = (case, workload, 8, output_tokens)
            for field in ("max_e2e_us", "wall_us"):
                values = [getattr(run[key], field) for run in runs]
                cv = stdev(values) / mean(values)
                if cv > MAXIMUM_CV:
                    if (case, workload) in RACE_SENSITIVE_CASES:
                        observed_race_variance.append((key, field, cv))
                    else:
                        violations.append((key, field, cv))
    print("\nVariance decision")
    if violations:
        print("Collect two additional runs; headline CV exceeded 5%:")
        for key, field, cv in violations:
            print(f"- {key} {field}: {cv:.1%}")
    else:
        print("Three runs are sufficient; every stable headline CV is at most 5%.")
    if observed_race_variance:
        print("Race-sensitive CVs above 5% (reported but not used to require more runs):")
        for key, field, cv in observed_race_variance:
            print(f"- {key} {field}: {cv:.1%}")


def _distribution(values: list[int | float]) -> str:
    return f"{mean(values):.1f} ± {stdev(values):.1f}"


def _milliseconds(values_microseconds: list[int]) -> str:
    return _distribution([value / 1000 for value in values_microseconds])


def _percent_distribution(values: list[float]) -> str:
    return f"{mean(values):.1%} ± {stdev(values):.1%}"


if __name__ == "__main__":
    raise SystemExit(main())
