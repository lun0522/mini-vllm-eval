#!/usr/bin/env python3
"""Aggregate repeated speculative-sampling tradeoff benchmark logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from statistics import stdev

from tabulate import tabulate


TEMPERATURES = (0.2, 0.7, 1.0, 1.6)
TOTAL_VARIATION_BUDGETS = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
REPRESENTATIVE_TEMPERATURE = 0.7
SEED_COUNT = 3
MAXIMUM_CV = 0.05
TARGET_KEY = ("target_only", REPRESENTATIVE_TEMPERATURE, None)
RESULT_PATTERN = re.compile(
    r"SPECULATIVE_SAMPLING_TRADEOFF_RESULT "
    r"configuration=(target_only|fixed_4) "
    r"temperature=([0-9.]+) "
    r"total_variation_budget=(none|[0-9.]+) "
    r"seed_count=(\d+) "
    r"median_e2e_seconds=([0-9.]+) "
    r"minimum_e2e_seconds=([0-9.]+) "
    r"maximum_e2e_seconds=([0-9.]+) "
    r"median_decode_tokens_per_second=([0-9.]+) "
    r"accepted_token_count=(\d+) "
    r"proposed_token_count=(\d+) "
    r"accepted_tokens_per_round=(none|[0-9.]+) "
    r"output_tokens_per_round=(none|[0-9.]+)"
)
EXCERPT_PATTERN = re.compile(
    r"^TV budget ([0-9.]+): (.*)$",
    re.MULTILINE,
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class Measurements:
    median_e2e_seconds: float
    minimum_e2e_seconds: float
    maximum_e2e_seconds: float
    median_decode_tokens_per_second: float
    accepted_token_count: int
    proposed_token_count: int
    accepted_tokens_per_round: float | None
    output_tokens_per_round: float | None


@dataclass(frozen=True)
class RunMeasurements:
    results: dict[tuple[str, float, float | None], Measurements]
    excerpts: dict[float, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate speculative-sampling tradeoff benchmark logs."
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
    paths = [path.expanduser().resolve() for path in args.logs]
    if len(paths) not in (3, 5):
        raise SystemExit("provide exactly three or five benchmark logs")
    if len(set(paths)) != len(paths):
        raise SystemExit("benchmark log paths must be distinct")
    runs = [_parse_log(path) for path in paths]
    _print_results(runs)
    return 0


def _expected_keys() -> set[tuple[str, float, float | None]]:
    return {
        TARGET_KEY,
        *{
            ("fixed_4", temperature, 0.0)
            for temperature in TEMPERATURES
        },
        *{
            ("fixed_4", REPRESENTATIVE_TEMPERATURE, budget)
            for budget in TOTAL_VARIATION_BUDGETS
        },
    }


def _parse_log(path: Path) -> RunMeasurements:
    if not path.is_file():
        raise SystemExit(f"benchmark log does not exist: {path}")
    text = ANSI_ESCAPE.sub("", path.read_text())
    results: dict[tuple[str, float, float | None], Measurements] = {}
    for match in RESULT_PATTERN.finditer(text):
        configuration = match.group(1)
        temperature = float(match.group(2))
        budget = None if match.group(3) == "none" else float(match.group(3))
        key = (configuration, temperature, budget)
        if key in results:
            raise SystemExit(f"duplicate result for {key} in {path}")
        if int(match.group(4)) != SEED_COUNT:
            raise SystemExit(f"result {key} in {path} does not contain three seeds")
        accepted_per_round = _optional_float(match.group(11))
        output_per_round = _optional_float(match.group(12))
        measurements = Measurements(
            median_e2e_seconds=float(match.group(5)),
            minimum_e2e_seconds=float(match.group(6)),
            maximum_e2e_seconds=float(match.group(7)),
            median_decode_tokens_per_second=float(match.group(8)),
            accepted_token_count=int(match.group(9)),
            proposed_token_count=int(match.group(10)),
            accepted_tokens_per_round=accepted_per_round,
            output_tokens_per_round=output_per_round,
        )
        _validate_measurements(path, key, measurements)
        results[key] = measurements

    expected = _expected_keys()
    if set(results) != expected:
        missing = sorted(expected - set(results), key=str)
        unexpected = sorted(set(results) - expected, key=str)
        raise SystemExit(
            f"{path} has incomplete results; missing={missing}, "
            f"unexpected={unexpected}"
        )

    excerpts: dict[float, str] = {}
    for match in EXCERPT_PATTERN.finditer(text):
        budget = float(match.group(1))
        if budget in excerpts:
            raise SystemExit(f"duplicate excerpt for TV budget {budget:g} in {path}")
        excerpts[budget] = match.group(2).strip()
    if set(excerpts) != set(TOTAL_VARIATION_BUDGETS):
        missing = sorted(set(TOTAL_VARIATION_BUDGETS) - set(excerpts))
        raise SystemExit(f"{path} is missing excerpts for TV budgets: {missing}")
    return RunMeasurements(results, excerpts)


def _optional_float(value: str) -> float | None:
    return None if value == "none" else float(value)


def _validate_measurements(
    path: Path,
    key: tuple[str, float, float | None],
    measurements: Measurements,
) -> None:
    if not (
        0 < measurements.minimum_e2e_seconds
        <= measurements.median_e2e_seconds
        <= measurements.maximum_e2e_seconds
    ):
        raise SystemExit(f"result {key} in {path} has invalid E2E values")
    if measurements.median_decode_tokens_per_second <= 0:
        raise SystemExit(f"result {key} in {path} has invalid decode throughput")
    if not 0 <= measurements.accepted_token_count <= measurements.proposed_token_count:
        raise SystemExit(f"result {key} in {path} has invalid draft-token totals")
    is_target = key == TARGET_KEY
    round_values = (
        measurements.accepted_tokens_per_round,
        measurements.output_tokens_per_round,
    )
    if is_target:
        if measurements.proposed_token_count != 0 or round_values != (None, None):
            raise SystemExit(f"target-only result in {path} has draft metrics")
    else:
        accepted_per_round, output_per_round = round_values
        if (
            measurements.proposed_token_count <= 0
            or accepted_per_round is None
            or accepted_per_round < 0
            or output_per_round is None
            or output_per_round <= 0
        ):
            raise SystemExit(f"fixed-4 result {key} in {path} lacks draft metrics")
    if key == ("fixed_4", REPRESENTATIVE_TEMPERATURE, 1.0) and (
        measurements.accepted_token_count != measurements.proposed_token_count
    ):
        raise SystemExit(f"TV budget 1 result in {path} did not accept every proposal")


def _print_results(runs: list[RunMeasurements]) -> None:
    print(f"Speculative-sampling tradeoff aggregate: {len(runs)} runs")
    print("\nTemperature sweep (TV budget 0)")
    _print_table(runs, (TARGET_KEY, *(_temperature_key(t) for t in TEMPERATURES)))
    print(
        "\nTotal-variation budget sweep "
        f"(temperature {REPRESENTATIVE_TEMPERATURE:g})"
    )
    _print_table(runs, (TARGET_KEY, *(_budget_key(b) for b in TOTAL_VARIATION_BUDGETS)))
    _print_excerpts(runs)
    _print_variance_decision(runs)


def _temperature_key(temperature: float) -> tuple[str, float, float]:
    return ("fixed_4", temperature, 0.0)


def _budget_key(budget: float) -> tuple[str, float, float]:
    return ("fixed_4", REPRESENTATIVE_TEMPERATURE, budget)


def _print_table(
    runs: list[RunMeasurements],
    keys: tuple[tuple[str, float, float | None], ...],
) -> None:
    rows = []
    for key in keys:
        values = [run.results[key] for run in runs]
        accepted = sum(value.accepted_token_count for value in values)
        proposed = sum(value.proposed_token_count for value in values)
        rows.append(
            (
                "Target only" if key[0] == "target_only" else "Fixed 4",
                f"{key[1]:g}",
                "-" if key[2] is None else f"{key[2]:g}",
                _e2e_distribution(values),
                _distribution(
                    [value.median_decode_tokens_per_second for value in values]
                ),
                _speedup_distribution(runs, key),
                "-" if not proposed else f"{accepted / proposed:.1%}",
                "-" if not proposed else f"{accepted}/{proposed}",
                _optional_distribution(
                    [value.accepted_tokens_per_round for value in values]
                ),
                _optional_distribution(
                    [value.output_tokens_per_round for value in values]
                ),
            )
        )
    print(
        tabulate(
            rows,
            headers=(
                "Configuration",
                "Temperature",
                "TV budget",
                "Median E2E s [observed range]",
                "Median decode tok/s",
                "vs target",
                "Pooled acceptance",
                "Accepted/proposed",
                "Accepted/round",
                "Output/round",
            ),
            tablefmt="simple",
        )
    )


def _e2e_distribution(values: list[Measurements]) -> str:
    medians = [value.median_e2e_seconds for value in values]
    observed_minimum = min(value.minimum_e2e_seconds for value in values)
    observed_maximum = max(value.maximum_e2e_seconds for value in values)
    return (
        f"{_distribution(medians)} "
        f"[{observed_minimum:.3f}, {observed_maximum:.3f}]"
    )


def _speedup_distribution(
    runs: list[RunMeasurements],
    key: tuple[str, float, float | None],
) -> str:
    if key == TARGET_KEY:
        return "1.000x ± 0.000x"
    if key[1] != REPRESENTATIVE_TEMPERATURE:
        return "-"
    speedups = [
        run.results[TARGET_KEY].median_e2e_seconds
        / run.results[key].median_e2e_seconds
        for run in runs
    ]
    return _distribution(speedups, suffix="x")


def _optional_distribution(values: list[float | None]) -> str:
    if all(value is None for value in values):
        return "-"
    if any(value is None for value in values):
        raise RuntimeError("inconsistent optional measurements across runs")
    return _distribution([value for value in values if value is not None])


def _print_excerpts(runs: list[RunMeasurements]) -> None:
    print("\nRepresentative creative-writing excerpts (first run, seed 101)")
    for budget in TOTAL_VARIATION_BUDGETS:
        excerpts = [run.excerpts[budget] for run in runs]
        unique_count = len(set(excerpts))
        consistency = (
            f"identical across {len(runs)} runs"
            if unique_count == 1
            else f"{unique_count} variants across {len(runs)} runs"
        )
        print(f"\nTV budget {budget:g} ({consistency}): {excerpts[0]}")


def _print_variance_decision(runs: list[RunMeasurements]) -> None:
    coefficients = []
    for key in _expected_keys():
        for measurement in (
            "median_e2e_seconds",
            "median_decode_tokens_per_second",
        ):
            values = [getattr(run.results[key], measurement) for run in runs]
            coefficients.append((stdev(values) / mean(values), key, measurement))
    maximum, key, measurement = max(coefficients)
    print(
        f"\nMaximum headline CV: {maximum:.2%} "
        f"({_format_key(key)} / {measurement})"
    )
    if maximum > MAXIMUM_CV and len(runs) == 3:
        print("Collect two more complete runs and analyze all five logs.")
    elif maximum > MAXIMUM_CV:
        print("Five-run variance remains above 5%; report it with the results.")
    else:
        print("Headline variance is at or below 5%; no extra runs are required.")


def _format_key(key: tuple[str, float, float | None]) -> str:
    configuration = "target-only" if key[0] == "target_only" else "fixed-4"
    budget = "none" if key[2] is None else f"{key[2]:g}"
    return f"{configuration}, temperature={key[1]:g}, TV budget={budget}"


def _distribution(values: list[float], *, suffix: str = "") -> str:
    return f"{mean(values):.3f}{suffix} ± {stdev(values):.3f}{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())
