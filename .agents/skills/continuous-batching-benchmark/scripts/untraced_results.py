#!/usr/bin/env python3
"""Aggregate repeated continuous-batching benchmark logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from statistics import stdev

from tabulate import tabulate


CASE_NAMES = (
    "Two requests / limit 1",
    "Two requests / limit 2",
    "Four requests / limit 1",
    "Four requests / limit 4",
)
CASE_PATTERN = "|".join(re.escape(case) for case in CASE_NAMES)
ROLE_NAMES = (
    "Anchor (short prefill)",
    "Follower (short prefill)",
    "Follower 1 (long prefill)",
    "Follower 2 (short prefill)",
    "Follower 3 (long prefill)",
)
ROLE_PATTERN = "|".join(re.escape(role) for role in ROLE_NAMES)
COMPARISONS = (
    (
        "Two-request handoff",
        "Two requests / limit 1",
        "Two requests / limit 2",
    ),
    (
        "Heterogeneous four-request workload",
        "Four requests / limit 1",
        "Four requests / limit 4",
    ),
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
REQUEST_ROW = re.compile(
    rf"^({CASE_PATTERN})\s+({ROLE_PATTERN})\s+(\d+)\s+(\d+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+(\d+)/(\d+)\s*$",
    re.MULTILINE,
)
SUMMARY_ROW = re.compile(
    rf"^({CASE_PATTERN})\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)%\s+\((\d+)/(\d+)\)\s*$",
    re.MULTILINE,
)
HEADLINE_MEASUREMENTS = (
    "follower_ttft_mean_ms",
    "anchor_e2e_seconds",
    "output_tokens_per_second",
)
MAXIMUM_CV = 0.05


@dataclass(frozen=True)
class RequestMeasurements:
    input_token_count: int
    output_token_count: int
    ttft_ms: float
    e2e_seconds: float
    draft_accepted_token_count: int
    draft_proposed_token_count: int

    @property
    def draft_acceptance_rate(self) -> float:
        return self.draft_accepted_token_count / self.draft_proposed_token_count


@dataclass(frozen=True)
class CaseMeasurements:
    follower_ttft_mean_ms: float
    follower_ttft_max_ms: float
    anchor_e2e_seconds: float
    wall_seconds: float
    output_tokens_per_second: float
    draft_acceptance_rate: float


@dataclass(frozen=True)
class RunMeasurements:
    requests: dict[tuple[str, str], RequestMeasurements]
    cases: dict[str, CaseMeasurements]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate untraced continuous-batching benchmark logs."
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
    _validate_consistent_requests(runs)
    _print_results(runs)
    return 0


def _parse_log(path: Path) -> RunMeasurements:
    if not path.is_file():
        raise SystemExit(f"benchmark log does not exist: {path}")
    text = ANSI_ESCAPE.sub("", path.read_text())

    requests: dict[tuple[str, str], RequestMeasurements] = {}
    for match in REQUEST_ROW.finditer(text):
        case, role = match.group(1), match.group(2)
        key = (case, role)
        if key in requests:
            raise SystemExit(f"duplicate request row for {case} / {role} in {path}")
        requests[key] = RequestMeasurements(
            input_token_count=int(match.group(3)),
            output_token_count=int(match.group(4)),
            ttft_ms=float(match.group(5)),
            e2e_seconds=float(match.group(6)),
            draft_accepted_token_count=int(match.group(7)),
            draft_proposed_token_count=int(match.group(8)),
        )

    cases: dict[str, CaseMeasurements] = {}
    for match in SUMMARY_ROW.finditer(text):
        case = match.group(1)
        if case in cases:
            raise SystemExit(f"duplicate summary row for {case} in {path}")
        accepted = int(match.group(8))
        proposed = int(match.group(9))
        cases[case] = CaseMeasurements(
            follower_ttft_mean_ms=float(match.group(2)),
            follower_ttft_max_ms=float(match.group(3)),
            anchor_e2e_seconds=float(match.group(4)),
            wall_seconds=float(match.group(5)),
            output_tokens_per_second=float(match.group(6)),
            draft_acceptance_rate=accepted / proposed,
        )

    expected_requests = {
        (case, role)
        for case in CASE_NAMES
        for role in _roles_for_case(case)
    }
    if set(requests) != expected_requests:
        missing = sorted(expected_requests - set(requests))
        raise SystemExit(f"{path} is missing request rows: {missing}")
    if set(cases) != set(CASE_NAMES):
        missing = sorted(set(CASE_NAMES) - set(cases))
        raise SystemExit(f"{path} is missing case summaries: {missing}")
    return RunMeasurements(requests=requests, cases=cases)


def _roles_for_case(case: str) -> tuple[str, ...]:
    if case.startswith("Two requests"):
        return ROLE_NAMES[:2]
    return (ROLE_NAMES[0], *ROLE_NAMES[2:])


def _validate_consistent_requests(runs: list[RunMeasurements]) -> None:
    expected = {
        key: (request.input_token_count, request.output_token_count)
        for key, request in runs[0].requests.items()
    }
    for index, run in enumerate(runs[1:], start=2):
        observed = {
            key: (request.input_token_count, request.output_token_count)
            for key, request in run.requests.items()
        }
        if observed != expected:
            raise SystemExit(f"run {index} has different request token counts")


def _print_results(runs: list[RunMeasurements]) -> None:
    print(f"Continuous-batching aggregate: {len(runs)} runs")
    _print_case_summary(runs)
    _print_comparisons(runs)
    _print_request_details(runs)
    _print_variance_decision(runs)


def _print_case_summary(runs: list[RunMeasurements]) -> None:
    rows = []
    for case in CASE_NAMES:
        values = [run.cases[case] for run in runs]
        rows.append(
            (
                case,
                _distribution([value.follower_ttft_mean_ms for value in values]),
                _distribution([value.follower_ttft_max_ms for value in values]),
                _distribution([value.anchor_e2e_seconds for value in values]),
                _distribution([value.wall_seconds for value in values]),
                _distribution(
                    [value.output_tokens_per_second for value in values]
                ),
                _percentage_distribution(
                    [value.draft_acceptance_rate for value in values]
                ),
            )
        )
    print("\nCase summary")
    print(
        tabulate(
            rows,
            headers=(
                "Case",
                "Follower mean TTFT ms",
                "Follower max TTFT ms",
                "Anchor E2E s",
                "Wall s",
                "Output tok/s",
                "Draft acceptance",
            ),
            tablefmt="simple",
        )
    )


def _print_comparisons(runs: list[RunMeasurements]) -> None:
    rows = []
    for workload, serial_case, batched_case in COMPARISONS:
        ttft_reductions = []
        throughput_speedups = []
        anchor_e2e_changes = []
        for run in runs:
            serial = run.cases[serial_case]
            batched = run.cases[batched_case]
            ttft_reductions.append(
                1.0
                - batched.follower_ttft_mean_ms / serial.follower_ttft_mean_ms
            )
            throughput_speedups.append(
                batched.output_tokens_per_second / serial.output_tokens_per_second
            )
            anchor_e2e_changes.append(
                batched.anchor_e2e_seconds / serial.anchor_e2e_seconds - 1.0
            )
        rows.append(
            (
                workload,
                _percentage_distribution(ttft_reductions),
                _distribution(throughput_speedups, suffix="x"),
                _percentage_distribution(anchor_e2e_changes, signed=True),
            )
        )
    print("\nSerial-to-batched comparison")
    print(
        tabulate(
            rows,
            headers=(
                "Workload",
                "Follower TTFT reduction",
                "Output throughput speedup",
                "Anchor E2E change",
            ),
            tablefmt="simple",
        )
    )


def _print_request_details(runs: list[RunMeasurements]) -> None:
    rows = []
    for case in CASE_NAMES:
        for role in _roles_for_case(case):
            values = [run.requests[(case, role)] for run in runs]
            first = values[0]
            rows.append(
                (
                    case,
                    role,
                    str(first.input_token_count),
                    str(first.output_token_count),
                    _distribution([value.ttft_ms for value in values]),
                    _distribution([value.e2e_seconds for value in values]),
                    _percentage_distribution(
                        [value.draft_acceptance_rate for value in values]
                    ),
                )
            )
    print("\nPer-request details")
    print(
        tabulate(
            rows,
            headers=(
                "Case",
                "Role",
                "Input",
                "Output",
                "TTFT ms",
                "E2E s",
                "Draft acceptance",
            ),
            tablefmt="simple",
        )
    )


def _print_variance_decision(runs: list[RunMeasurements]) -> None:
    coefficients = []
    for case in CASE_NAMES:
        for measurement in HEADLINE_MEASUREMENTS:
            values = [getattr(run.cases[case], measurement) for run in runs]
            coefficients.append((stdev(values) / mean(values), case, measurement))
    maximum, case, measurement = max(coefficients)
    print(
        f"\nMaximum headline CV: {maximum * 100:.2f}% "
        f"({case}, {measurement})"
    )
    if maximum > MAXIMUM_CV and len(runs) == 3:
        print("Variance decision: collect two additional runs and analyze all five.")
    elif maximum > MAXIMUM_CV:
        print("Variance decision: five runs collected; report the observed variance.")
    else:
        print("Variance decision: three runs are sufficient.")


def _distribution(values: list[float], suffix: str = "") -> str:
    return f"{mean(values):.3f} ± {stdev(values):.3f}{suffix}"


def _percentage_distribution(values: list[float], *, signed: bool = False) -> str:
    mean_value = mean(values) * 100
    sign = "+" if signed and mean_value >= 0 else ""
    return f"{sign}{mean_value:.1f}% ± {stdev(values) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
