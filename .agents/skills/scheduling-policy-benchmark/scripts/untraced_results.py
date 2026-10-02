#!/usr/bin/env python3
"""Aggregate repeated scheduling-policy benchmark logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from statistics import stdev

from tabulate import tabulate


POLICIES = (
    "first-come-first-served",
    "shortest-prefill-first",
    "round-robin",
)
WORKLOAD_ROLES = {
    "head-of-line": (
        "Anchor",
        "Oldest long follower",
        "Short follower",
    ),
    "long-fairness": (
        "Anchor",
        "Oldest long follower",
        "Long peer 1",
        "Long peer 2",
    ),
}
POLICY_PATTERN = "|".join(re.escape(policy) for policy in POLICIES)
WORKLOAD_PATTERN = "|".join(re.escape(workload) for workload in WORKLOAD_ROLES)
ROLE_PATTERN = "|".join(
    re.escape(role)
    for roles in WORKLOAD_ROLES.values()
    for role in roles
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
REQUEST_ROW = re.compile(
    rf"^({WORKLOAD_PATTERN})\s+({POLICY_PATTERN})\s+({ROLE_PATTERN})\s+"
    r"(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
    re.MULTILINE,
)
SUMMARY_ROW = re.compile(
    rf"^({WORKLOAD_PATTERN})\s+({POLICY_PATTERN})\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
    re.MULTILINE,
)
HEADLINE_MEASUREMENTS = {
    "head-of-line": (
        "oldest_follower_ttft_ms",
        "later_follower_mean_ttft_ms",
        "follower_max_ttft_ms",
        "anchor_e2e_seconds",
        "output_tokens_per_second",
    ),
    "long-fairness": (
        "oldest_follower_ttft_ms",
        "later_follower_mean_ttft_ms",
        "follower_max_ttft_ms",
        "follower_spread_ms",
        "anchor_e2e_seconds",
        "output_tokens_per_second",
    ),
}
MAXIMUM_CV = 0.05


@dataclass(frozen=True)
class RequestMeasurements:
    input_token_count: int
    output_token_count: int
    ttft_ms: float
    e2e_seconds: float


@dataclass(frozen=True)
class WorkloadMeasurements:
    oldest_follower_ttft_ms: float
    later_follower_mean_ttft_ms: float
    later_follower_max_ttft_ms: float
    follower_max_ttft_ms: float
    follower_spread_ms: float
    anchor_e2e_seconds: float
    wall_seconds: float
    output_tokens_per_second: float


@dataclass(frozen=True)
class RunMeasurements:
    requests: dict[tuple[str, str, str], RequestMeasurements]
    workloads: dict[tuple[str, str], WorkloadMeasurements]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate untraced scheduling-policy benchmark logs."
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

    requests: dict[tuple[str, str, str], RequestMeasurements] = {}
    for match in REQUEST_ROW.finditer(text):
        key = (match.group(1), match.group(2), match.group(3))
        if key in requests:
            raise SystemExit(f"duplicate request row for {key} in {path}")
        requests[key] = RequestMeasurements(
            input_token_count=int(match.group(4)),
            output_token_count=int(match.group(5)),
            ttft_ms=float(match.group(6)),
            e2e_seconds=float(match.group(7)),
        )

    workloads: dict[tuple[str, str], WorkloadMeasurements] = {}
    for match in SUMMARY_ROW.finditer(text):
        key = (match.group(1), match.group(2))
        if key in workloads:
            raise SystemExit(f"duplicate workload row for {key} in {path}")
        workloads[key] = WorkloadMeasurements(
            oldest_follower_ttft_ms=float(match.group(3)),
            later_follower_mean_ttft_ms=float(match.group(4)),
            later_follower_max_ttft_ms=float(match.group(5)),
            follower_max_ttft_ms=float(match.group(6)),
            follower_spread_ms=float(match.group(7)),
            anchor_e2e_seconds=float(match.group(8)),
            wall_seconds=float(match.group(9)),
            output_tokens_per_second=float(match.group(10)),
        )

    expected_requests = {
        (workload, policy, role)
        for workload, roles in WORKLOAD_ROLES.items()
        for policy in POLICIES
        for role in roles
    }
    if set(requests) != expected_requests:
        missing = sorted(expected_requests - set(requests))
        raise SystemExit(f"{path} is missing request rows: {missing}")
    expected_workloads = {
        (workload, policy) for workload in WORKLOAD_ROLES for policy in POLICIES
    }
    if set(workloads) != expected_workloads:
        missing = sorted(expected_workloads - set(workloads))
        raise SystemExit(f"{path} is missing workload summaries: {missing}")
    return RunMeasurements(requests=requests, workloads=workloads)


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
    print(f"Scheduling-policy aggregate: {len(runs)} runs")
    _print_workload_summary(runs)
    _print_fcfs_comparisons(runs)
    _print_request_details(runs)
    _print_variance_decision(runs)


def _print_workload_summary(runs: list[RunMeasurements]) -> None:
    rows = []
    for workload in WORKLOAD_ROLES:
        for policy in POLICIES:
            values = [run.workloads[(workload, policy)] for run in runs]
            rows.append(
                (
                    workload,
                    policy,
                    _distribution(
                        [value.oldest_follower_ttft_ms for value in values]
                    ),
                    _distribution(
                        [value.later_follower_mean_ttft_ms for value in values]
                    ),
                    _distribution(
                        [value.later_follower_max_ttft_ms for value in values]
                    ),
                    _distribution([value.follower_max_ttft_ms for value in values]),
                    _distribution([value.follower_spread_ms for value in values]),
                    _distribution([value.anchor_e2e_seconds for value in values]),
                    _distribution([value.wall_seconds for value in values]),
                    _distribution(
                        [value.output_tokens_per_second for value in values]
                    ),
                )
            )
    print("\nWorkload summary")
    print(
        tabulate(
            rows,
            headers=(
                "Workload",
                "Policy",
                "Oldest TTFT ms",
                "Later mean TTFT ms",
                "Later max TTFT ms",
                "Follower max TTFT ms",
                "Follower spread ms",
                "Anchor E2E s",
                "Wall s",
                "Output tok/s",
            ),
            tablefmt="simple",
        )
    )


def _print_fcfs_comparisons(runs: list[RunMeasurements]) -> None:
    rows = []
    for workload in WORKLOAD_ROLES:
        for policy in POLICIES[1:]:
            measurements = {
                name: []
                for name in (
                    "oldest_follower_ttft_ms",
                    "later_follower_mean_ttft_ms",
                    "follower_max_ttft_ms",
                    "anchor_e2e_seconds",
                    "output_tokens_per_second",
                )
            }
            for run in runs:
                baseline = run.workloads[(workload, POLICIES[0])]
                candidate = run.workloads[(workload, policy)]
                for name, changes in measurements.items():
                    changes.append(
                        getattr(candidate, name) / getattr(baseline, name) - 1.0
                    )
            rows.append(
                (
                    workload,
                    policy,
                    *(
                        _percentage_distribution(changes)
                        for changes in measurements.values()
                    ),
                )
            )
    print("\nChange from FCFS")
    print(
        tabulate(
            rows,
            headers=(
                "Workload",
                "Policy",
                "Oldest TTFT",
                "Later mean TTFT",
                "Follower max TTFT",
                "Anchor E2E",
                "Output throughput",
            ),
            tablefmt="simple",
        )
    )


def _print_request_details(runs: list[RunMeasurements]) -> None:
    rows = []
    for workload, roles in WORKLOAD_ROLES.items():
        for policy in POLICIES:
            for role in roles:
                values = [run.requests[(workload, policy, role)] for run in runs]
                rows.append(
                    (
                        workload,
                        policy,
                        role,
                        values[0].input_token_count,
                        values[0].output_token_count,
                        _distribution([value.ttft_ms for value in values]),
                        _distribution([value.e2e_seconds for value in values]),
                    )
                )
    print("\nPer-request details")
    print(
        tabulate(
            rows,
            headers=(
                "Workload",
                "Policy",
                "Role",
                "Input",
                "Output",
                "TTFT ms",
                "E2E s",
            ),
            tablefmt="simple",
        )
    )


def _print_variance_decision(runs: list[RunMeasurements]) -> None:
    coefficients = []
    for workload in WORKLOAD_ROLES:
        for policy in POLICIES:
            for measurement in HEADLINE_MEASUREMENTS[workload]:
                values = [
                    getattr(run.workloads[(workload, policy)], measurement)
                    for run in runs
                ]
                coefficients.append(
                    (stdev(values) / mean(values), workload, policy, measurement)
                )
    maximum, workload, policy, measurement = max(coefficients)
    print(
        f"\nMaximum headline CV: {maximum * 100:.2f}% "
        f"({workload} / {policy} / {measurement})"
    )
    if maximum > MAXIMUM_CV and len(runs) == 3:
        print("Collect two more complete runs and analyze all five logs.")
    elif maximum > MAXIMUM_CV:
        print("Five-run variance remains above 5%; report it with the results.")
    else:
        print("Headline variance is at or below 5%; no extra runs are required.")


def _distribution(values: list[float]) -> str:
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def _percentage_distribution(values: list[float]) -> str:
    return f"{mean(values) * 100:+.1f}% ± {stdev(values) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
