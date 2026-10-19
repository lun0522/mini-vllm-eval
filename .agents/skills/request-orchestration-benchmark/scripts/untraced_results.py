#!/usr/bin/env python3
"""Aggregate repeated request-orchestration benchmark logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from statistics import stdev

from tabulate import tabulate


CASE_METADATA = {
    "policy_round_robin": ("round-robin", "not-used"),
    "policy_least_requests": ("least-requests", "not-used"),
    "policy_least_estimated_work_detailed": (
        "least-estimated-work",
        "detailed",
    ),
    "policy_most_cached_prefix": ("most-cached-prefix", "not-used"),
    "policy_cache_aware_least_estimated_work_detailed": (
        "cache-aware-least-estimated-work",
        "detailed",
    ),
    "profile_least_estimated_work_fallback": (
        "least-estimated-work",
        "fallback",
    ),
    "profile_cache_aware_least_estimated_work_fallback": (
        "cache-aware-least-estimated-work",
        "fallback",
    ),
}
POLICY_CASES = (
    "policy_round_robin",
    "policy_least_requests",
    "policy_least_estimated_work_detailed",
    "policy_most_cached_prefix",
    "policy_cache_aware_least_estimated_work_detailed",
)
PROFILE_PAIRS = (
    (
        "least-estimated-work",
        "profile_least_estimated_work_fallback",
        "policy_least_estimated_work_detailed",
    ),
    (
        "cache-aware-least-estimated-work",
        "profile_cache_aware_least_estimated_work_fallback",
        "policy_cache_aware_least_estimated_work_detailed",
    ),
)
REQUEST_OUTPUT_TOKEN_COUNTS = {
    "short-shared": 16,
    "short-unique": 16,
    "decode-heavy-shared": 512,
    "decode-heavy-unique": 512,
    "medium-shared": 128,
    "medium-unique": 128,
    "prefill-heavy-shared": 16,
    "prefill-heavy-unique": 16,
    "long-shared": 1024,
    "long-unique": 1024,
}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SUMMARY_ROW = re.compile(
    r"REQUEST_ORCHESTRATION_RESULT "
    r"case=(\S+) policy=(\S+) profiles=(\S+) "
    r"backend_requests=cpu:(\d+),gpu:(\d+) wall_seconds=([0-9.]+) "
    r"output_tokens_per_second=([0-9.]+) median_ttft_ms=([0-9.]+) "
    r"p95_ttft_ms=([0-9.]+) median_e2e_seconds=([0-9.]+) "
    r"p95_e2e_seconds=([0-9.]+) shared_restored_tokens=(\d+) "
    r"unique_restored_tokens=(\d+)"
)
REQUEST_ROW = re.compile(
    r"REQUEST_ORCHESTRATION_REQUEST "
    r"case=(\S+) policy=(\S+) profiles=(\S+) request=(\S+) kind=(\S+) "
    r"input_tokens=(\d+) output_tokens=(\d+) backend_id=(\d+) "
    r"restored_tokens=(\d+) prefill_us=(\d+) decode_us=(\d+) "
    r"ttft_us=(\d+) e2e_us=(\d+)"
)
HEADLINE_MEASUREMENTS = (
    "wall_seconds",
    "output_tokens_per_second",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_e2e_seconds",
    "p95_e2e_seconds",
)
MAXIMUM_CV = 0.05


@dataclass(frozen=True)
class SummaryMeasurements:
    cpu_request_count: int
    gpu_request_count: int
    wall_seconds: float
    output_tokens_per_second: float
    median_ttft_ms: float
    p95_ttft_ms: float
    median_e2e_seconds: float
    p95_e2e_seconds: float
    shared_restored_token_count: int
    unique_restored_token_count: int


@dataclass(frozen=True)
class RequestMeasurements:
    kind: str
    input_token_count: int
    output_token_count: int
    backend_id: int
    restored_token_count: int
    prefill_microseconds: int
    decode_microseconds: int
    ttft_microseconds: int
    e2e_microseconds: int


@dataclass(frozen=True)
class RunMeasurements:
    summaries: dict[str, SummaryMeasurements]
    requests: dict[tuple[str, str], RequestMeasurements]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate untraced request-orchestration benchmark logs."
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

    summaries: dict[str, SummaryMeasurements] = {}
    for match in SUMMARY_ROW.finditer(text):
        case_name, policy, profiles = match.group(1, 2, 3)
        _validate_case_metadata(path, case_name, policy, profiles)
        if case_name in summaries:
            raise SystemExit(f"duplicate summary for {case_name} in {path}")
        summary = SummaryMeasurements(
            cpu_request_count=int(match.group(4)),
            gpu_request_count=int(match.group(5)),
            wall_seconds=float(match.group(6)),
            output_tokens_per_second=float(match.group(7)),
            median_ttft_ms=float(match.group(8)),
            p95_ttft_ms=float(match.group(9)),
            median_e2e_seconds=float(match.group(10)),
            p95_e2e_seconds=float(match.group(11)),
            shared_restored_token_count=int(match.group(12)),
            unique_restored_token_count=int(match.group(13)),
        )
        if summary.cpu_request_count + summary.gpu_request_count != len(
            REQUEST_OUTPUT_TOKEN_COUNTS
        ):
            raise SystemExit(f"{case_name} in {path} has an invalid request count")
        summaries[case_name] = summary

    requests: dict[tuple[str, str], RequestMeasurements] = {}
    for match in REQUEST_ROW.finditer(text):
        case_name, policy, profiles, request_name, kind = match.group(1, 2, 3, 4, 5)
        _validate_case_metadata(path, case_name, policy, profiles)
        key = (case_name, request_name)
        if key in requests:
            raise SystemExit(f"duplicate request row for {key} in {path}")
        if request_name not in REQUEST_OUTPUT_TOKEN_COUNTS:
            raise SystemExit(f"unexpected request {request_name} in {path}")
        expected_kind = "shared" if request_name.endswith("-shared") else "unique"
        if kind != expected_kind:
            raise SystemExit(f"{key} in {path} has kind {kind}, expected {expected_kind}")
        request = RequestMeasurements(
            kind=kind,
            input_token_count=int(match.group(6)),
            output_token_count=int(match.group(7)),
            backend_id=int(match.group(8)),
            restored_token_count=int(match.group(9)),
            prefill_microseconds=int(match.group(10)),
            decode_microseconds=int(match.group(11)),
            ttft_microseconds=int(match.group(12)),
            e2e_microseconds=int(match.group(13)),
        )
        if request.output_token_count != REQUEST_OUTPUT_TOKEN_COUNTS[request_name]:
            raise SystemExit(f"{key} in {path} has an invalid output token count")
        if request.backend_id not in (1, 2):
            raise SystemExit(f"{key} in {path} has invalid backend {request.backend_id}")
        requests[key] = request

    if set(summaries) != set(CASE_METADATA):
        missing = sorted(set(CASE_METADATA) - set(summaries))
        raise SystemExit(f"{path} is missing case summaries: {missing}")
    expected_requests = {
        (case_name, request_name)
        for case_name in CASE_METADATA
        for request_name in REQUEST_OUTPUT_TOKEN_COUNTS
    }
    if set(requests) != expected_requests:
        missing = sorted(expected_requests - set(requests))
        raise SystemExit(f"{path} is missing request rows: {missing}")
    _validate_backend_counts(path, summaries, requests)
    return RunMeasurements(summaries=summaries, requests=requests)


def _validate_case_metadata(
    path: Path,
    case_name: str,
    policy: str,
    profiles: str,
) -> None:
    expected = CASE_METADATA.get(case_name)
    if expected is None:
        raise SystemExit(f"unexpected case {case_name} in {path}")
    if (policy, profiles) != expected:
        raise SystemExit(
            f"{case_name} in {path} has policy/profile {(policy, profiles)}, "
            f"expected {expected}"
        )


def _validate_backend_counts(
    path: Path,
    summaries: dict[str, SummaryMeasurements],
    requests: dict[tuple[str, str], RequestMeasurements],
) -> None:
    for case_name, summary in summaries.items():
        cpu_count = sum(
            request.backend_id == 1
            for (request_case, _), request in requests.items()
            if request_case == case_name
        )
        gpu_count = len(REQUEST_OUTPUT_TOKEN_COUNTS) - cpu_count
        if (cpu_count, gpu_count) != (
            summary.cpu_request_count,
            summary.gpu_request_count,
        ):
            raise SystemExit(f"{case_name} in {path} has inconsistent backend counts")


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
    print(f"Request-orchestration aggregate: {len(runs)} runs")
    _print_summary(runs)
    _print_policy_comparison(runs)
    _print_profile_comparison(runs)
    _print_request_details(runs)
    _print_variance_decision(runs)


def _print_summary(runs: list[RunMeasurements]) -> None:
    rows = []
    for case_name in CASE_METADATA:
        policy, profiles = CASE_METADATA[case_name]
        values = [run.summaries[case_name] for run in runs]
        rows.append(
            (
                policy,
                profiles,
                _percentage_distribution(
                    [
                        value.cpu_request_count
                        / (value.cpu_request_count + value.gpu_request_count)
                        for value in values
                    ],
                    signed=False,
                ),
                _measurement_distribution(values, "wall_seconds"),
                _measurement_distribution(values, "output_tokens_per_second"),
                _measurement_distribution(values, "median_ttft_ms"),
                _measurement_distribution(values, "p95_ttft_ms"),
                _measurement_distribution(values, "median_e2e_seconds"),
                _measurement_distribution(values, "p95_e2e_seconds"),
                _measurement_distribution(values, "shared_restored_token_count"),
                _measurement_distribution(values, "unique_restored_token_count"),
            )
        )
    print("\nCase summary")
    print(
        tabulate(
            rows,
            headers=(
                "Policy",
                "Profiles",
                "CPU requests",
                "Wall s",
                "Output tok/s",
                "Median TTFT ms",
                "p95 TTFT ms",
                "Median E2E s",
                "p95 E2E s",
                "Shared restored",
                "Unique restored",
            ),
            tablefmt="simple",
        )
    )


def _print_policy_comparison(runs: list[RunMeasurements]) -> None:
    rows = []
    baseline_case = POLICY_CASES[0]
    for case_name in POLICY_CASES[1:]:
        policy, profiles = CASE_METADATA[case_name]
        rows.append(
            (
                policy,
                profiles,
                *(
                    _paired_change_distribution(
                        runs,
                        case_name,
                        baseline_case,
                        measurement,
                    )
                    for measurement in HEADLINE_MEASUREMENTS
                ),
            )
        )
    print("\nChange from round robin")
    print(
        tabulate(
            rows,
            headers=(
                "Policy",
                "Profiles",
                "Wall",
                "Output throughput",
                "Median TTFT",
                "p95 TTFT",
                "Median E2E",
                "p95 E2E",
            ),
            tablefmt="simple",
        )
    )


def _print_profile_comparison(runs: list[RunMeasurements]) -> None:
    rows = []
    for policy, fallback_case, detailed_case in PROFILE_PAIRS:
        rows.append(
            (
                policy,
                *(
                    _paired_change_distribution(
                        runs,
                        detailed_case,
                        fallback_case,
                        measurement,
                    )
                    for measurement in HEADLINE_MEASUREMENTS
                ),
            )
        )
    print("\nDetailed-profile change from fallback")
    print(
        tabulate(
            rows,
            headers=(
                "Policy",
                "Wall",
                "Output throughput",
                "Median TTFT",
                "p95 TTFT",
                "Median E2E",
                "p95 E2E",
            ),
            tablefmt="simple",
        )
    )


def _print_request_details(runs: list[RunMeasurements]) -> None:
    rows = []
    for case_name, (policy, profiles) in CASE_METADATA.items():
        for request_name in REQUEST_OUTPUT_TOKEN_COUNTS:
            values = [run.requests[(case_name, request_name)] for run in runs]
            rows.append(
                (
                    policy,
                    profiles,
                    request_name,
                    values[0].input_token_count,
                    values[0].output_token_count,
                    f"{sum(value.backend_id == 1 for value in values)}/{len(values)}",
                    _distribution([value.restored_token_count for value in values]),
                    _distribution([value.ttft_microseconds / 1_000 for value in values]),
                    _distribution([value.e2e_microseconds / 1_000_000 for value in values]),
                )
            )
    print("\nPer-request details")
    print(
        tabulate(
            rows,
            headers=(
                "Policy",
                "Profiles",
                "Request",
                "Input",
                "Output",
                "CPU runs",
                "Restored tokens",
                "TTFT ms",
                "E2E s",
            ),
            tablefmt="simple",
        )
    )


def _print_variance_decision(runs: list[RunMeasurements]) -> None:
    coefficients = []
    for case_name in CASE_METADATA:
        for measurement in HEADLINE_MEASUREMENTS:
            values = [getattr(run.summaries[case_name], measurement) for run in runs]
            coefficients.append(
                (stdev(values) / mean(values), case_name, measurement)
            )
    maximum, case_name, measurement = max(coefficients)
    unstable_routes = sum(
        len({run.requests[key].backend_id for run in runs}) > 1
        for key in runs[0].requests
    )
    print(
        f"\nMaximum headline CV: {maximum * 100:.2f}% "
        f"({case_name} / {measurement})"
    )
    print(
        f"Backend selection changed across runs for {unstable_routes}/"
        f"{len(runs[0].requests)} case/request pairs."
    )
    if maximum > MAXIMUM_CV and len(runs) == 3:
        print("Collect two more complete runs and analyze all five logs.")
    elif maximum > MAXIMUM_CV:
        print("Five-run variance remains above 5%; report it with the results.")
    else:
        print("Headline variance is at or below 5%; no extra runs are required.")


def _measurement_distribution(
    values: list[SummaryMeasurements],
    measurement: str,
) -> str:
    return _distribution([float(getattr(value, measurement)) for value in values])


def _paired_change_distribution(
    runs: list[RunMeasurements],
    candidate_case: str,
    baseline_case: str,
    measurement: str,
) -> str:
    changes = [
        getattr(run.summaries[candidate_case], measurement)
        / getattr(run.summaries[baseline_case], measurement)
        - 1.0
        for run in runs
    ]
    return _percentage_distribution(changes)


def _distribution(values: list[float | int]) -> str:
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def _percentage_distribution(values: list[float], signed: bool = True) -> str:
    sign = "+" if signed else ""
    return f"{mean(values) * 100:{sign}.1f}% ± {stdev(values) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
