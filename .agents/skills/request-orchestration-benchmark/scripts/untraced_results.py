#!/usr/bin/env python3
"""Aggregate repeated request-orchestration benchmark logs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from statistics import stdev

from tabulate import tabulate


# Use the benchmark's actual manifest rather than maintaining a second workload.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.request_orchestration import CASES, CASES_BY_NAME

CASE_METADATA = {case.name: (case.policy, case.profile_mode) for case in CASES}
REQUEST_DEFINITIONS = {
    case.name: {request.label: request for request in case.requests} for case in CASES
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
    r"ttft_us=(\d+) e2e_us=(\d+) request_id=(\d+) submitted_ms=([0-9.]+)"
)
REPLAY_ROW = re.compile(
    r"REQUEST_ORCHESTRATION_REPLAY case=(\S+) version=2 interval_ms=(\d+) "
    r"preprocessing_threads=(\d+) order=(\S+)"
)
SEED_ROW = re.compile(
    r"REQUEST_ORCHESTRATION_SEED case=(\S+) request_id=(\d+) backend_id=(\d+) "
    r"input_tokens=(\d+) indexed_tokens=(\d+)"
)
ROUTING_ROW = re.compile(
    r"Request routing: request_id=(\d+) routing_epoch=(\d+) input_tokens=\d+ "
    r"max_output_tokens=\d+ backend_id=(\d+)"
)
CANDIDATE_ROW = re.compile(
    r"Request routing candidate: request_id=(\d+) routing_epoch=(\d+) "
    r"selected=(true|false) backend_id=(\d+) ([^\n]*)"
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
    request_id: int
    submitted_ms: float


@dataclass(frozen=True)
class EstimatedRoutingCost:
    selected_duration_ns: int
    alternative_duration_ns: int


@dataclass(frozen=True)
class RunMeasurements:
    summaries: dict[str, SummaryMeasurements]
    requests: dict[tuple[str, str], RequestMeasurements]
    estimated_costs: dict[tuple[str, str], EstimatedRoutingCost]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate untraced request-orchestration benchmark logs."
    )
    parser.add_argument(
        "logs",
        type=Path,
        nargs="+",
        help="one diagnostic log, or three or five complete performance logs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_paths = [path.expanduser().resolve() for path in args.logs]
    if len(log_paths) not in (1, 3, 5):
        raise SystemExit("provide one diagnostic log, or three or five benchmark logs")
    if len(set(log_paths)) != len(log_paths):
        raise SystemExit("benchmark log paths must be distinct")
    runs = [_parse_log(path) for path in log_paths]
    _validate_consistent_requests(runs)
    print("Verified 50 ms replay with one preprocessing worker and matching routing order.")
    if len(runs) == 1:
        print(f"Validated {len(CASES)} cases and {len(runs[0].requests)} measured requests; "
              "diagnostic only, no aggregate.")
        _print_diagnostic(runs[0])
        _print_estimated_margins(runs)
        return 0
    _print_results(runs)
    return 0


def _parse_log(path: Path) -> RunMeasurements:
    if not path.is_file():
        raise SystemExit(f"benchmark log does not exist: {path}")
    text = ANSI_ESCAPE.sub("", path.read_text())
    if not REPLAY_ROW.search(text):
        raise SystemExit(
            f"{path} lacks version-2 replay metadata; legacy ten-request logs "
            "are not comparable with this benchmark"
        )

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
            REQUEST_DEFINITIONS[case_name]
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
        if request_name not in REQUEST_DEFINITIONS[case_name]:
            raise SystemExit(f"unexpected request {request_name} in {path}")
        definition = REQUEST_DEFINITIONS[case_name][request_name]
        expected_kind = definition.kind
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
            request_id=int(match.group(14)),
            submitted_ms=float(match.group(15)),
        )
        if request.output_token_count != definition.shape.max_output_token_count:
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
        for request_name in REQUEST_DEFINITIONS[case_name]
    }
    if set(requests) != expected_requests:
        missing = sorted(expected_requests - set(requests))
        raise SystemExit(f"{path} is missing request rows: {missing}")
    _validate_backend_counts(path, summaries, requests)
    estimated_costs = _validate_routing_order(path, text, requests)
    return RunMeasurements(summaries=summaries, requests=requests, estimated_costs=estimated_costs)


def _validate_routing_order(
    path: Path,
    text: str,
    requests: dict[tuple[str, str], RequestMeasurements],
) -> dict[tuple[str, str], EstimatedRoutingCost]:
    estimated_costs = {}
    case_logs = re.split(r"Starting benchmark case: (\S+)", text)
    observed_cases = set()
    for case_name, case_log in zip(case_logs[1::2], case_logs[2::2]):
        if case_name not in CASE_METADATA or case_name in observed_cases:
            raise SystemExit(f"unexpected or duplicate case {case_name} in {path}")
        observed_cases.add(case_name)
        definition = CASES_BY_NAME[case_name]
        expected_order = tuple(REQUEST_DEFINITIONS[case_name])
        replay_rows = list(REPLAY_ROW.finditer(case_log))
        expected_replay = (case_name, "50", "1", ",".join(expected_order))
        if len(replay_rows) != 1 or replay_rows[0].groups() != expected_replay:
            raise SystemExit(
                f"{case_name} in {path} lacks the expected 50 ms/single-worker replay; "
                "legacy ten-request logs are not comparable"
            )

        labels_by_id = {}
        for label in expected_order:
            request = requests[(case_name, label)]
            if request.request_id <= 0 or request.request_id in labels_by_id:
                raise SystemExit(
                    f"invalid or duplicate request ID for {case_name}/{label} in {path}"
                )
            labels_by_id[request.request_id] = label
        # Logs are emitted outside the routing lock; the epoch, not log line
        # order or request-ID order, records the actual selection order.
        routes = [tuple(map(int, match.groups())) for match in ROUTING_ROW.finditer(case_log)]
        routes.sort(key=lambda route: route[1])
        if "estimated-work" in definition.policy:
            candidates = {}
            for match in CANDIDATE_ROW.finditer(case_log):
                request_id, epoch, selected, backend_id, fields = match.groups()
                duration = re.search(r"\bestimated_duration_ns=(\d+)\b", fields)
                if duration is None:
                    raise SystemExit(f"{case_name} in {path} lacks estimated-duration diagnostics")
                key = (int(request_id), int(epoch), int(backend_id))
                if key in candidates:
                    raise SystemExit(f"duplicate estimated candidate {key} in {path}")
                candidates[key] = (selected == "true", int(duration.group(1)))
            expected_keys = {
                (request_id, epoch, backend_id)
                for request_id, epoch, _ in routes for backend_id in (1, 2)
            }
            if set(candidates) != expected_keys:
                raise SystemExit(f"{case_name} in {path} has incomplete estimated candidates")
            for request_id, epoch, backend_id in routes:
                selected, duration = candidates[(request_id, epoch, backend_id)]
                alternative_selected, alternative = candidates[(request_id, epoch, 3 - backend_id)]
                if not selected or alternative_selected or duration > alternative:
                    raise SystemExit(f"{case_name} in {path} has inconsistent estimated selection")
                label = labels_by_id.get(request_id)
                if label is not None:
                    estimated_costs[(case_name, label)] = EstimatedRoutingCost(duration, alternative)
        seeds = list(SEED_ROW.finditer(case_log))
        seed_count = int(definition.scenario == "affinity")
        if len(seeds) != seed_count:
            raise SystemExit(f"{case_name} in {path} has unexpected seed metadata")
        if seeds:
            seed_case, seed_id, seed_backend, seed_input, seed_indexed = seeds[0].groups()
            if (seed_case != case_name or not routes
                    or routes[0] != (int(seed_id), 1, int(seed_backend))
                    or int(seed_input) <= 0 or int(seed_indexed) <= 0):
                raise SystemExit(f"{case_name} in {path} has inconsistent seed metadata")
        if [epoch for _, epoch, _ in routes] != list(range(1, len(expected_order) + seed_count + 1)):
            raise SystemExit(
                f"{case_name} in {path} lacks unique, consecutive routing epochs "
                "for setup and measured requests; capture RUST_LOG=info"
            )
        if len({request_id for request_id, _, _ in routes}) != len(routes):
            raise SystemExit(f"duplicate routing request ID for {case_name} in {path}")
        observed_order = []
        for request_id, _, backend_id in routes:
            label = labels_by_id.get(request_id)
            if label is not None:
                if backend_id != requests[(case_name, label)].backend_id:
                    raise SystemExit(f"routing/backend mismatch for {case_name}/{label}")
                observed_order.append(label)
        if tuple(observed_order) != expected_order:
            raise SystemExit(
                f"routing order mismatch for {case_name} in {path}: "
                f"expected {expected_order}, observed {tuple(observed_order)}. "
                "Investigate the replay before aggregating performance samples."
            )
        offsets = [requests[(case_name, label)].submitted_ms for label in expected_order]
        if any(b < a for a, b in zip(offsets, offsets[1:])):
            raise SystemExit(f"{case_name} in {path} has nonmonotonic submission times")
        lateness = max(offset - index * 50 for index, offset in enumerate(offsets))
        if lateness > 25:
            print(f"WARNING: {case_name} client submission was up to {lateness:.1f} ms late")
    if observed_cases != set(CASE_METADATA):
        raise SystemExit(f"{path} is missing benchmark case boundaries for routing validation")
    return estimated_costs


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
        gpu_count = len(REQUEST_DEFINITIONS[case_name]) - cpu_count
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
    _print_estimated_margins(runs)
    _print_variance_decision(runs)


def _print_estimated_margins(runs: list[RunMeasurements]) -> None:
    rows = []
    for case_name in CASE_METADATA:
        costs = [(label, cost) for run in runs
                 for (case, label), cost in run.estimated_costs.items() if case == case_name]
        if not costs:
            continue
        label, cost = min(costs, key=lambda item:
                          item[1].alternative_duration_ns - item[1].selected_duration_ns)
        rows.append((case_name, label, f"{cost.selected_duration_ns / 1e9:.6f}",
                     f"{cost.alternative_duration_ns / 1e9:.6f}",
                     f"{(cost.alternative_duration_ns - cost.selected_duration_ns) / 1e9:.6f}"))
    print("\nClosest estimated-work decision per case (minimum margin across supplied runs)")
    print(tabulate(rows, headers=("Case", "Request", "Selected estimate s",
                                 "Alternative estimate s", "Margin s")))


def _print_diagnostic(run: RunMeasurements) -> None:
    rows = []
    for case_name in CASE_METADATA:
        summary = run.summaries[case_name]
        requests = [(label, request) for (case, label), request in run.requests.items()
                    if case == case_name]
        slowest_label, slowest = max(requests, key=lambda item: item[1].e2e_microseconds)
        for backend_id in (1, 2):
            work = [request for _, request in requests if request.backend_id == backend_id]
            rows.append((case_name, "CPU" if backend_id == 1 else "GPU", len(work),
                         sum(request.input_token_count for request in work),
                         sum(request.input_token_count - request.restored_token_count for request in work),
                         sum(request.output_token_count for request in work),
                         sum(request.restored_token_count for request in work),
                         f"{summary.wall_seconds:.2f}",
                         f"{slowest_label}/backend-{slowest.backend_id}"))
    print(tabulate(rows, headers=("Case", "Backend", "Requests", "Input", "Uncached input",
                                 "Output", "Restored", "Wall s", "Slowest request")))


def _print_summary(runs: list[RunMeasurements]) -> None:
    rows = []
    for case_name in CASE_METADATA:
        policy, profiles = CASE_METADATA[case_name]
        values = [run.summaries[case_name] for run in runs]
        rows.append(
            (
                case_name,
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
                "Case",
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
    for case in CASES:
        if not case.policy_comparison or case.policy == "round-robin":
            continue
        case_name = case.name
        baseline_case = f"policy_round_robin__{case.scenario}__{case.schedule}"
        policy, profiles = CASE_METADATA[case_name]
        rows.append(
            (
                case_name,
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
                "Case",
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
    for case in CASES:
        if case.profile_mode != "detailed":
            continue
        detailed_case = case.name
        fallback_case = detailed_case.replace("policy_", "profile_", 1).replace(
            "_detailed__", "_fallback__")
        rows.append(
            (
                detailed_case,
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
                "Case",
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
        for request_name in REQUEST_DEFINITIONS[case_name]:
            values = [run.requests[(case_name, request_name)] for run in runs]
            rows.append(
                (
                    case_name,
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
                "Case",
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
    if maximum > MAXIMUM_CV:
        print(
            "Headline variance exceeds 5%. Inspect per-request backend placement, "
            "cache restoration, and system conditions before deciding on more runs; "
            "do not automatically collect more samples or discard valid runs."
        )
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
