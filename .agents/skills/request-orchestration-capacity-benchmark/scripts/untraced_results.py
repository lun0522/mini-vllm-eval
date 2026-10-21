#!/usr/bin/env python3
"""Validate and aggregate request-orchestration capacity benchmark logs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from tabulate import tabulate


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.request_orchestration_capacity import CASES, CASES_BY_NAME, WORKLOADS


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SUMMARY_ROW = re.compile(
    r"REQUEST_ORCHESTRATION_CAPACITY_RESULT case=(\S+) "
    r"backend_requests=cpu:(\d+),gpu:(\d+) workload=(\S+) routing_mode=(\S+) "
    r"request_count=(\d+) wall_seconds=([0-9.]+) "
    r"output_tokens_per_second=([0-9.]+) median_ttft_ms=([0-9.]+) "
    r"p95_ttft_ms=([0-9.]+) median_e2e_seconds=([0-9.]+) "
    r"p95_e2e_seconds=([0-9.]+) restored_tokens=(\d+)"
)
REQUEST_ROW = re.compile(
    r"REQUEST_ORCHESTRATION_CAPACITY_REQUEST case=(\S+) request=(\S+) "
    r"input_tokens=(\d+) output_tokens=(\d+) backend_id=(\d+) "
    r"restored_tokens=(\d+) prefill_us=(\d+) decode_us=(\d+) "
    r"ttft_us=(\d+) e2e_us=(\d+) request_id=(\d+) submitted_ms=([0-9.]+)"
)
REPLAY_ROW = re.compile(
    r"REQUEST_ORCHESTRATION_CAPACITY_REPLAY case=(\S+) workload=(\S+) "
    r"routing_mode=(\S+) version=2 interval_ms=(\d+) "
    r"preprocessing_threads=(\d+) requests=(\d+) order=(\S+)"
)
SEED_ROW = re.compile(
    r"REQUEST_ORCHESTRATION_CAPACITY_SEED case=(\S+) requested_backend_id=(\d+) "
    r"actual_backend_id=(\d+) input_tokens=(\d+) indexed_tokens=(\d+)"
)
ROUTING_ROW = re.compile(
    r"Request routing: request_id=(\d+) routing_epoch=(\d+) input_tokens=\d+ "
    r"max_output_tokens=\d+ backend_id=(\d+)"
)
CANDIDATE_ROW = re.compile(
    r"Request routing candidate: request_id=(\d+) routing_epoch=(\d+) "
    r"selected=(true|false) backend_id=(\d+) ([^\n]*)"
)
HEADLINES = (
    "wall_seconds",
    "output_tokens_per_second",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_e2e_seconds",
    "p95_e2e_seconds",
)
MAXIMUM_CV = 0.05


@dataclass(frozen=True)
class Summary:
    cpu_requests: int
    gpu_requests: int
    wall_seconds: float
    output_tokens_per_second: float
    median_ttft_ms: float
    p95_ttft_ms: float
    median_e2e_seconds: float
    p95_e2e_seconds: float
    restored_tokens: int


@dataclass(frozen=True)
class Request:
    input_tokens: int
    output_tokens: int
    backend_id: int
    restored_tokens: int
    request_id: int
    submitted_ms: float


@dataclass(frozen=True)
class Run:
    summaries: dict[str, Summary]
    requests: dict[tuple[str, str], Request]
    closest_margins_ns: dict[str, int]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate request-orchestration capacity benchmark logs."
    )
    parser.add_argument("logs", type=Path, nargs="+")
    paths = [path.expanduser().resolve() for path in parser.parse_args().logs]
    if len(paths) not in (1, 3, 5):
        raise SystemExit("provide one diagnostic log, or three or five benchmark logs")
    if len(set(paths)) != len(paths):
        raise SystemExit("benchmark log paths must be distinct")
    runs = [_parse_log(path) for path in paths]
    _validate_across_runs(runs)
    if len(runs) == 1:
        print("Validated one diagnostic invocation; no performance aggregate.")
    else:
        print(f"Request-orchestration capacity aggregate: {len(runs)} runs")
    _print_summary(runs)
    _print_paired_changes(runs)
    _print_routing(runs)
    if len(runs) > 1:
        _print_variance(runs)
    return 0


def _parse_log(path: Path) -> Run:
    if not path.is_file():
        raise SystemExit(f"benchmark log does not exist: {path}")
    text = ANSI_ESCAPE.sub("", path.read_text())
    summaries = {}
    for match in SUMMARY_ROW.finditer(text):
        (case_name, cpu, gpu, workload, routing_mode, request_count, *values) = match.groups()
        definition = _definition(path, case_name, workload, routing_mode, request_count)
        if case_name in summaries:
            raise SystemExit(f"duplicate summary for {case_name} in {path}")
        summary = Summary(int(cpu), int(gpu), *(float(value) for value in values[:-1]), int(values[-1]))
        if summary.cpu_requests + summary.gpu_requests != definition.workload.request_count:
            raise SystemExit(f"invalid backend counts for {case_name} in {path}")
        summaries[case_name] = summary

    requests = {}
    for match in REQUEST_ROW.finditer(text):
        (case_name, label, input_tokens, output_tokens, backend_id, restored,
         _prefill, _decode, _ttft, _e2e, request_id, submitted_ms) = match.groups()
        key = (case_name, label)
        if case_name not in CASES_BY_NAME or key in requests:
            raise SystemExit(f"unexpected or duplicate request {key} in {path}")
        requests[key] = Request(
            int(input_tokens), int(output_tokens), int(backend_id), int(restored),
            int(request_id), float(submitted_ms)
        )

    if set(summaries) != set(CASES_BY_NAME):
        raise SystemExit(f"{path} does not contain every capacity case summary")
    _validate_requests(path, summaries, requests)
    margins = _validate_case_logs(path, text, requests)
    return Run(summaries, requests, margins)


def _definition(path: Path, case_name: str, workload: str, routing_mode: str, count: str):
    definition = CASES_BY_NAME.get(case_name)
    if definition is None or (
        definition.workload.name,
        definition.routing_mode,
        definition.workload.request_count,
    ) != (workload, routing_mode, int(count)):
        raise SystemExit(f"invalid metadata for {case_name} in {path}")
    return definition


def _validate_requests(path: Path, summaries: dict[str, Summary], requests: dict[tuple[str, str], Request]) -> None:
    input_counts = set()
    for definition in CASES:
        labels = tuple(f"{definition.workload.name}-{index + 1:02d}" for index in range(definition.workload.request_count))
        observed = [requests.get((definition.name, label)) for label in labels]
        if any(request is None for request in observed):
            raise SystemExit(f"{path} lacks measured requests for {definition.name}")
        typed = [request for request in observed if request is not None]
        input_counts.update(request.input_tokens for request in typed)
        if any(request.output_tokens != 128 or request.backend_id not in (1, 2)
               or request.restored_tokens <= 0 for request in typed):
            raise SystemExit(f"invalid request measurements for {definition.name} in {path}")
        cpu = sum(request.backend_id == 1 for request in typed)
        if (cpu, len(typed) - cpu) != (
            summaries[definition.name].cpu_requests,
            summaries[definition.name].gpu_requests,
        ):
            raise SystemExit(f"backend-count mismatch for {definition.name} in {path}")
        if definition.routing_mode == "forced_gpu" and cpu:
            raise SystemExit(f"forced-GPU case {definition.name} used CPU in {path}")
    if len(input_counts) != 1:
        raise SystemExit(f"{path} did not keep one measured request shape")


def _validate_case_logs(path: Path, text: str, requests: dict[tuple[str, str], Request]) -> dict[str, int]:
    margins = {}
    sections = re.split(r"Starting benchmark case: (\S+)", text)
    observed = set()
    for case_name, section in zip(sections[1::2], sections[2::2]):
        if case_name not in CASES_BY_NAME or case_name in observed:
            raise SystemExit(f"unexpected or duplicate case boundary {case_name} in {path}")
        observed.add(case_name)
        definition = CASES_BY_NAME[case_name]
        labels = tuple(f"{definition.workload.name}-{index + 1:02d}" for index in range(definition.workload.request_count))
        replay = list(REPLAY_ROW.finditer(section))
        expected = (
            case_name,
            definition.workload.name,
            definition.routing_mode,
            "50",
            "1",
            str(len(labels)),
            ",".join(labels),
        )
        if len(replay) != 1 or replay[0].groups() != expected:
            raise SystemExit(f"invalid replay metadata for {case_name} in {path}")
        seeds = list(SEED_ROW.finditer(section))
        if len(seeds) != 2 or {(int(row.group(2)), int(row.group(3))) for row in seeds} != {(1, 1), (2, 2)}:
            raise SystemExit(f"invalid cache seeds for {case_name} in {path}")
        if any(int(row.group(5)) <= 0 for row in seeds):
            raise SystemExit(f"empty cache seed for {case_name} in {path}")

        labels_by_id = {requests[(case_name, label)].request_id: label for label in labels}
        routes = [tuple(map(int, match.groups())) for match in ROUTING_ROW.finditer(section)]
        routes.sort(key=lambda route: route[1])
        measured = [(labels_by_id[request_id], backend) for request_id, _, backend in routes if request_id in labels_by_id]
        if tuple(label for label, _ in measured) != labels:
            raise SystemExit(f"routing order mismatch for {case_name} in {path}; use RUST_LOG=info")
        if any(backend != requests[(case_name, label)].backend_id for label, backend in measured):
            raise SystemExit(f"routing/backend mismatch for {case_name} in {path}")
        offsets = [requests[(case_name, label)].submitted_ms for label in labels]
        if any(right < left for left, right in zip(offsets, offsets[1:])):
            raise SystemExit(f"nonmonotonic submissions for {case_name} in {path}")

        candidates = {}
        for match in CANDIDATE_ROW.finditer(section):
            request_id, epoch, selected, backend, fields = match.groups()
            duration = re.search(r"\bestimated_duration_ns=(\d+)\b", fields)
            if duration is None:
                raise SystemExit(f"missing estimated duration for {case_name} in {path}")
            candidates[(int(request_id), int(epoch), int(backend))] = (selected == "true", int(duration.group(1)))
        case_margins = []
        for request_id, epoch, backend in routes:
            if request_id not in labels_by_id:
                continue
            selected = candidates.get((request_id, epoch, backend))
            alternative = candidates.get((request_id, epoch, 3 - backend))
            if selected is None or alternative is None:
                raise SystemExit(f"incomplete routing candidates for {case_name} in {path}")
            if definition.routing_mode == "forced_gpu":
                continue
            if not selected[0] or alternative[0] or selected[1] > alternative[1]:
                raise SystemExit(f"inconsistent routing candidates for {case_name} in {path}")
            case_margins.append(alternative[1] - selected[1])
        if case_margins:
            margins[case_name] = min(case_margins)
    if observed != set(CASES_BY_NAME):
        raise SystemExit(f"{path} lacks benchmark case boundaries")
    return margins


def _validate_across_runs(runs: list[Run]) -> None:
    shapes = {(key, request.input_tokens, request.output_tokens) for key, request in runs[0].requests.items()}
    for index, run in enumerate(runs[1:], start=2):
        if {(key, request.input_tokens, request.output_tokens) for key, request in run.requests.items()} != shapes:
            raise SystemExit(f"run {index} used different request token counts")
    for run in runs:
        counts = [run.summaries[f"{workload.name}_mixed_routing"].cpu_requests for workload in WORKLOADS]
        if counts != sorted(counts):
            raise SystemExit("CPU use decreased as request count increased")


def _print_summary(runs: list[Run]) -> None:
    rows = []
    for definition in CASES:
        values = [run.summaries[definition.name] for run in runs]
        rows.append((definition.workload.name, definition.routing_mode,
                     _distribution([value.cpu_requests for value in values]),
                     _distribution([value.wall_seconds for value in values]),
                     _distribution([value.output_tokens_per_second for value in values]),
                     _distribution([value.median_ttft_ms for value in values]),
                     _distribution([value.p95_ttft_ms for value in values])))
    print("\nCase summary")
    print(tabulate(rows, headers=("Workload", "Routing", "CPU requests", "Wall s", "Output tok/s", "Median TTFT ms", "p95 TTFT ms"), tablefmt="simple"))


def _print_paired_changes(runs: list[Run]) -> None:
    rows = []
    for workload in WORKLOADS:
        mixed = f"{workload.name}_mixed_routing"
        forced = f"{workload.name}_forced_gpu"
        rows.append((workload.name, *(_paired_change(runs, mixed, forced, field) for field in HEADLINES)))
    print("\nMixed-routing change from forced GPU")
    print(tabulate(rows, headers=("Workload", "Wall", "Output throughput", "Median TTFT", "p95 TTFT", "Median E2E", "p95 E2E"), tablefmt="simple"))


def _print_routing(runs: list[Run]) -> None:
    rows = []
    for workload in WORKLOADS:
        case = f"{workload.name}_mixed_routing"
        cpu_counts = [run.summaries[case].cpu_requests for run in runs]
        margins = [run.closest_margins_ns[case] / 1e9 for run in runs]
        rows.append((workload.name, _distribution(cpu_counts), _distribution(margins)))
    print("\nMixed-routing placement and closest estimated-score margin")
    print(tabulate(rows, headers=("Workload", "CPU requests", "Closest margin s"), tablefmt="simple"))


def _print_variance(runs: list[Run]) -> None:
    coefficients = []
    for case_name in CASES_BY_NAME:
        for field in HEADLINES:
            values = [getattr(run.summaries[case_name], field) for run in runs]
            coefficients.append((stdev(values) / mean(values), case_name, field))
    maximum, case_name, field = max(coefficients)
    print(f"\nMaximum headline CV: {maximum * 100:.2f}% ({case_name} / {field})")
    if maximum > MAXIMUM_CV:
        print("Headline variance exceeds 5%; inspect placement and system conditions before collecting more runs.")
    else:
        print("Headline variance is at or below 5%; no extra runs are required.")


def _distribution(values: list[float | int]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def _paired_change(runs: list[Run], candidate: str, baseline: str, field: str) -> str:
    changes = [getattr(run.summaries[candidate], field) / getattr(run.summaries[baseline], field) - 1 for run in runs]
    if len(changes) == 1:
        return f"{changes[0] * 100:+.1f}%"
    return f"{mean(changes) * 100:+.1f}% ± {stdev(changes) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
