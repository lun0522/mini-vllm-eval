#!/usr/bin/env python3
"""Aggregate speculative-sampling proposal break-even benchmark logs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

from tabulate import tabulate


MAXIMUM_CV = 0.05
TOKENS_PER_REQUEST = 256
RESULT_PATTERN = re.compile(
    r"SPECULATIVE_SAMPLING_FOLLOWUP_RESULT "
    r"configuration=(\S+) "
    r"proposal_length=(none|\d+) "
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
    r"(?: verification_round_count=(none|\d+))?"
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True, order=True)
class Configuration:
    kind: str
    proposal_length: int | None
    tv_budget: float | None

    @property
    def label(self) -> str:
        if self.kind == "target":
            return "Target-only"
        if self.kind == "draft":
            return "Draft-only"
        return f"Fixed-{self.proposal_length}, TV={self.tv_budget:g}"


@dataclass(frozen=True)
class Measurement:
    path: Path
    seed_count: int
    median_e2e: float
    minimum_e2e: float
    maximum_e2e: float
    median_decode: float
    accepted: int
    proposed: int
    rounds: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate proposal break-even performance logs."
    )
    parser.add_argument("logs", type=Path, nargs="+", help="performance logs")
    parser.add_argument(
        "--samples-per-point",
        type=int,
        default=3,
        choices=(3, 5),
        help="required independent invocations for each configuration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.logs]
    if len(set(paths)) != len(paths):
        raise SystemExit("log paths must be distinct")

    grouped: dict[Configuration, list[Measurement]] = defaultdict(list)
    for path in paths:
        for configuration, measurement in _parse_log(path).items():
            grouped[configuration].append(measurement)

    _validate_series(grouped, args.samples_per_point)
    _print_results(grouped)
    return 0


def _parse_log(path: Path) -> dict[Configuration, Measurement]:
    if not path.is_file():
        raise SystemExit(f"benchmark log does not exist: {path}")
    text = ANSI_ESCAPE.sub("", path.read_text())
    results: dict[Configuration, Measurement] = {}
    for match in RESULT_PATTERN.finditer(text):
        configuration = _configuration(match.group(1), match.group(2), match.group(3))
        if configuration in results:
            raise SystemExit(f"duplicate result for {configuration.label} in {path}")
        seed_count = int(match.group(4))
        accepted = int(match.group(9))
        proposed = int(match.group(10))
        output_per_round = _optional_float(match.group(12))
        explicit_rounds = match.group(13)
        rounds = (
            None
            if explicit_rounds in (None, "none")
            else int(explicit_rounds)
        )
        if rounds is None and output_per_round is not None:
            estimated = seed_count * TOKENS_PER_REQUEST / output_per_round
            rounds = round(estimated)
            if abs(estimated - rounds) > 1e-6:
                raise SystemExit(
                    f"cannot reconstruct verification count for "
                    f"{configuration.label} in {path}"
                )
        measurement = Measurement(
            path=path,
            seed_count=seed_count,
            median_e2e=float(match.group(5)),
            minimum_e2e=float(match.group(6)),
            maximum_e2e=float(match.group(7)),
            median_decode=float(match.group(8)),
            accepted=accepted,
            proposed=proposed,
            rounds=rounds,
        )
        _validate_measurement(configuration, measurement)
        results[configuration] = measurement
    if not results:
        raise SystemExit(f"no proposal break-even results found in {path}")
    return results


def _configuration(name: str, length: str, budget: str) -> Configuration:
    proposal_length = None if length == "none" else int(length)
    tv_budget = None if budget == "none" else float(budget)
    if name == "Target_only" and proposal_length is None and tv_budget is None:
        return Configuration("target", None, None)
    if name == "Draft_only" and proposal_length is None and tv_budget is None:
        return Configuration("draft", None, None)
    if proposal_length is None or tv_budget is None:
        raise SystemExit(f"invalid configuration fields: {name}")
    return Configuration("fixed", proposal_length, tv_budget)


def _optional_float(value: str) -> float | None:
    return None if value == "none" else float(value)


def _validate_measurement(
    configuration: Configuration, measurement: Measurement
) -> None:
    if measurement.seed_count != 3:
        raise SystemExit(
            f"{configuration.label} in {measurement.path} does not contain three seeds"
        )
    if not (
        0 < measurement.minimum_e2e
        <= measurement.median_e2e
        <= measurement.maximum_e2e
    ):
        raise SystemExit(f"invalid E2E values for {configuration.label}")
    if measurement.median_decode <= 0:
        raise SystemExit(f"invalid decode throughput for {configuration.label}")
    if not 0 <= measurement.accepted <= measurement.proposed:
        raise SystemExit(f"invalid draft-token totals for {configuration.label}")
    if configuration.kind in ("target", "draft"):
        if measurement.proposed != 0 or measurement.rounds is not None:
            raise SystemExit(f"unexpected draft metrics for {configuration.label}")
    elif measurement.proposed == 0 or measurement.rounds is None:
        raise SystemExit(f"missing draft metrics for {configuration.label}")
    if configuration.kind == "fixed" and configuration.tv_budget == 1:
        if measurement.accepted != measurement.proposed:
            raise SystemExit(
                f"TV=1 did not accept every proposal in {measurement.path}"
            )


def _validate_series(
    grouped: dict[Configuration, list[Measurement]], samples_per_point: int
) -> None:
    target = Configuration("target", None, None)
    if target not in grouped:
        raise SystemExit("the series does not contain target-only measurements")
    invalid = {
        configuration.label: len(values)
        for configuration, values in grouped.items()
        if len(values) != samples_per_point
    }
    if invalid:
        raise SystemExit(
            f"every point must have {samples_per_point} samples; observed={invalid}"
        )


def _sort_key(configuration: Configuration) -> tuple[float, float, float]:
    if configuration.kind == "target":
        return (0, 0, 0)
    if configuration.kind == "draft":
        return (1, 0, 0)
    return (2, configuration.proposal_length or 0, configuration.tv_budget or 0)


def _cv(values: list[float]) -> float:
    return 0.0 if len(values) < 2 else stdev(values) / mean(values)


def _classification(values: list[Measurement], target: list[Measurement]) -> str:
    observed = [value.median_e2e for value in values]
    baseline = [value.median_e2e for value in target]
    if max(observed) < min(baseline):
        return "clearly faster"
    if min(observed) > max(baseline):
        return "clearly slower"
    return "approximately equal"


def _print_results(grouped: dict[Configuration, list[Measurement]]) -> None:
    target_key = Configuration("target", None, None)
    target = grouped[target_key]
    target_e2e = median(value.median_e2e for value in target)
    target_decode = median(value.median_decode for value in target)
    rows = []
    high_variance = []
    for configuration in sorted(grouped, key=_sort_key):
        values = grouped[configuration]
        e2e_values = [value.median_e2e for value in values]
        decode_values = [value.median_decode for value in values]
        e2e = median(e2e_values)
        decode = median(decode_values)
        accepted = sum(value.accepted for value in values)
        proposed = sum(value.proposed for value in values)
        rounds = sum(value.rounds or 0 for value in values)
        output_tokens = sum(value.seed_count * TOKENS_PER_REQUEST for value in values)
        verification_counts = [
            value.rounds for value in values if value.rounds is not None
        ]
        classification = (
            "baseline"
            if configuration == target_key
            else _classification(values, target)
        )
        rows.append(
            (
                configuration.label,
                f"{e2e:.3f} [{min(e2e_values):.3f}, {max(e2e_values):.3f}]",
                "-"
                if configuration == target_key
                else f"{(e2e / target_e2e - 1):+.1%}",
                f"{decode:.2f}",
                "-"
                if configuration == target_key
                else f"{(decode / target_decode - 1):+.1%}",
                classification,
                "-" if not proposed else f"{accepted / proposed:.1%}",
                "-" if not rounds else f"{accepted / rounds:.2f}",
                "-" if not rounds else f"{output_tokens / rounds:.2f}",
                "-"
                if not rounds
                else f"{median(verification_counts):g}",
            )
        )
        if _cv(e2e_values) > MAXIMUM_CV or _cv(decode_values) > MAXIMUM_CV:
            high_variance.append(configuration.label)

    sample_count = len(next(iter(grouped.values())))
    print(f"Proposal break-even aggregate: {sample_count} samples per point")
    print(
        tabulate(
            rows,
            headers=(
                "Configuration",
                "E2E s [invocation-median range]",
                "E2E vs target",
                "Decode tok/s",
                "Decode vs target",
                "E2E classification",
                "Acceptance",
                "Accepted/round",
                "Output/round",
                "Verifications/invocation",
            ),
            tablefmt="simple",
        )
    )
    _print_break_even(grouped, target)
    if high_variance:
        configurations = ", ".join(high_variance)
        print(f"\nCollect five samples per point; CV exceeded 5% for: {configurations}")
    else:
        print("\nAll E2E and decode-throughput CVs are at most 5%.")


def _print_break_even(
    grouped: dict[Configuration, list[Measurement]], target: list[Measurement]
) -> None:
    lengths = sorted(
        {key.proposal_length for key in grouped if key.kind == "fixed"}
    )
    print("\nObserved E2E break-even")
    for length in lengths:
        points = sorted(
            (
                (key.tv_budget, values)
                for key, values in grouped.items()
                if key.kind == "fixed" and key.proposal_length == length
            ),
            key=lambda item: item[0],
        )
        winners = [
            budget
            for budget, values in points
            if _classification(values, target) == "clearly faster"
        ]
        if not winners:
            print(
                f"- Fixed-{length}: no tested TV budget clearly beats target-only"
            )
            continue
        winner = min(winners)
        lower = [budget for budget, _values in points if budget < winner]
        if lower:
            print(
                f"- Fixed-{length}: lowest clear win TV={winner:g}; "
                f"observed interval ({max(lower):g}, {winner:g}]"
            )
        else:
            print(
                f"- Fixed-{length}: clearly faster at the lowest tested budget, "
                f"TV={winner:g}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
