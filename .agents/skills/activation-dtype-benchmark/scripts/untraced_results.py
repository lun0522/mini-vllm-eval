#!/usr/bin/env python3
"""Aggregate untraced CPU activation-dtype benchmark logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean
from statistics import stdev
from typing import Any

from tabulate import tabulate


CONFIGURATIONS = (
    "F32",
    "F16",
    "F16-QMatMul",
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
BENCHMARK_ROW = re.compile(
    rf"^\s*({'|'.join(re.escape(value) for value in CONFIGURATIONS)})\s+"
    r"([0-9.]+)\s+\S+\s+([0-9.]+)\s+([0-9.]+)\s+\S+\s+([0-9.]+)\s+\S+\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
    re.MULTILINE,
)
REQUEST_DESCRIPTION = re.compile(
    r"CPU activation dtype comparison with a (\d+)-token small prefill and a "
    r"(\d+)-token input / (\d+)-token output long request"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate untraced CPU activation-dtype benchmark logs."
    )
    parser.add_argument(
        "logs",
        type=Path,
        nargs="+",
        help="untraced benchmark logs to aggregate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _aggregate_logs(args.logs)
    return 0


def _parse_benchmark_log(log: Path) -> tuple[dict[str, Any], dict[str, int]]:
    text = ANSI_ESCAPE.sub("", log.read_text())
    request_match = REQUEST_DESCRIPTION.search(text)
    if request_match is None:
        raise SystemExit(f"benchmark request description not found in {log}")

    rows: dict[str, tuple[str, ...]] = {}
    for match in BENCHMARK_ROW.finditer(text):
        rows[match.group(1)] = match.groups()[1:]
    if set(rows) != set(CONFIGURATIONS):
        raise SystemExit(f"all three activation benchmark rows not found in {log}")

    measurements: dict[str, Any] = {"benchmark": {}}
    names = (
        "small_prefill_ttft_ms",
        "large_prefill_ttft_ms",
        "e2e_seconds",
        "decode_tokens_per_second",
        "starting_rss_mib",
        "peak_rss_mib",
        "peak_increase_mib",
    )
    for index, name in enumerate(names):
        measurements["benchmark"][name] = {
            configuration: float(rows[configuration][index])
            for configuration in CONFIGURATIONS
        }
    request = {
        "small_prefill_input_token_count": int(request_match.group(1)),
        "input_token_count": int(request_match.group(2)),
        "output_token_count": int(request_match.group(3)),
    }
    return measurements, request


def _aggregate_logs(log_paths: list[Path]) -> None:
    if len(log_paths) < 2:
        raise SystemExit("at least two benchmark logs are required")
    parsed = [
        _parse_benchmark_log(path.expanduser().resolve()) for path in log_paths
    ]
    request = parsed[0][1]
    if any(parsed_request != request for _, parsed_request in parsed[1:]):
        raise SystemExit("benchmark logs describe different requests")
    _print_aggregate([measurements for measurements, _ in parsed], request)


def _print_aggregate(
    measurements: list[dict[str, Any]],
    request: dict[str, int],
) -> None:
    flattened = [_flatten(run) for run in measurements]
    keys = sorted(flattened[0])
    if any(sorted(result) != keys for result in flattened[1:]):
        raise SystemExit("result files contain different measurement sets")

    rows = []
    for key in keys:
        if not key.endswith(".F32"):
            continue
        base = key.removesuffix(".F32")
        f32 = [result[key] for result in flattened]
        distributions = []
        ratios = []
        for configuration in CONFIGURATIONS:
            values = [result[f"{base}.{configuration}"] for result in flattened]
            distributions.append(_distribution(values))
            if configuration != "F32":
                ratios.append(
                    f"{mean(values) / mean(f32):.3f}x" if mean(f32) else "-"
                )
        rows.append((base, *distributions, *ratios))

    print(
        f"Activation benchmark aggregate: {len(measurements)} runs, "
        f"{request['small_prefill_input_token_count']} small-prefill tokens, "
        f"{request['input_token_count']} input tokens, "
        f"{request['output_token_count']} output tokens"
    )
    print(
        tabulate(
            rows,
            headers=(
                "Measurement",
                *(f"{configuration} mean ± SD" for configuration in CONFIGURATIONS),
                *(f"{configuration}/F32" for configuration in CONFIGURATIONS[1:]),
            ),
            tablefmt="simple",
        )
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(child, child_prefix))
        return flattened
    if not isinstance(value, (int, float)):
        raise TypeError(f"measurement {prefix} is not numeric: {value!r}")
    return {prefix: float(value)}


def _distribution(values: list[float]) -> str:
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
