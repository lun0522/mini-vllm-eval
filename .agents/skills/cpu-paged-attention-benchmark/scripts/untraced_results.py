#!/usr/bin/env python3
"""Aggregate repeated F32 and F16 CPU paged-attention benchmark logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean
from statistics import stdev

from tabulate import tabulate


REQUESTS = ("Small-Prefill", "Long-Request")
MODES = (
    ("Contiguous", "Repeated KV", "Full V"),
    ("Contiguous", "Grouped Q", "Full V"),
    ("Paged", "Repeated KV", "Concatenated V"),
    ("Paged", "Repeated KV", "Page-wise V"),
    ("Paged", "Grouped Q", "Concatenated V"),
    ("Paged", "Grouped Q", "Page-wise V"),
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
REQUEST_HEADING = re.compile(
    r"CPU attention comparison for (Small-Prefill|Long-Request):"
)
RSS_HEADING = "Peak RSS comparison during the large request:"

Mode = tuple[str, str, str]
Measurements = dict[str, dict[Mode, dict[str, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate F32 and F16 CPU paged-attention benchmark logs."
    )
    parser.add_argument(
        "--f32",
        type=Path,
        nargs=5,
        required=True,
        metavar="LOG",
        help="five F32 benchmark logs",
    )
    parser.add_argument(
        "--f16",
        type=Path,
        nargs=5,
        required=True,
        metavar="LOG",
        help="five F16 benchmark logs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = {
        "F32": [_parse_log(path.expanduser().resolve()) for path in args.f32],
        "F16": [_parse_log(path.expanduser().resolve()) for path in args.f16],
    }
    _print_results(runs)
    return 0


def _parse_log(path: Path) -> Measurements:
    if not path.is_file():
        raise SystemExit(f"benchmark log does not exist: {path}")
    return _parse_log_text(ANSI_ESCAPE.sub("", path.read_text()), str(path))


def _parse_log_text(text: str, source: str) -> Measurements:
    measurements: Measurements = {request: {} for request in REQUESTS}
    measurements["RSS"] = {}
    section: str | None = None

    for line in text.splitlines():
        request_match = REQUEST_HEADING.search(line)
        if request_match is not None:
            section = request_match.group(1)
            continue
        if RSS_HEADING in line:
            section = "RSS"
            continue

        columns = re.split(r"\s{2,}", line.strip())
        if len(columns) < 6:
            continue
        mode = tuple(columns[:3])
        if mode not in MODES or section is None:
            continue

        try:
            if section == "RSS":
                values = {"peak_rss_increase_mib": float(columns[5])}
            else:
                values = {
                    "ttft_ms": float(columns[3]),
                    "e2e_seconds": float(columns[4]),
                    "decode_tokens_per_second": _optional_float(columns[7]),
                }
        except (IndexError, ValueError) as error:
            raise SystemExit(
                f"invalid {section} row for {_mode_name(mode)} in {source}: {line}"
            ) from error
        measurements[section][mode] = values

    for section_name, modes in measurements.items():
        if set(modes) != set(MODES):
            missing = ", ".join(
                _mode_name(mode) for mode in MODES if mode not in modes
            )
            raise SystemExit(f"{source} is missing {section_name} rows: {missing}")
    return measurements


def _optional_float(value: str) -> float:
    return float("nan") if value == "-" else float(value)


def _print_results(runs: dict[str, list[Measurements]]) -> None:
    print("CPU paged-attention aggregate: 5 F32 runs and 5 F16 runs")
    _print_request_table(runs, "Small-Prefill", include_decode=False)
    _print_request_table(runs, "Long-Request", include_decode=True)
    _print_rss_table(runs)


def _print_request_table(
    runs: dict[str, list[Measurements]],
    request: str,
    *,
    include_decode: bool,
) -> None:
    rows = []
    for mode in MODES:
        f32_ttft = _values(runs["F32"], request, mode, "ttft_ms")
        f16_ttft = _values(runs["F16"], request, mode, "ttft_ms")
        row = [
            _mode_name(mode),
            _distribution(f32_ttft),
            _distribution(f16_ttft),
            _ratio(f16_ttft, f32_ttft),
        ]
        if include_decode:
            f32_e2e = _values(runs["F32"], request, mode, "e2e_seconds")
            f16_e2e = _values(runs["F16"], request, mode, "e2e_seconds")
            f32_decode = _values(
                runs["F32"], request, mode, "decode_tokens_per_second"
            )
            f16_decode = _values(
                runs["F16"], request, mode, "decode_tokens_per_second"
            )
            row.extend(
                (
                    _distribution(f32_e2e),
                    _distribution(f16_e2e),
                    _ratio(f16_e2e, f32_e2e),
                    _distribution(f32_decode),
                    _distribution(f16_decode),
                    _ratio(f16_decode, f32_decode),
                )
            )
        rows.append(row)

    headers = ["Configuration", "F32 TTFT ms", "F16 TTFT ms", "F16/F32"]
    if include_decode:
        headers.extend(
            (
                "F32 E2E s",
                "F16 E2E s",
                "F16/F32",
                "F32 decode tok/s",
                "F16 decode tok/s",
                "F16/F32",
            )
        )
    print(f"\n{request}")
    print(tabulate(rows, headers=headers, tablefmt="simple"))


def _print_rss_table(runs: dict[str, list[Measurements]]) -> None:
    rows = []
    for mode in MODES:
        f32 = _values(runs["F32"], "RSS", mode, "peak_rss_increase_mib")
        f16 = _values(runs["F16"], "RSS", mode, "peak_rss_increase_mib")
        rows.append(
            (
                _mode_name(mode),
                _distribution(f32),
                _distribution(f16),
                _ratio(f16, f32),
            )
        )
    print("\nLong-Request peak RSS increase")
    print(
        tabulate(
            rows,
            headers=("Configuration", "F32 MiB", "F16 MiB", "F16/F32"),
            tablefmt="simple",
        )
    )


def _values(
    runs: list[Measurements],
    section: str,
    mode: Mode,
    measurement: str,
) -> list[float]:
    return [run[section][mode][measurement] for run in runs]


def _distribution(values: list[float]) -> str:
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def _ratio(numerator: list[float], denominator: list[float]) -> str:
    return f"{mean(numerator) / mean(denominator):.3f}x"


def _mode_name(mode: Mode) -> str:
    return " / ".join(mode)


if __name__ == "__main__":
    raise SystemExit(main())
