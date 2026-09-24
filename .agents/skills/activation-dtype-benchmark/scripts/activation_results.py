#!/usr/bin/env python3
"""Record and aggregate CPU activation-dtype benchmark results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from statistics import stdev
from typing import Any

from tabulate import tabulate

from activation_traces import ForwardMetrics
from activation_traces import _read_forwards
from activation_traces import _select_forwards


DTYPES = ("F32", "F16")
PHASES = ("prefill", "last decode")
SUMMARY_SPANS = (
    "model",
    "attn",
    "qmatmul",
    "attn-rope",
    "attn-cache",
    "attn-qk",
    "attn-mask-softmax",
    "attn-v",
    "mlp",
    "output",
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
BENCHMARK_ROW = re.compile(
    r"^\s*(F32|F16)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+\S+\s+([0-9.]+)\s+\S+\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
    re.MULTILINE,
)
REQUEST_DESCRIPTION = re.compile(
    r"CPU activation dtype comparison with a (\d+)-token input and (\d+)-token output"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record or aggregate CPU activation-dtype benchmark results."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="record one benchmark run")
    record.add_argument("benchmark_log", type=Path)
    record.add_argument("f32_trace", type=Path)
    record.add_argument("f16_trace", type=Path)
    record.add_argument("output", type=Path)

    aggregate = subparsers.add_parser(
        "aggregate", help="aggregate two or more recorded runs"
    )
    aggregate.add_argument("results", type=Path, nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "record":
        result = _record(args.benchmark_log, args.f32_trace, args.f16_trace)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as file:
            json.dump(result, file, indent=2, sort_keys=True)
            file.write("\n")
        print(f"Recorded activation benchmark result: {output}")
        return 0

    _aggregate(args.results)
    return 0


def _record(benchmark_log: Path, f32_trace: Path, f16_trace: Path) -> dict[str, Any]:
    benchmark_log = benchmark_log.expanduser().resolve()
    traces = {
        "F32": f32_trace.expanduser().resolve(),
        "F16": f16_trace.expanduser().resolve(),
    }
    for path in (benchmark_log, *traces.values()):
        if not path.is_file():
            raise SystemExit(f"input file does not exist: {path}")

    measurements, request = _parse_benchmark_log(benchmark_log)
    forwards = {
        dtype: _select_forwards(_read_forwards(trace))
        for dtype, trace in traces.items()
    }
    for phase in PHASES:
        phase_forwards = {dtype: forwards[dtype][phase] for dtype in DTYPES}
        _record_trace_phase(measurements, phase, phase_forwards)

    return {
        "schema_version": 1,
        "request": request,
        "measurements": measurements,
    }


def _parse_benchmark_log(log: Path) -> tuple[dict[str, Any], dict[str, int]]:
    text = ANSI_ESCAPE.sub("", log.read_text())
    request_match = REQUEST_DESCRIPTION.search(text)
    if request_match is None:
        raise SystemExit(f"benchmark request description not found in {log}")

    rows: dict[str, tuple[str, ...]] = {}
    for match in BENCHMARK_ROW.finditer(text):
        rows[match.group(1)] = match.groups()[1:]
    if set(rows) != set(DTYPES):
        raise SystemExit(f"F32/F16 benchmark rows not found in {log}")

    measurements: dict[str, Any] = {"benchmark": {}}
    names = (
        "ttft_ms",
        "e2e_seconds",
        "decode_tokens_per_second",
        "starting_rss_mib",
        "peak_rss_mib",
        "peak_increase_mib",
    )
    for index, name in enumerate(names):
        measurements["benchmark"][name] = {
            dtype: float(rows[dtype][index]) for dtype in DTYPES
        }
    request = {
        "input_token_count": int(request_match.group(1)),
        "output_token_count": int(request_match.group(2)),
    }
    return measurements, request


def _record_trace_phase(
    measurements: dict[str, Any],
    phase: str,
    forwards: dict[str, ForwardMetrics],
) -> None:
    phase_measurements: dict[str, Any] = {"summary": {}, "qmatmul": {}, "attention": {}}
    measurements[phase] = phase_measurements

    for span in SUMMARY_SPANS:
        values = {
            dtype: (
                forward.duration_microseconds
                if span == "model"
                else forward.aggregate(span).total_microseconds
            )
            / 1_000
            for dtype, forward in forwards.items()
        }
        if any(values.values()):
            phase_measurements["summary"][span] = values

    operations = sorted(
        {
            dict(args).get("operation")
            for forward in forwards.values()
            for (name, args) in forward.spans
            if name == "qmatmul"
        }
        - {None}
    )
    for operation in operations:
        phase_measurements["qmatmul"][operation] = {
            dtype: forward.aggregate(
                "qmatmul", operation=operation
            ).total_microseconds
            / 1_000
            for dtype, forward in forwards.items()
        }

    attention_keys = sorted(
        {
            (name, args)
            for forward in forwards.values()
            for (name, args) in forward.spans
            if name.startswith("attn-")
        },
        key=lambda item: (item[0], item[1]),
    )
    for name, args_tuple in attention_keys:
        args = dict(args_tuple)
        label = name
        if args_tuple:
            label += "[" + ",".join(f"{key}={value}" for key, value in args_tuple) + "]"
        phase_measurements["attention"][label] = {
            dtype: forward.aggregate(name, **args).total_microseconds / 1_000
            for dtype, forward in forwards.items()
        }


def _aggregate(result_paths: list[Path]) -> None:
    if len(result_paths) < 2:
        raise SystemExit("aggregate requires at least two result files")
    documents = []
    for path in result_paths:
        path = path.expanduser().resolve()
        with path.open() as file:
            document = json.load(file)
        if document.get("schema_version") != 1:
            raise SystemExit(f"unsupported result schema in {path}")
        documents.append(document)

    request = documents[0]["request"]
    if any(document["request"] != request for document in documents[1:]):
        raise SystemExit("result files describe different benchmark requests")

    flattened = [_flatten(document["measurements"]) for document in documents]
    keys = sorted(flattened[0])
    if any(sorted(result) != keys for result in flattened[1:]):
        raise SystemExit("result files contain different measurement sets")

    rows = []
    for key in keys:
        if not key.endswith(".F32"):
            continue
        base = key.removesuffix(".F32")
        f16_key = f"{base}.F16"
        f32 = [result[key] for result in flattened]
        f16 = [result[f16_key] for result in flattened]
        rows.append(
            (
                base,
                _distribution(f32),
                _distribution(f16),
                f"{mean(f16) / mean(f32):.3f}x" if mean(f32) else "-",
            )
        )

    print(
        f"Activation benchmark aggregate: {len(documents)} runs, "
        f"{request['input_token_count']} input tokens, "
        f"{request['output_token_count']} output tokens"
    )
    print(
        tabulate(
            rows,
            headers=("Measurement", "F32 mean ± SD", "F16 mean ± SD", "F16/F32"),
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
