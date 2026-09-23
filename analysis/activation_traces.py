#!/usr/bin/env python3
"""Analyze measured prefill and final decode in F32/F16 Chrome traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from tabulate import tabulate


@dataclass
class SpanMetrics:
    count: int = 0
    total_microseconds: float = 0
    maximum_microseconds: float = 0

    def record(self, duration_microseconds: float) -> None:
        self.count += 1
        self.total_microseconds += duration_microseconds
        self.maximum_microseconds = max(
            self.maximum_microseconds,
            duration_microseconds,
        )


@dataclass
class ForwardMetrics:
    index: int
    duration_microseconds: float = 0
    spans: dict[tuple[str, tuple[tuple[str, Any], ...]], SpanMetrics] = field(
        default_factory=lambda: defaultdict(SpanMetrics)
    )

    def record_span(
        self,
        name: str,
        args: dict[str, Any],
        duration_microseconds: float,
    ) -> None:
        normalized_args = tuple(
            sorted((key, _normalize_arg(value)) for key, value in args.items())
        )
        self.spans[(name, normalized_args)].record(duration_microseconds)

    def aggregate(self, name: str, **required_args: Any) -> SpanMetrics:
        result = SpanMetrics()
        for (span_name, span_args), metrics in self.spans.items():
            if span_name != name:
                continue
            args = dict(span_args)
            if any(args.get(key) != value for key, value in required_args.items()):
                continue
            result.count += metrics.count
            result.total_microseconds += metrics.total_microseconds
            result.maximum_microseconds = max(
                result.maximum_microseconds,
                metrics.maximum_microseconds,
            )
        return result


@dataclass(frozen=True)
class OpenSpan:
    name: str
    timestamp_microseconds: float
    args: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare F32 and F16 mini-vllm Chrome traces."
    )
    parser.add_argument(
        "f32_trace",
        type=Path,
        help="F32 Chrome trace file",
    )
    parser.add_argument(
        "f16_trace",
        type=Path,
        help="F16 Chrome trace file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    traces = {
        "F32": args.f32_trace.expanduser().resolve(),
        "F16": args.f16_trace.expanduser().resolve(),
    }
    for dtype, trace in traces.items():
        if not trace.is_file():
            raise SystemExit(f"{dtype} trace file does not exist: {trace}")
    selected_forwards = {
        dtype: _select_forwards(_read_forwards(trace))
        for dtype, trace in traces.items()
    }

    print("Traces:")
    for dtype, trace in traces.items():
        print(f"  {dtype}: {trace}")
    print()

    for phase in ("prefill", "last decode"):
        phase_forwards = {
            dtype: forwards[phase] for dtype, forwards in selected_forwards.items()
        }
        _print_summary(phase, phase_forwards)
        _print_qmatmul_breakdown(phase_forwards)
        _print_attention_breakdown(phase_forwards)
    return 0


def _read_forwards(trace: Path) -> list[ForwardMetrics]:
    with trace.open() as file:
        document = json.load(file)
    events = document["traceEvents"] if isinstance(document, dict) else document

    stacks: dict[tuple[int, int], list[OpenSpan]] = defaultdict(list)
    active_forwards: dict[tuple[int, int], ForwardMetrics] = {}
    forwards: list[ForwardMetrics] = []
    for event in events:
        phase = event.get("ph")
        if phase not in ("B", "E"):
            continue
        thread = (event["pid"], event["tid"])
        if phase == "B":
            span = OpenSpan(
                name=event["name"],
                timestamp_microseconds=event["ts"],
                args=event.get("args", {}),
            )
            stacks[thread].append(span)
            if span.name == "model":
                if thread in active_forwards:
                    raise RuntimeError(f"nested model spans in {trace}")
                active_forwards[thread] = ForwardMetrics(index=len(forwards) + 1)
            continue

        if not stacks[thread]:
            raise RuntimeError(f"unmatched end event in {trace}: {event}")
        span = stacks[thread].pop()
        if event.get("name") != span.name:
            raise RuntimeError(
                f"mismatched span events in {trace}: {span.name} and {event.get('name')}"
            )
        duration = event["ts"] - span.timestamp_microseconds
        active_forward = active_forwards.get(thread)
        if span.name == "model":
            if active_forward is None:
                raise RuntimeError(f"model span ended without starting in {trace}")
            active_forward.duration_microseconds = duration
            forwards.append(active_forward)
            del active_forwards[thread]
        elif active_forward is not None:
            active_forward.record_span(span.name, span.args, duration)

    if active_forwards or any(stacks.values()):
        raise RuntimeError(f"trace contains unfinished spans: {trace}")
    if len(forwards) < 2:
        raise RuntimeError(f"trace contains fewer than two model forwards: {trace}")
    return forwards


def _select_forwards(forwards: list[ForwardMetrics]) -> dict[str, ForwardMetrics]:
    return {
        "prefill": forwards[1],
        "last decode": forwards[-1],
    }


def _print_summary(
    phase: str,
    forwards: dict[str, ForwardMetrics],
) -> None:
    metric_names = (
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
    rows = []
    for name in metric_names:
        durations = {
            dtype: (
                forward.duration_microseconds
                if name == "model"
                else forward.aggregate(name).total_microseconds
            )
            for dtype, forward in forwards.items()
        }
        if durations["F32"] == 0 and durations["F16"] == 0:
            continue
        rows.append(
            (
                name,
                _milliseconds(durations["F32"]),
                _milliseconds(durations["F16"]),
                _ratio(durations["F16"], durations["F32"]),
            )
        )
    print(f"{phase.title()} summary:")
    print(
        tabulate(
            rows,
            headers=("Span", "F32 (ms)", "F16 (ms)", "F16 / F32"),
            tablefmt="simple",
        )
    )
    print()


def _print_qmatmul_breakdown(forwards: dict[str, ForwardMetrics]) -> None:
    operations = sorted(
        {
            dict(args).get("operation")
            for forward in forwards.values()
            for (name, args) in forward.spans
            if name == "qmatmul"
        }
        - {None}
    )
    rows = []
    for operation in operations:
        metrics = {
            dtype: forward.aggregate("qmatmul", operation=operation)
            for dtype, forward in forwards.items()
        }
        rows.append(
            (
                operation,
                metrics["F32"].count,
                _milliseconds(metrics["F32"].total_microseconds),
                _milliseconds(metrics["F16"].total_microseconds),
                _ratio(
                    metrics["F16"].total_microseconds,
                    metrics["F32"].total_microseconds,
                ),
            )
        )
    print("Quantized matmul breakdown:")
    print(
        tabulate(
            rows,
            headers=(
                "Operation",
                "Count",
                "F32 (ms)",
                "F16 (ms)",
                "F16 / F32",
            ),
            tablefmt="simple",
        )
    )
    print()


def _print_attention_breakdown(forwards: dict[str, ForwardMetrics]) -> None:
    keys = sorted(
        {
            (name, args)
            for forward in forwards.values()
            for (name, args) in forward.spans
            if name.startswith("attn-")
        },
        key=lambda item: (item[0], item[1]),
    )
    rows = []
    for name, args_tuple in keys:
        args = dict(args_tuple)
        metrics = {
            dtype: forward.aggregate(name, **args)
            for dtype, forward in forwards.items()
        }
        rows.append(
            (
                name,
                ", ".join(f"{key}={value}" for key, value in args_tuple) or "-",
                metrics["F32"].count,
                _milliseconds(metrics["F32"].total_microseconds),
                _milliseconds(metrics["F16"].total_microseconds),
                _ratio(
                    metrics["F16"].total_microseconds,
                    metrics["F32"].total_microseconds,
                ),
            )
        )
    print("Attention breakdown:")
    print(
        tabulate(
            rows,
            headers=(
                "Span",
                "Arguments",
                "Count",
                "F32 (ms)",
                "F16 (ms)",
                "F16 / F32",
            ),
            tablefmt="simple",
        )
    )
    print()


def _normalize_arg(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _milliseconds(microseconds: float) -> str:
    return f"{microseconds / 1_000:.3f}"


def _ratio(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "-"
    return f"{numerator / denominator:.2f}x"


if __name__ == "__main__":
    raise SystemExit(main())
