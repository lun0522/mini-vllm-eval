---
name: speculative-sampling-proposal-breakeven-benchmark
description: >-
  Run and interpret the focused mini-vllm speculative-sampling proposal-length
  break-even benchmark. Use when testing whether fixed proposals longer than
  four overcome target-only latency and selectively bracketing the required
  total-variation budget.
---

# Speculative Sampling Proposal Break-Even Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark owns the model
pair, creative-writing prompt, temperature 0.7, sampling parameters, three
seeds, 256-token output length, warm-up, hardware settings, and full-output
capture. Keep those choices fixed within a measurement series.

## Execution Environment

Run release builds and every benchmark invocation outside the sandbox from the
outset. Sandboxing can prevent Metal device discovery and process supervision,
and sandboxed timings are not valid performance samples. Run configurations
sequentially on an otherwise idle system because concurrent inference shares
GPU and unified-memory resources. Offline log analysis may run inside the
sandbox.

## Measurement Sequence

Run three independent untraced invocations for each stage. Save every
invocation to a distinct log with `pipefail` enabled.

Start with the endpoints:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark speculative_sampling_proposal_breakeven 2>&1 | tee BENCHMARK_LOG
```

This measures target-only, draft-only, and fixed-4, fixed-6, and fixed-8 at
TV=1. For proposal lengths that beat target-only or reach approximate parity,
measure TV=0 and TV=0.5, then add TV=0.25 or TV=0.75 only as needed to bracket
the crossover:

```shell
set -o pipefail
MINI_VLLM_SAMPLING_PROPOSAL_BREAKEVEN_CASES='6:0,6:0.5,8:0,8:0.5' \
  .venv/bin/python3 main.py \
  --benchmark speculative_sampling_proposal_breakeven 2>&1 | tee BENCHMARK_LOG
```

Do not run a full grid. Test fixed-5 or fixed-7 only when the measured boundary
between proposal lengths needs refinement. Do not assume latency is monotonic
in TV budget.

## Analysis

Pass all performance logs from the series to the analyzer. Each measured point
must occur in exactly three logs, but different stages may contain different
configuration sets:

```shell
.venv/bin/python3 \
  .agents/skills/speculative-sampling-proposal-breakeven-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOGS...
```

The analyzer uses the median of the three invocation-level medians. It calls a
point clearly faster or slower only when its invocation-median range does not
overlap the target-only range; overlapping ranges are approximately equal. If
any E2E or decode-throughput coefficient of variation exceeds 5%, collect two
additional samples for every point in the series and rerun with
`--samples-per-point 5`. Do not selectively discard valid runs.

Use E2E latency as the break-even measure. Report decode throughput separately
against target-only, along with pooled acceptance, accepted tokens per
verification round, actual output tokens per round, and verification count.
Draft-only is a standalone cost proxy, not isolated draft time inside the
speculative loop. Do not report phase-level draft or verification timings
unless reliable instrumentation is added.

The benchmark logs complete generated text. Keep output inspection separate
from headline performance measurement, and retain representative target-only,
lossless, near-break-even, and draft-only outputs. Treat them as qualitative
examples; one prompt cannot establish a safe quality threshold, and TV
distance does not measure semantic quality.

Record the exact commands, raw logs, repository revisions, hardware, operating
system, and environment overrides with the report.
