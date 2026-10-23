---
name: speculative-sampling-tradeoff-benchmark
description: >-
  Run and interpret the mini-vllm speculative-sampling tradeoff benchmark using
  repeated untraced samples. Use when studying how temperature or a
  total-variation budget changes fixed-4 speculative decoding speed, draft
  acceptance, and creative-writing output.
---

# Speculative Sampling Tradeoff Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark owns the Qwen2.5
7B target, Qwen2.5 0.5B draft, creative-writing prompt, three seeds, output
length, sampling parameters, fixed draft-token count, warm-up, validation, and
reporting. Do not override those choices within a measurement series.

The benchmark has two related studies:

- The lossless temperature sweep holds the total-variation budget at zero and
  measures fixed-4 speculation at temperatures 0.2, 0.7, 1.0, and 1.6. Use it
  to show how the target and draft distributions' agreement changes with
  temperature. Do not assume acceptance must decline monotonically as
  temperature rises.
- The budget sweep holds temperature at 0.7 and measures budgets 0, 0.1, 0.3,
  0.5, 0.7, and 1.0. Use it to study the tradeoff between distributional
  fidelity and speed. Budget zero is lossless; budget one accepts every draft
  proposal under this algorithm.

The invocation contains 30 measured requests across target-only and fixed-4
server configurations, plus one warm-up request for each configuration.

## Execution Environment

Run release builds and benchmark invocations outside the sandbox from the
outset when required for Metal device discovery and process supervision. Run
samples sequentially on an otherwise idle system; concurrent inference shares
GPU and unified-memory resources and invalidates timing comparisons. Offline
log analysis may run inside the sandbox.

## Smoke Test

After changing the benchmark or speculative-sampling implementation, run one
complete invocation:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark speculative_sampling_tradeoff 2>&1 | tee BENCHMARK_LOG
```

Treat this as a functional check rather than a performance sample.

## Performance Measurement

Run three complete untraced invocations under comparable idle-system
conditions and save each to a distinct log. Keep `pipefail` enabled so `tee`
cannot hide a benchmark failure:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark speculative_sampling_tradeoff 2>&1 | tee BENCHMARK_LOG
```

Aggregate the logs with:

```shell
.venv/bin/python3 \
  .agents/skills/speculative-sampling-tradeoff-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG_1 BENCHMARK_LOG_2 BENCHMARK_LOG_3
```

The analyzer treats each complete invocation as one performance sample. It
averages the per-invocation medians, pools accepted/proposed token counts, and
shows the first run's seed-101 excerpts while reporting whether those excerpts
were identical across runs. If any median E2E or decode-throughput coefficient
of variation exceeds 5%, collect two additional complete runs and analyze all
five. Do not selectively discard valid runs.

Record the mini-vllm-rs and mini-vllm-eval revisions, hardware, operating
system, and relevant environment overrides with reported measurements.

## Interpretation

- Compare fixed-4 at temperature 0.7 and budget zero with target-only for the
  lossless speed baseline. Target-only was not measured at the other
  temperatures, so do not construct cross-temperature speedups against it.
- In the temperature sweep, use acceptance and tokens per verification round
  to explain changes in E2E latency and decode throughput.
- In the budget sweep, report latency or throughput together with pooled
  acceptance and the representative excerpts. Treat the excerpts as
  qualitative examples, not proof of output quality or a universal safe
  budget.
- A nonzero total-variation budget deliberately permits deviation from the
  target distribution. Describe this as a speed-versus-fidelity tradeoff; do
  not label sampled text correct or incorrect solely from its budget.
- Do not infer the optimal draft-token count from this benchmark. It fixes the
  count at four so temperature and budget remain isolated variables.
- Do not require seeded excerpts to match across repeated invocations. Report
  variation if it occurs; different execution shapes can produce small
  numerical differences that change multinomial samples.
- Do not treat a single invocation as a performance conclusion.
