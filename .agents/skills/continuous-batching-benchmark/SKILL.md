---
name: continuous-batching-benchmark
description: >-
  Run and interpret the mini-vllm continuous-batching benchmark using repeated
  untraced samples. Use when measuring queued-request TTFT, aggregate
  throughput, or anchor-request latency under serial and continuously batched
  Qwen inference.
---

# Continuous Batching Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark owns the Qwen
target and draft models, fixed-4 speculation, request arrival sequence, prompts,
output lengths, scheduler settings, validation, and per-run reporting. Do not
override those choices in the measurement workflow.

## Execution Environment

Run release builds and every benchmark invocation, including smoke tests,
outside the sandbox from the outset. Request approval before the first such
command when required. Sandboxing can prevent Metal device discovery or process
inspection, so do not count a sandboxed sample or wait for it to fail before
escalating. Offline log analysis may run inside the sandbox.

Run samples sequentially on an otherwise idle system. Concurrent benchmark
processes compete for the same GPU and unified memory and invalidate latency and
throughput comparisons.

## Smoke Test

After changing the benchmark or continuous-batching implementation, run one
complete invocation:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark continuous_batching 2>&1 | tee BENCHMARK_LOG
```

Treat the smoke test as a functional check, not as one of the performance
samples.

## Performance Measurement

Run three complete untraced invocations under comparable idle-system
conditions, saving each invocation to a distinct log. Keep `pipefail` enabled
so `tee` cannot hide a benchmark failure:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark continuous_batching 2>&1 | tee BENCHMARK_LOG
```

Aggregate the three logs:

```shell
.venv/bin/python3 \
  .agents/skills/continuous-batching-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG_1 BENCHMARK_LOG_2 BENCHMARK_LOG_3
```

The analyzer reports means, sample standard deviations, serial-to-batched
comparisons, and coefficients of variation for headline measurements. If any
headline coefficient of variation exceeds 5%, collect two additional complete
runs and analyze all five logs together. Do not selectively discard valid
runs.

Record the mini-vllm-rs and mini-vllm-eval revisions, hardware, operating
system, and relevant environment overrides with reported measurements.

## Interpretation

- For the two-request handoff, use follower TTFT reduction as the primary
  result. It measures how quickly a request arriving during decode begins
  producing output.
- For the serial heterogeneous case, use the follower mean and maximum for the
  comparison. Concurrent input preprocessing can change backend arrival order,
  so individual follower roles are diagnostic rather than matched queue
  positions.
- In the continuously batched heterogeneous case, also inspect individual
  short- and long-prefill follower rows so the average does not hide different
  prefill behavior.
- Use aggregate output throughput to measure total work completed, and anchor
  E2E change to show the latency paid by the request already decoding. Present
  them together; continuous batching is a latency-throughput tradeoff.
- Use fixed-4 accepted/proposed totals only as a diagnostic for differences in
  generated work. Do not impose an acceptance threshold.
- Do not require generated text to match across active-request limits.
  Different batch shapes can cause small numerical differences despite
  identical prompts and generation limits.
- Do not combine the two workloads into one headline number or treat a single
  invocation as a performance conclusion.

Keep repetition, aggregation, and log management outside the benchmark script.
