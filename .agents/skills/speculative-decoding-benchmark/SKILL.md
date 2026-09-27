---
name: speculative-decoding-benchmark
description: Run and interpret the mini-vllm speculative-decoding policy benchmark using repeated untraced samples. Use when validating or comparing fixed, acceptance-rate, accepted-length, and target-only decoding performance.
---

# Speculative Decoding Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark owns models,
prompts, policy parameters, warm-up, output length, validation, and reporting;
do not duplicate or override them in the measurement workflow.

## Smoke Test

After changing the benchmark or speculative-decoding implementation, run one
complete invocation to verify all cases and workloads:

```shell
.venv/bin/python3 main.py --benchmark speculative_decoding
```

Treat this as a functional check, not a publishable performance result.

## Performance Measurement

Run three complete untraced invocations under comparable idle-system
conditions. Save each invocation to a separate log:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark speculative_decoding 2>&1 | tee BENCHMARK_LOG
```

Keep `pipefail` enabled so a benchmark failure is not hidden by a successful
`tee` process.

For every configuration and workload, calculate the mean and sample standard
deviation of decode throughput. If any decode-throughput coefficient of
variation exceeds 5%, collect two more complete runs and calculate results from
all five. Do not selectively discard valid runs.

Record the mini-vllm-rs and mini-vllm-eval revisions, hardware, operating
system, and relevant environment overrides with reported measurements.

## Interpretation

- Use per-workload decode throughput as the primary policy comparison.
- Retain E2E latency as the complete-request measure and TTFT as a diagnostic
  for target and draft prefill cost.
- Use accepted/proposed totals and selected-count histograms to explain policy
  behavior; do not impose acceptance or performance pass thresholds.
- Do not combine the code-refactoring and creative-writing workloads into one
  headline number. They intentionally represent different predictability.
- Do not require generated text to match across configurations. Different
  target execution shapes can cause small numerical differences in wording or
  formatting despite identical prompts and generation limits.

Keep repetition, aggregation, and log management outside the benchmark script.
