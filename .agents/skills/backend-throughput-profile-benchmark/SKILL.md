---
name: backend-throughput-profile-benchmark
description: Measure and aggregate mini-vllm CPU/GPU throughput profiles for heterogeneous request-routing policies. Use when calibrating routing weights or remeasuring them after model, hardware, dtype, or inference-kernel changes.
---

# Backend Throughput Profile Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark owns the Qwen2.5
0.5B model, F32 activation dtype, paged cache, single-active-request scheduler,
and the twelve combinations of 128, 512, and 2048 input tokens with 16, 128,
512, and 1024 output tokens. It runs one warm-up followed by three measurements
for every shape on each backend, for 72 measured requests per invocation. Do
not change those settings while collecting samples for one profile set.

The benchmark reports prefill and decode separately. Prefill throughput divides
the input-token count by prefill time. Decode throughput divides one fewer than
the output-token count by decode time, because the first output token is sampled
at the end of prefill. Rates are rounded to integer tokens per second. Keeping
the phases separate avoids assuming that one prefill token and one decode token
have equal cost. The benchmark forces the requested output length and disables
prefix reuse so CPU and GPU measure the same work.

## Execution

Run release benchmarks sequentially on an otherwise idle system. CPU and GPU
samples must not overlap because they share memory bandwidth and thermal
headroom. Run benchmark invocations outside the sandbox when required for
Metal device discovery and process supervision.

After changing the benchmark or routing implementation, run one complete smoke
test. Treat it only as a functional check:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark backend_throughput_profiles 2>&1 | tee BENCHMARK_LOG
```

For performance measurement, run one clean untraced invocation and inspect the
within-workload CPU and GPU prefill/decode coefficients of variation. If every
phase value is at or below 5%, use the emitted profiles. If any value exceeds
5%, collect two more complete invocations under the same idle-system
conditions, saving each to a distinct log. Keep `pipefail` enabled so `tee`
cannot hide a benchmark failure. Aggregate the three logs with:

```shell
.venv/bin/python3 \
  .agents/skills/backend-throughput-profile-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG_1 BENCHMARK_LOG_2 BENCHMARK_LOG_3
```

If the analyzer still reports a maximum phase-duration coefficient of variation
above 5%, collect two more complete runs and analyze all five. Do not discard
valid samples selectively.

Use the analyzer's final phase-specific profile blocks together, in their
printed order. Their field names describe the intended data and may not match
the current mini-vllm-rs CLI schema until separate prefill/decode configuration
is implemented. The values are specific to the measured hardware, model,
activation dtype, cache layout, and inference kernels; rerun calibration when
any of those materially changes. Record both repository revisions and the
machine configuration with reported profiles.

Do not use Chrome tracing for the calibration samples. Tracing overhead changes
the values the routing policy is meant to consume.
