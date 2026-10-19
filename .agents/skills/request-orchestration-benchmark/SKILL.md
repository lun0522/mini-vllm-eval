---
name: request-orchestration-benchmark
description: >-
  Run and interpret the mini-vllm mixed CPU/GPU request-orchestration benchmark.
  Use when comparing routing policies, cache affinity, backend distribution, or
  fallback versus workload-specific throughput profiles.
---

# Request Orchestration Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark owns the Qwen2.5
0.5B model, mixed CPU/GPU backends, F32 activation dtype, paged prefix cache,
scheduler settings, request shapes, prefix seed, concurrent arrival pattern,
validation, and reporting. Do not override those choices within one result set.

The benchmark runs seven server configurations for two related comparisons:

- The policy comparison covers round robin, least requests, least estimated
  work, most cached prefix, and cache-aware least estimated work. The two
  estimated-work policies use the complete measured throughput-profile matrix.
- The profile comparison reruns the two estimated-work policies with the single
  fallback profile. It reuses their detailed-profile results from the policy
  comparison rather than running duplicate cases.

Each case first completes an unmeasured request that seeds one backend with the
long shared prefix. It then submits ten measured requests concurrently: shared
and unique variants of five input/output shapes. Backend 1 is CPU and backend 2
is GPU in mixed mode.

## Execution Environment

Run release benchmark invocations outside the sandbox from the outset. Request
approval before the first such command when required. Sandboxing can prevent
Metal device discovery or process inspection. Offline log analysis may run
inside the sandbox.

Run samples sequentially on an otherwise idle system. CPU and GPU share memory
bandwidth and thermal headroom, so concurrent benchmark processes invalidate
the comparisons. Do not enable Chrome tracing for performance samples.

## Smoke Test

After changing the benchmark or routing implementation, run one complete
invocation and treat it only as a functional check:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark request_orchestration 2>&1 | tee BENCHMARK_LOG
```

## Performance Measurement

Run three complete untraced invocations under comparable idle-system
conditions. Save each invocation to a distinct log and keep `pipefail` enabled
so `tee` cannot hide a benchmark failure:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark request_orchestration 2>&1 | tee BENCHMARK_LOG
```

Aggregate the three logs:

```shell
.venv/bin/python3 \
  .agents/skills/request-orchestration-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG_1 BENCHMARK_LOG_2 BENCHMARK_LOG_3
```

If the analyzer reports a headline coefficient of variation above 5%, collect
two additional complete invocations and analyze all five. Do not selectively
discard valid runs. Record both repository revisions, hardware, operating
system, and relevant environment overrides with reported measurements.

## Interpretation

- Use wall time and aggregate output throughput as the primary capacity
  measurements. Read median and p95 TTFT and E2E latency alongside them so an
  average improvement does not hide a slow tail.
- Treat round robin as a simple distribution baseline, not an expected winner.
  Least requests sees load but not heterogeneous throughput; most cached prefix
  sees affinity but not load. The estimated-work policies model heterogeneous
  throughput, and only the cache-aware variant combines that estimate with
  affinity.
- Compare detailed and fallback profiles within each estimated-work policy.
  Their paired changes isolate profile granularity from policy behavior.
- Inspect CPU/GPU request and token distributions before explaining throughput
  differences. A policy can appear fast simply because it avoided the CPU, or
  can preserve CPU capacity for work where the GPU is substantially faster.
- Compare shared-prefix restoration with unique-prompt restoration. Unique
  prompts can still restore a small common chat-template prefix, so do not
  require their restored count to be zero.
- The seed is intentionally routed by the policy itself. Cache placement and
  subsequent affinity are therefore part of the observed policy behavior.
- Backend choices may vary across repeated concurrent runs. Use the analyzer's
  per-request CPU selection frequency and routing-instability count rather than
  discarding runs whose assignments differ.
- Do not treat a single invocation as a performance conclusion.
