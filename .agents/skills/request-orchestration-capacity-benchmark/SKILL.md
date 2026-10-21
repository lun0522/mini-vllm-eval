---
name: request-orchestration-capacity-benchmark
description: >-
  Run and interpret the mini-vllm mixed CPU/GPU orchestration-capacity
  benchmark. Use when locating the CPU-offload crossover or comparing mixed
  routing with an all-GPU control under increasing hot-prefix load.
---

# Request Orchestration Capacity Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark keeps the model,
request shape, shared cached prefix, 50 ms submission interval, one preprocessing
worker, scheduler, and throughput profiles fixed. Only measured request count
changes:

- `gpu_only`: four requests, which fit one GPU active batch.
- `crossover_boundary`: six requests, the last calibrated point before the
  policy begins offloading to CPU.
- `saturated`: 48 requests, which give the CPU useful work while the GPU is
  backlogged.

Each workload has a forced-GPU control and a mixed-routing case using
cache-aware least estimated work. Benchmark-only seed requests first place the
same approximately 512-token prefix in both backend caches. Backend 1 is CPU;
backend 2 is GPU.

## Execution

Run release benchmarks outside the sandbox when required because sandboxing can
prevent Metal discovery. Keep the system otherwise idle, run samples
sequentially, and do not enable Chrome tracing.

After benchmark or routing changes, collect one diagnostic run:

```shell
set -o pipefail
RUST_LOG=info .venv/bin/python3 main.py \
  --benchmark request_orchestration_capacity 2>&1 | tee BENCHMARK_LOG
```

Validate it before collecting repetitions:

```shell
.venv/bin/python3 \
  .agents/skills/request-orchestration-capacity-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG
```

The analyzer verifies the workload manifest, seeds, token counts, routing order,
backend IDs, estimated-duration minimum selection, and monotonically increasing
CPU use. A single log is exploratory and prints diagnostic results only.

For report-quality results, collect three complete invocations with identical
settings and distinct log files, then pass all three to the analyzer. It also
accepts five logs. If a headline coefficient of variation exceeds 5%, inspect
backend placement and system conditions before deciding whether two additional
runs are useful; do not automatically rerun or discard valid samples.

## Interpretation

- Use paired mixed-versus-forced-GPU changes in wall time and aggregate output
  throughput as the primary capacity result.
- Confirm backend assignments before attributing improvement to CPU capacity.
  The expected progression is no CPU work in the first two workloads and
  multiple CPU requests under saturation, but investigate rather than discard
  a valid run if a near-boundary decision changes.
- Treat `crossover_boundary` as a routing boundary, not proof that CPU and GPU
  have equal measured throughput. Inspect the analyzer's closest estimated
  decision margin.
- Compare latency distributions alongside capacity. Offloading can improve wall
  time while changing median or tail latency.
- All measured requests should restore the seeded prefix on either backend. A
  cache miss means the capacity comparison is no longer controlled.
- Do not generalize the calibrated request counts beyond the documented model,
  hardware, pacing, and throughput profiles without recalibration.
