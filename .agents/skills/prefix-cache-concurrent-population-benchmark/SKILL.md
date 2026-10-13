---
name: prefix-cache-concurrent-population-benchmark
description: Run and interpret repeated mini-vllm concurrent prefix-cache population benchmarks. Use for report-quality measurements or changes to the prefix_cache_concurrent_population benchmark; do not use for general prefix reuse or cache-aware admission benchmarks.
---

# Prefix Cache Single-Flight Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark crosses completion
and incremental prefix publication with independent and exact-input
single-flight coordination. Do not change its model, cache geometry, prompts,
concurrency, or output lengths within a measurement series.

## Workload Invariants

- Every identical-input burst uses a unique namespace, so it starts cold even
  though all workloads share one server process.
- Cold simultaneous bursts at concurrency 2/4/8 expose how duplicate prefill
  scales.
- Output lengths 16 and 128 expose the cost of making followers wait for the
  leader to finish decoding before they can reuse its cache.
- In-flight fan-out starts one 128-token leader and submits the remaining
  identical requests after its first streamed text. This is the publication
  timing workload: incremental publication can serve those followers while
  completion publication cannot.
- The distinct-input burst is a control: single-flight should not block requests
  whose inputs differ.
- Keep all clients connected until completion. Waiting-follower cancellation is
  outside this benchmark's scope.

## Execution

Run release builds and all benchmark invocations outside the sandbox from the
outset. Sandboxing can prevent Metal discovery and process inspection. Run on
an otherwise idle system with tracing disabled.

First run one smoke test:

```shell
.venv/bin/python3 main.py --benchmark prefix_cache_concurrent_population
```

Confirm that exact-input followers restore tokens with single-flight,
incremental publication increases restoration for independent in-flight
fan-out, and all distinct-input controls complete.

Then collect three sequential complete logs:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark prefix_cache_concurrent_population 2>&1 | tee BENCHMARK_LOG
```

Analyze all logs deterministically:

```shell
.venv/bin/python3 \
  .agents/skills/prefix-cache-concurrent-population-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG_1 BENCHMARK_LOG_2 BENCHMARK_LOG_3
```

If the analyzer requests more samples, collect two more full runs and analyze
all five. Do not selectively discard valid runs. Record both repository
revisions, hardware, operating system, and relevant environment overrides.
The analyzer reports variance from incremental-publication/independent cold
bursts, but does not use it to require more runs: their opportunistic reuse is
inherently sensitive to which concurrent request publishes first.

## Interpretation

- Use restored tokens and summed prefill time to show avoided duplicate prefill.
- Compare incremental against completion publication while holding coordination
  constant, especially for in-flight fan-out.
- Compare single-flight against independent execution while holding publication
  constant, especially for simultaneous cold bursts.
- Use mean TTFT, maximum E2E, and burst wall time to capture the follower wait.
- Compare short and long output variants explicitly: completion-gated followers
  wait through the leader's decode, so long output may offset prefill savings.
- The distinct-input control should remain close between policies; large changes
  indicate noise or an unintended scheduling interaction.
- Do not claim this benchmark covers partial-prefix coordination, warm-cache
  bypass, cancellation, or production-grade request coalescing. The first-text
  trigger establishes that prompt prefill completed; it does not measure reuse
  at arbitrary points within prefill.
