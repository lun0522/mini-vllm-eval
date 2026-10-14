---
name: prefix-cache-concurrent-population-benchmark
description: Run and interpret repeated mini-vllm concurrent prefix-cache population benchmarks. Use for report-quality measurements or changes to the prefix_cache_concurrent_population benchmark; do not use for general prefix reuse or cache-aware admission benchmarks.
---

# Prefix Cache Concurrent-Population Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark crosses completion
and incremental prefix publication with independent execution and exact-input
single-flight released at either request completion or prefix publication. Do
not change its model, cache geometry, prompts, concurrency, or output lengths
within a measurement series.

## Workload Invariants

- Every identical-input burst uses a unique namespace, so it starts cold even
  though all workloads share one server process.
- Cold simultaneous bursts at concurrency 2/4/8 expose how duplicate prefill
  scales.
- Output lengths 16 and 128 expose the cost of making followers wait for the
  leader to finish decoding, and whether prefix-publication release avoids it.
- Prefix-publication release forces the leader's completed prefix to be
  published even with the on-request-completion publication policy. In that
  pairing, the publication-policy comparison only measures additional
  incremental checkpoints after the release checkpoint.
- In-flight fan-out starts one 128-token leader and submits the remaining
  identical requests after its first streamed text. This is the publication
  timing workload: incremental publication can serve those followers while
  completion publication cannot. Under prefix-publication release, these are
  warm follow-on requests rather than followers of the original flight; use the
  simultaneous cold bursts to compare the two follower-release points.
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

Confirm that exact-input followers restore tokens with both single-flight
release policies, prefix-publication release starts followers before the
leader completes, incremental publication increases restoration for
independent in-flight fan-out, and all distinct-input controls complete.

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
- Compare configured incremental against completion publication while holding
  coordination constant, especially for in-flight fan-out. Account for the
  required release checkpoint under prefix-publication coordination.
- Compare both single-flight release points against independent execution while
  holding publication constant, especially for simultaneous cold bursts.
- Compare prefix-publication release against request-completion release while
  holding publication constant. The former should retain duplicate-prefill
  avoidance without making followers wait through the leader's decode.
- Use mean TTFT, maximum E2E, and burst wall time to capture the follower wait.
- Compare short and long output variants explicitly: request-completion-gated
  followers wait through the leader's decode, so long output may offset prefill
  savings; prefix-publication-gated followers should not pay that delay.
- The distinct-input control should remain close between policies; large changes
  indicate noise or an unintended scheduling interaction.
- Do not claim this benchmark covers partial-prefix coordination, warm-cache
  bypass, cancellation, or production-grade request coalescing. The first-text
  trigger establishes that prompt prefill completed; it does not measure reuse
  at arbitrary points within prefill.
