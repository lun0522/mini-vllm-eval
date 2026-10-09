---
name: prefix-cache-reuse-benchmark
description: Run and interpret repeated mini-vllm prefix-cache reuse benchmarks, including overlap, warm fan-out, cold-versus-warm publication, and working-set churn. Use for report-quality prefix-cache measurements or changes to this benchmark; do not use for incremental-publication or cache-aware-orchestration benchmarks.
---

# Prefix Cache Reuse Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark owns the model,
cache geometry, prompts, output lengths, concurrency, workload order, and
reporting. Do not override those choices during a measurement series.

## Workload Invariants

- Keep workload namespaces distinct so one workload cannot restore another's
  prefixes.
- Keep working-set churn last because it deliberately fills and evicts the
  cache.
- The KV cache must admit all four cold-burst requests simultaneously while
  remaining smaller than the eight-prefix churn working set.
- Overlap and churn use short outputs to emphasize prefill. Fan-out and burst
  use longer outputs so continuous batching remains observable.
- Compare ordinary paged caching with paged prefix caching using identical
  requests. Missing prefix-cache telemetry in the ordinary-paged case is
  expected.

## Execution Environment

Run release builds and every benchmark invocation, including smoke tests,
outside the sandbox from the outset. Request approval before the first such
command when required. Sandboxing can prevent Metal device discovery or process
inspection, so do not count a sandboxed sample or wait for it to fail before
escalating. Offline log analysis may run inside the sandbox.

Run samples sequentially on an otherwise idle system. Concurrent inference or
graphics workloads invalidate latency comparisons. Keep tracing disabled for
performance samples.

## Smoke Test

After changing the benchmark or prefix-cache implementation, run:

```shell
.venv/bin/python3 main.py --benchmark prefix_cache_reuse
```

Treat it as a functional check, not a performance sample. Confirm that the
cold burst restores zero tokens, the warm burst restores tokens, overlap reuse
increases with overlap, and churn causes eviction and re-indexing.

## Performance Measurement

Collect three sequential untraced runs and save each complete log:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark prefix_cache_reuse 2>&1 | tee BENCHMARK_LOG
```

Analyze all three logs deterministically:

```shell
.venv/bin/python3 \
  .agents/skills/prefix-cache-reuse-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG_1 BENCHMARK_LOG_2 BENCHMARK_LOG_3
```

The analyzer validates workload completeness and token-count consistency,
reports distributions and prefix-cache improvements, and checks headline
coefficients of variation. If it requests more samples, collect two additional
complete runs and analyze all five. Never selectively discard valid runs.

Record both repository revisions, hardware, operating system, and relevant
environment overrides with published results.

## Interpretation

- Overlap: relate restored-token coverage to prefill and TTFT reduction.
- Warm fan-out: report TTFT and maximum E2E across concurrency levels. Decode
  remains significant because these requests generate 128 tokens.
- Cold versus warm burst: use the cold phase to expose finish-time publication
  and the warm phase to measure completed-cache reuse. A cold request may report
  zero newly indexed tokens when an identical peer publishes first; it still
  performed duplicate prefill work.
- Churn: use restored/indexed tokens plus engine telemetry to establish
  eviction and re-indexing. A miss-heavy second cycle is expected when the
  working set exceeds capacity.
- Prefer TTFT and prefill for direct cache benefit. Present E2E separately
  because decode and continuous batching also influence it.

Keep repetition, aggregation, and log management outside the benchmark script.
