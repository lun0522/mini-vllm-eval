---
name: scheduling-policy-benchmark
description: Run and interpret the mini-vllm FCFS, shortest-prefill-first, and round-robin scheduling benchmark using repeated untraced samples. Use when comparing scheduler fairness, follower TTFT, throughput, or anchor-request latency.
---

# Scheduling Policy Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark owns the Qwen
model, prompts, request arrival sequence, output lengths, scheduler budget,
validation, and reporting. Do not override those choices in the measurement
workflow.

The benchmark runs two targeted workloads for every policy. Both start a
short-prefill anchor and wait for it to begin decoding. They then submit one
long-prefill follower, wait briefly to establish its earlier arrival, and
submit the remaining followers behind it. A single input-preprocessing thread
preserves the oldest follower's position.

- `head-of-line` uses one long and one short follower. Use it to measure how
  much FCFS delays a short request behind an older multi-iteration prefill.
- `long-fairness` uses three different long followers. Use it to compare the
  low oldest-request TTFT provided by FCFS with the lower TTFT spread provided
  by round robin. Shortest-prefill-first orders these requests by remaining
  prompt length rather than arrival time.

The 128-token scheduling budget is slightly larger than a short prefill and
far smaller than a long prefill. This makes the selected prefill order visible
in TTFT without changing the shared decode-first behavior.

## Execution Environment

Run release builds and benchmark invocations, including smoke tests, outside
the sandbox from the outset. Request approval before the first such command
when required. Sandboxing can prevent Metal device discovery or process
inspection. Offline log analysis may run inside the sandbox.

Run samples sequentially on an otherwise idle system. Concurrent benchmark
processes compete for the same GPU and unified memory and invalidate latency
and throughput comparisons.

## Smoke Test

After changing the benchmark or scheduling implementation, run one complete
invocation:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark scheduling_policies 2>&1 | tee BENCHMARK_LOG
```

Treat the smoke test as a functional check, not as a performance sample.

## Performance Measurement

Run three complete untraced invocations under comparable idle-system
conditions. Save each invocation to a distinct log and keep `pipefail` enabled
so `tee` cannot hide a benchmark failure:

```shell
set -o pipefail
.venv/bin/python3 main.py \
  --benchmark scheduling_policies 2>&1 | tee BENCHMARK_LOG
```

Aggregate the three logs:

```shell
.venv/bin/python3 \
  .agents/skills/scheduling-policy-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG_1 BENCHMARK_LOG_2 BENCHMARK_LOG_3
```

If the analyzer reports a headline coefficient of variation above 5%, collect
two additional complete runs and analyze all five. Do not selectively discard
valid runs.

Record the mini-vllm-rs and mini-vllm-eval revisions, hardware, operating
system, and relevant environment overrides with reported measurements.

## Interpretation

- In `head-of-line`, compare the long and short follower TTFTs directly. FCFS
  should favor the older long request; shortest-prefill-first and round robin
  should allow the short follower to bypass or take the next turn.
- In `long-fairness`, use the oldest follower's TTFT to quantify FCFS's age
  preference and follower TTFT spread to quantify round robin's fairness. Also
  report the absolute maximum: a small spread is not useful if every request
  is uniformly slow.
- Use follower TTFT spread only as a supporting fairness diagnostic. A small
  spread can mean that all requests were uniformly slow, so interpret it with
  the absolute TTFT measurements.
- Report aggregate output throughput with anchor E2E latency. Round robin
  preserves decode priority and rotates which prefill can consume the remaining
  iteration budget; throughput should remain a secondary guardrail rather than
  the justification for a scheduling policy.
- Treat FCFS-relative changes as descriptive rather than pass thresholds. A
  scheduling policy is a workload-dependent latency-throughput tradeoff.
- Do not require generated text to match across policies. Different batch
  shapes can cause small numerical differences despite identical prompts and
  generation limits.
- Do not treat a single invocation as a performance conclusion.
