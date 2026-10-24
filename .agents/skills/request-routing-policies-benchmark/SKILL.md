---
name: request-routing-policies-benchmark
description: >-
  Run and interpret the mini-vllm mixed CPU/GPU request-routing policy benchmark.
  Use when comparing routing policies, cache affinity, backend distribution, or
  fallback versus workload-specific throughput profiles.
---

# Request Routing Policies Benchmark

Run from the `mini-vllm-eval` repository root. The benchmark owns the Qwen2.5
0.5B model, mixed CPU/GPU backends, F32 activation dtype, paged prefix cache,
scheduler settings, request shapes, prefix seed, paced arrival pattern,
validation, and reporting. Do not override those choices within one result set.

The benchmark runs seven server configurations for two related comparisons:

- The policy comparison covers round robin, least requests, least estimated
  work, most cached prefix, and cache-aware least estimated work. The two
  estimated-work policies use the complete measured throughput-profile matrix.
- The profile comparison reruns the two estimated-work policies with the single
  fallback profile. It reuses their detailed-profile results from the policy
  comparison rather than running duplicate cases.

Each configuration runs two scenarios under two deterministic schedules, with a
fresh server per replay: 28 cases and 336 measured requests in a full invocation.

- `load`: twelve distinct prompts, without a prefix seed. Two 128/512 backlog
  requests lead eight 128/16 and two 512/64 requests, giving the CPU opportunities
  to absorb cheap work while the GPU is busy.
- `affinity`: a completed, unmeasured 1,024-token prefix seed, then two 128/512
  backlog requests, six 128/16 requests, and four 1024/16 cache probes. Three short
  requests and two probes match the seed; the others have distinct prompt bodies.
- Schedule B swaps adjacent positions in A. The request set is identical, but
  every request changes round-robin parity. Neither schedule strictly alternates
  shared and unique prompts. Report schedules separately; they are workload
  variants, not statistical repetitions.

Input lengths are approximate. The 1,024-token probes use the existing routing
profile selection, not a new calibration point. Unique prompts identify the
request near the beginning, preventing body-prefix reuse between measured
requests. Small common template-prefix hits remain possible.

Requests start 50 ms apart with one preprocessing worker; responses are consumed
concurrently. Backend 1 is CPU and backend 2 is GPU. This is a controlled replay,
not a simultaneous-burst test. Replacing long-context 1,024-token decodes with
short-context backlog requests limits runtime: provision roughly 15–20 minutes
per full invocation on the documented M2, pending the first updated diagnostic.
Do not launch repetitions automatically after that diagnostic.

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
RUST_LOG=info .venv/bin/python3 main.py \
  --benchmark request_routing_policies 2>&1 | tee BENCHMARK_LOG
```

For routing diagnostics, explicitly enable `RUST_LOG=info`; the Rust default
otherwise omits INFO-level routing decisions and candidate diagnostics. Check
early in the run that the log contains `Request routing candidate:` and
`Request routing:` entries. Inspect published operational state and outstanding
reserved load separately when explaining backend selection. Treat this as a
diagnostic run, not an additional sample in a quieter performance result set.

Validate the complete diagnostic log before collecting repetitions:

```shell
.venv/bin/python3 \
  .agents/skills/request-routing-policies-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG
```

The analyzer matches client workload labels to routing logs using the returned
`request_id`, then sorts decisions by `routing_epoch`. Logs are emitted outside
the routing lock, so neither log-line order nor request-ID order is authoritative.
It rejects missing/duplicate epochs, mismatched backend IDs, unexpected seed
metadata, and an unexpected workload order. A single diagnostic also prints
per-backend input, uncached-input, output and restored-token totals and the
slowest request. Client submission offsets reveal late submissions; investigate
warnings rather than silently excluding samples.
For both estimated-work policies, candidate logs must contain
`estimated_duration_ns`: the exact integer score used for selection, not measured
latency. The analyzer checks both candidates and minimum-score selection, and
reports the closest decision's selected estimate, alternative estimate and
absolute margin in seconds for each case. Equal scores are valid ties.
Spacing and one preprocessing worker reduce reordering but do
not guarantee it. If validation fails, investigate the replay rather than
increasing the sample count or silently discarding the run. Old simultaneous
burst and version-1 ten-request logs are exploratory results, not comparable
samples for this version-2 replay.

## Performance Measurement

Run three complete untraced invocations under comparable idle-system
conditions. Save each invocation to a distinct log and keep `pipefail` enabled
so `tee` cannot hide a benchmark failure:

```shell
set -o pipefail
RUST_LOG=info .venv/bin/python3 main.py \
  --benchmark request_routing_policies 2>&1 | tee BENCHMARK_LOG
```

Aggregate the three logs:

```shell
.venv/bin/python3 \
  .agents/skills/request-routing-policies-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG_1 BENCHMARK_LOG_2 BENCHMARK_LOG_3
```

If a headline coefficient of variation exceeds 5%, first inspect per-request
backend assignments, cache restoration, and system conditions. Do not
automatically run more samples to try to lower CV. Additional repetitions may
help quantify remaining variability after that investigation; the analyzer also
accepts five logs. Do not selectively discard valid runs. Record both repository
revisions, hardware, operating system, and relevant environment overrides with
reported measurements. Set and
record the same `RUST_LOG` filter for every sample in a comparison; do not mix
INFO diagnostic runs with quieter performance samples without acknowledging
the logging difference. Both commands above explicitly enable INFO logs so
routing order can be validated in every sample.

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
- Compare the identities of requests sent to each backend, not just request
  counts. One uncached long request on CPU can dominate wall time. Matching
  routing order controls the workload presented to each policy, but snapshot
  timing and worker progress can still change load-aware decisions.
- Compare shared-prefix restoration with unique-prompt restoration. Unique
  prompts can still restore a small common chat-template prefix, so do not
  require their restored count to be zero.
- The seed is intentionally routed by the policy itself. Cache placement and
  subsequent affinity are therefore part of the observed policy behavior, not
  an isolated routing-formula comparison with identical initial cache placement.
  Record the seed backend and indexed tokens from `REQUEST_ROUTING_POLICIES_SEED`.
- Compare load and affinity scenarios separately. A low CPU share is not itself
  a failure: inspect whether CPU could reduce completion time. Do not change
  measured profiles or require a particular policy to win to force a split.
- Inspect the logged `estimated_duration_ns` scores and analyzer's decision
  margins for differing decisions, rather than reconstructing rounded costs.
  Near-ties can flip
  with worker progress even when replay order matches. Cache-aware and load-only
  policies can legitimately select the same backend; cache savings matter only
  if they change the trade-off for this workload.
- With twelve requests, p95 is the maximum. Show the slowest request and backend,
  and do not interpret this as a stable population tail-latency estimate.
- Backend choices may vary across repeated paced runs. Use the analyzer's
  per-request CPU selection frequency and routing-instability count rather than
  discarding runs whose assignments differ.
- Do not treat a single invocation as a performance conclusion.
- Paced replay is not a strong test of the ingress-publication race. Keep burst
  reservation correctness covered by tests or a separate diagnostic burst run.
