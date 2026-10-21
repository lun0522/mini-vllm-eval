# mini-vllm-eval

## Setup

Create a virtual environment and install the dependencies:

```shell
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

When returning to the project later, activate the virtual environment first:

```shell
source .venv/bin/activate
```

When finished working in the project, leave the virtual environment:

```shell
deactivate
```

## Run

### `simple_generation`

Generate one response and print its streamed output:

```shell
python3 main.py --benchmark simple_generation
```

### `backend_throughput_profiles`

Measure isolated CPU and GPU execution throughput across the twelve
combinations of 128, 512, and 2048 input tokens with 16, 128, 512, and 1024
maximum output tokens using Qwen2.5 0.5B. Each combination has three measured
runs per backend. The benchmark reports prefill and decode timings and derives
request-orchestrator throughput profiles:

```shell
python3 main.py --benchmark backend_throughput_profiles
```

For repeated measurements and deterministic aggregation, follow the
[`backend-throughput-profile-benchmark`](.agents/skills/backend-throughput-profile-benchmark/SKILL.md)
skill.

### `prefix_cache_reuse`

Measure prefix-cache overlap, warm fan-out, cold-versus-warm bursts, and
working-set churn against an ordinary paged-cache baseline:

```shell
python3 main.py --benchmark prefix_cache_reuse
```

For report-quality repeated runs and deterministic result analysis, follow the
[`prefix-cache-reuse-benchmark`](.agents/skills/prefix-cache-reuse-benchmark/SKILL.md)
skill.

### `prefix_cache_concurrent_population`

Compare completion and incremental prefix publication with independent
execution and exact-input single-flight released at request completion or
prefix publication, using cold bursts and in-flight fan-out:

```shell
python3 main.py --benchmark prefix_cache_concurrent_population
```

For repeated measurement and deterministic analysis, follow the
[`prefix-cache-concurrent-population-benchmark`](.agents/skills/prefix-cache-concurrent-population-benchmark/SKILL.md)
skill.

### `continuous_batching`

Compare serial admission with continuous batching using two-request and
heterogeneous four-request workloads:

```shell
python3 main.py --benchmark continuous_batching
```

For repeated measurement and interpretation, use the
`continuous-batching-benchmark` skill.

### `scheduling_policies`

Compare first-come-first-served, shortest-prefill-first, and round-robin
scheduling with a decoding anchor and three mixed-length follower prefills:

```shell
python3 main.py --benchmark scheduling_policies
```

For repeated measurement and interpretation, use the
`scheduling-policy-benchmark` skill.

### `request_orchestration`

Compare all five mixed CPU/GPU request-routing policies with a concurrent,
heterogeneous workload containing unique and reusable-prefix requests. The
same suite also compares the single fallback throughput profile with the full
measured profile matrix for both estimated-work policies:

```shell
python3 main.py --benchmark request_orchestration
```

The benchmark runs seven server configurations across load-balancing and
cache-affinity scenarios, each with two fixed schedules: 28 fresh-server cases
and 336 measured requests. Requests start 50 ms apart with one preprocessing
worker; response consumption remains concurrent. The schedules use the same
requests but swap round-robin parity. Short-context backlog requests create
offloading opportunities without repeating the previous long-decode workload.
Results include wall time,
output throughput, latency percentiles, prefix-cache restoration, and
per-backend request and token distributions.
The analyzer validates the estimated-work policies' logged duration scores and
reports their closest decision margins, helping explain changes in placement.

For repeated measurements and deterministic aggregation, follow the
[`request-orchestration-benchmark`](.agents/skills/request-orchestration-benchmark/SKILL.md)
skill.

### `request_orchestration_capacity`

Compare forcing every measured request to the GPU with mixed CPU/GPU routing
under **cache-aware least estimated work**. Both cases run in mixed mode.
Unmeasured benchmark-only requests first place the same approximately 512-token
prefix in both backend caches. Three workloads then submit 4, 7, or 48 requests
that reuse the prefix and generate 128 tokens each. They represent a GPU-only
decision, the first calibrated CPU-offload point, and sustained load where CPU
capacity improves completion time:

```shell
python3 main.py --benchmark request_orchestration_capacity
```

The measured requests start 50 ms apart with one preprocessing worker. Both
cases otherwise use the same Qwen2.5 0.5B model, F32 activations, scheduler,
prefix-cache configuration, and measured throughput-profile matrix.

For repeated measurements and deterministic aggregation, follow the
[`request-orchestration-capacity-benchmark`](.agents/skills/request-orchestration-capacity-benchmark/SKILL.md)
skill.

### `speculative_decoding_qwen` and `speculative_decoding_llama`

Compare target-only decoding with fixed-4, acceptance-rate adaptive, and
accepted-length adaptive speculative decoding on two similarly sized workloads.
The Qwen benchmark uses Qwen2.5 7B and 0.5B; the Llama benchmark uses Llama 3.1
8B and Llama 3.2 1B. Both also run the draft model by itself as a standalone
cost proxy and report per-workload decode throughput, latency, acceptance
totals, and policy-selected draft-count histograms.

```shell
python3 main.py --benchmark speculative_decoding_qwen
# Or compare the Llama target and draft models with the same workload:
python3 main.py --benchmark speculative_decoding_llama
```

For repeated measurement and interpretation, use the
`speculative-decoding-benchmark` skill.

### `cpu_activation_dtype`

Compare three CPU activation configurations: F32, unoptimized F16, and F16
using quantized matmul via F32. Each case uses a short one-token warm-up, a
distinct short-prompt request producing one token, and a long-prompt request
producing 1024 tokens. This exposes both
small- and large-prefill behavior, long-context decoding, and memory use. All
cases allocate equal KV-cache token capacity by giving F16 half the F32 byte
budget.

```shell
python3 main.py --benchmark cpu_activation_dtype
```

To collect Chrome traces for all three configurations, provide a trace
directory. The benchmark writes each case beneath its own subdirectory:

```shell
python3 main.py \
  --benchmark cpu_activation_dtype \
  --trace-directory /tmp/mini-vllm-activation-traces
```

Compare the measured small prefill, large prefill, and final long-context decode
in the latest F32 and F16 traces:

```shell
python3 .agents/skills/activation-dtype-benchmark/scripts/activation_traces.py \
  /tmp/mini-vllm-activation-traces/f32/model-runner-TIMESTAMP.json \
  /tmp/mini-vllm-activation-traces/f16/model-runner-TIMESTAMP.json \
  /tmp/mini-vllm-activation-traces/f16-qmatmul/model-runner-TIMESTAMP.json
```

### `cpu_paged_attention_f32`

Compare contiguous and paged CPU attention with repeated-KV/grouped-Q and
concatenated/page-wise value matmul:

```shell
python3 main.py --benchmark cpu_paged_attention_f32
```

### `cpu_paged_attention_f16`

Test the same attention configurations with F16 activations:

```shell
python3 main.py --benchmark cpu_paged_attention_f16
```

The prefix-caching benchmark sends the same request twice to demonstrate prefix
reuse. It reports progress after every 100 streamed words without printing the
generated text. The launcher generates its Python gRPC clients from
`mini-vllm-rs` at startup, keeping the Rust project as the single source of
truth. It shuts down mini-vllm when the benchmark finishes.

Press Ctrl-C to stop the benchmark early and gracefully shut down mini-vllm.

## Add a benchmark

Create a `Benchmark` subclass under `benchmarks/` and override:

- `server_flags()` to return additional mini-vllm-rs command-line arguments.
- `run_benchmark()` to implement the benchmark using the ready request-handler
  client and generated protobuf modules.

Add an instance of the subclass to `BENCHMARKS` in `benchmarks/__init__.py`. The
base class handles server startup arguments, connection setup, and shutdown.
