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

Generate one response and print its streamed output:

```shell
python3 main.py --benchmark simple_generation
```

Run the prefix-caching benchmark:

```shell
python3 main.py --benchmark prefix_caching
```

Run two generation requests concurrently:

```shell
python3 main.py --benchmark concurrent_requests
```

Compare F32 and F16 CPU activations using a short one-token warm-up followed by
a long-prompt, 1024-token measured request. The two cases allocate equal
KV-cache token capacity by giving F16 half the F32 byte budget.

```shell
python3 main.py --benchmark cpu_activation_dtype
```

To collect Chrome traces for both cases, provide a trace directory. The
benchmark writes each case beneath separate `f32` and `f16` subdirectories:

```shell
python3 main.py \
  --benchmark cpu_activation_dtype \
  --trace-directory /tmp/mini-vllm-activation-traces
```

Compare the measured prefill and final long-context decode in the latest F32
and F16 traces:

```shell
python3 .agents/skills/activation-dtype-benchmark/scripts/activation_traces.py \
  /tmp/mini-vllm-activation-traces/f32/model-runner-TIMESTAMP.json \
  /tmp/mini-vllm-activation-traces/f16/model-runner-TIMESTAMP.json
```

Compare contiguous and paged CPU attention with repeated-KV/grouped-Q and
concatenated/page-wise value matmul:

```shell
python3 main.py --benchmark cpu_paged_attention
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
