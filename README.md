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

## Run

Run the prefix-caching benchmark:

```shell
python3 main.py --benchmark prefix_caching
```

The benchmark sends the same request twice to demonstrate prefix reuse. It
reports progress after every 100 streamed words without printing the generated
text. The launcher generates its Python gRPC clients from `mini-vllm-rs` at
startup, keeping the Rust project as the single source of truth. It shuts down
mini-vllm when the benchmark finishes.

Press Ctrl-C to stop the benchmark early and gracefully shut down mini-vllm.

## Add a benchmark

Create a `Benchmark` subclass under `benchmarks/` and override:

- `features()` to return the Cargo features needed by the server.
- `server_flags()` to return additional mini-vllm-rs command-line arguments.
- `run_benchmark()` to implement the benchmark using the ready request-handler
  client and generated protobuf modules.

Add an instance of the subclass to `BENCHMARKS` in `benchmarks/__init__.py`. The
base class handles server startup arguments, connection setup, and shutdown.

When finished working in the project, leave the virtual environment:

```shell
deactivate
```
