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

Start mini-vllm on the CPU:

```shell
python3 launch_mini_vllm.py
```

Use the GPU:

```shell
python3 launch_mini_vllm.py --use_gpu
```

The launcher waits for mini-vllm to become ready, sends an example text
generation request, and streams the response and performance statistics to the
terminal. It generates its Python gRPC clients from the protocol definitions in
`mini-vllm-rs` at launch, keeping the Rust project as the single source of
truth. The server remains running afterward.

Press Ctrl-C to gracefully stop mini-vllm and all of its processes.

When finished working in the project, leave the virtual environment:

```shell
deactivate
```
