---
name: cpu-paged-attention-benchmark
description: Benchmark and compare the six mini-vllm CPU attention implementations with F32 and F16 activations using repeated performance and memory measurements. Use when selecting or documenting CPU paged-attention settings for either activation dtype.
---

# CPU Paged Attention Benchmark

Run experiments from the `mini-vllm-eval` repository root. The benchmark covers
the same six attention configurations for F32 and F16. Keep repetition and
aggregation outside the benchmark implementation.

## Collect Results

Build the release binary before collecting samples so compilation does not add
noise to the workflow. This repository currently needs
`CARGO_UNSTABLE_NEXT_LOCKFILE_BUMP=1` to read its version-4 lockfile with the
configured Cargo toolchain; keep it in the environment because each benchmark
launch uses `cargo run`:

```shell
export CARGO_UNSTABLE_NEXT_LOCKFILE_BUMP=1
cargo build --release --manifest-path ../mini-vllm-rs/Cargo.toml
```

Run each activation dtype five times, saving every invocation to its own log:

```shell
.venv/bin/python3 main.py \
  --benchmark cpu_paged_attention_f32 2>&1 | tee F32_LOG

.venv/bin/python3 main.py \
  --benchmark cpu_paged_attention_f16 2>&1 | tee F16_LOG
```

Run cases sequentially. Parallel runs compete for CPU and memory bandwidth and
make both latency and RSS results unreliable. A failed or interrupted invocation
does not count as one of the five samples.

## Analyze Results

Aggregate the ten logs:

```shell
.venv/bin/python3 \
  .agents/skills/cpu-paged-attention-benchmark/scripts/untraced_results.py \
  --f32 F32_LOG_1 F32_LOG_2 F32_LOG_3 F32_LOG_4 F32_LOG_5 \
  --f16 F16_LOG_1 F16_LOG_2 F16_LOG_3 F16_LOG_4 F16_LOG_5
```

Use the reported mean and sample standard deviation for comparisons. Compare
attention configurations within each dtype before comparing F16 with F32:

- Small-prefill TTFT measures short-prompt overhead.
- Long-request TTFT measures large-prefill performance.
- Long-request decode throughput measures long-context decode performance.
- Peak RSS increase is the useful memory comparison; starting and absolute peak
  RSS can vary substantially between processes.

Treat small differences cautiously when distributions overlap or an outlier
dominates the mean. Run more than five samples only when the results remain
unstable. This workflow does not require Chrome traces; collect a trace only as
a separate diagnostic when aggregate results expose an unexplained regression.
