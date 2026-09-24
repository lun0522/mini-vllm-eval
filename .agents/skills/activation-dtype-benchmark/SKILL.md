---
name: activation-dtype-benchmark
description: Benchmark and analyze mini-vllm CPU F32 versus F16 activations, including repeated trace comparisons and memory measurements. Use when measuring activation-dtype performance or investigating F16 bottlenecks in mini-vllm.
---

# Activation Dtype Benchmark

Run experiments from the `mini-vllm-eval` repository root. Keep repetition orchestration outside the benchmark implementation.

For each traced repetition:

1. Choose a fresh trace directory supplied by the user or created for this run. Do not hardcode a personal path or reuse a directory containing JSON traces.
2. Run one `cpu_activation_dtype` benchmark invocation with tracing, saving its combined output to a log. Explicitly enable both CPU F16 optimizations so the experimental configuration is recorded in the command:

   ```shell
   MINI_VLLM_ENABLE_CPU_F16_QMATMUL_VIA_F32=true \
   MINI_VLLM_ENABLE_CPU_F16_CHUNKED_EMBEDDING_DEQUANTIZATION=true \
   .venv/bin/python3 main.py \
     --benchmark cpu_activation_dtype \
     --trace-directory TRACE_DIRECTORY 2>&1 | tee BENCHMARK_LOG
   ```

3. Identify the single new JSON file under each of `TRACE_DIRECTORY/f32` and `TRACE_DIRECTORY/f16`.
4. Analyze that pair with:

   ```shell
   .venv/bin/python3 \
     .agents/skills/activation-dtype-benchmark/scripts/activation_traces.py \
     F32_TRACE F16_TRACE
   ```

5. Record the benchmark and trace measurements in a compact result file:

   ```shell
   .venv/bin/python3 \
     .agents/skills/activation-dtype-benchmark/scripts/activation_results.py \
     record BENCHMARK_LOG F32_TRACE F16_TRACE RESULT_JSON
   ```

6. Delete only the two exact trace files after both analysis and result recording succeed; retain them if either step fails.
7. Repeat from step 1 when more samples are needed. Prefer three traced repetitions for bottleneck analysis. Run five or more untraced repetitions separately when a more stable throughput, latency, or RSS comparison is needed.

After all repetitions, aggregate the compact results:

```shell
.venv/bin/python3 \
  .agents/skills/activation-dtype-benchmark/scripts/activation_results.py \
  aggregate RESULT_JSON...
```

Report per-run values as needed and use the aggregate mean, sample standard deviation, and F16/F32 ratio for comparisons. Treat small differences cautiously when distributions overlap or an outlier dominates the mean.

Do not add repetition, aggregation, or deletion logic to the benchmark runner merely to execute this workflow. The human or agent invoking the benchmark owns those steps.
