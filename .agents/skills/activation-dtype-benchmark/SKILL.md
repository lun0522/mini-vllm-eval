---
name: activation-dtype-benchmark
description: Benchmark three mini-vllm CPU activation configurations using repeated untraced performance measurements and representative operation-level traces. Use when measuring activation-dtype performance or investigating F16 bottlenecks in mini-vllm.
---

# Activation Dtype Benchmark

Run experiments from the `mini-vllm-eval` repository root. Keep repetition orchestration outside the benchmark implementation.

## Overall Performance and Memory

Use untraced runs for TTFT, end-to-end latency, decode throughput, and RSS.
Run the benchmark five times, saving each invocation to a separate log:

```shell
.venv/bin/python3 main.py \
  --benchmark cpu_activation_dtype 2>&1 | tee BENCHMARK_LOG
```

Aggregate the five logs:

```shell
.venv/bin/python3 \
  .agents/skills/activation-dtype-benchmark/scripts/untraced_results.py \
  BENCHMARK_LOG...
```

Use the aggregate mean, sample standard deviation, and F16/F32 ratio for
headline comparisons. Treat small differences cautiously when distributions
overlap or an outlier dominates the mean. Run more than five samples only when
the results remain unstable.

## Operation-Level Diagnosis

Use one representative traced run to compare individual model spans. Tracing
is diagnostic and should not supply the headline latency, throughput, or RSS
measurements.

1. Choose a fresh trace directory supplied by the user or created for this run. Do not hardcode a personal path or reuse a directory containing JSON traces.
2. Run one `cpu_activation_dtype` benchmark invocation with tracing, saving its combined output to a log. The benchmark runs F32, unoptimized F16, and F16 with QMatMul via F32:

   ```shell
   .venv/bin/python3 main.py \
     --benchmark cpu_activation_dtype \
     --trace-directory TRACE_DIRECTORY 2>&1 | tee BENCHMARK_LOG
   ```

3. Identify the single new JSON file under each of `TRACE_DIRECTORY/f32`,
   `TRACE_DIRECTORY/f16`, and `TRACE_DIRECTORY/f16-qmatmul`. Keep all three
   traces until the analysis output has been saved successfully.
4. Analyze the small prefill, large prefill, and final long-context decode in
   those traces, saving the output for later documentation:

   ```shell
   .venv/bin/python3 \
     .agents/skills/activation-dtype-benchmark/scripts/activation_traces.py \
     F32_TRACE F16_TRACE F16_QMATMUL_TRACE | tee TRACE_ANALYSIS
   ```

5. Delete only the three exact trace files after analysis output has been saved
   successfully; retain them if analysis fails.

Report traced span timings as values from one representative run, without a
sample standard deviation. Repeat the traced run only when the first trace has
surprising results, visible interference, or differences too small to
interpret confidently. If another traced run is required, save and inspect its
analysis separately rather than aggregating traced runs into headline metrics.

Do not add repetition, aggregation, or deletion logic to the benchmark runner merely to execute this workflow. The human or agent invoking the benchmark owns those steps.
