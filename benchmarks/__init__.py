"""Available mini-vllm benchmarks."""

from benchmarks.continuous_batching import ContinuousBatchingBenchmark
from benchmarks.cpu_activation_dtype import CpuActivationDtypeBenchmark
from benchmarks.cpu_paged_attention import CpuPagedAttentionBenchmark
from benchmarks.prefix_caching import PrefixCachingBenchmark
from benchmarks.simple_generation import SimpleGenerationBenchmark
from benchmarks.speculative_decoding import SpeculativeDecodingBenchmark


BENCHMARKS = {
    "continuous_batching": ContinuousBatchingBenchmark(),
    "cpu_activation_dtype": CpuActivationDtypeBenchmark(),
    "cpu_paged_attention_f32": CpuPagedAttentionBenchmark("f32"),
    "cpu_paged_attention_f16": CpuPagedAttentionBenchmark("f16"),
    "prefix_caching": PrefixCachingBenchmark(),
    "simple_generation": SimpleGenerationBenchmark(),
    "speculative_decoding_qwen": SpeculativeDecodingBenchmark("qwen"),
    "speculative_decoding_llama": SpeculativeDecodingBenchmark("llama"),
}
