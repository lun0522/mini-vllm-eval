"""Available mini-vllm benchmarks."""

from benchmarks.concurrent_requests import ConcurrentRequestsBenchmark
from benchmarks.cpu_activation_dtype import CpuActivationDtypeBenchmark
from benchmarks.cpu_paged_attention import CpuPagedAttentionBenchmark
from benchmarks.prefix_caching import PrefixCachingBenchmark
from benchmarks.simple_generation import SimpleGenerationBenchmark


BENCHMARKS = {
    "concurrent_requests": ConcurrentRequestsBenchmark(),
    "cpu_activation_dtype": CpuActivationDtypeBenchmark(),
    "cpu_paged_attention_f32": CpuPagedAttentionBenchmark("f32"),
    "cpu_paged_attention_f16": CpuPagedAttentionBenchmark("f16"),
    "prefix_caching": PrefixCachingBenchmark(),
    "simple_generation": SimpleGenerationBenchmark(),
}
