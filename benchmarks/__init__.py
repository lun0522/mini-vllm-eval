"""Available mini-vllm benchmarks."""

from benchmarks.backend_throughput_profiles import (
    BackendThroughputProfilesBenchmark,
)
from benchmarks.continuous_batching import ContinuousBatchingBenchmark
from benchmarks.cpu_activation_dtype import CpuActivationDtypeBenchmark
from benchmarks.cpu_paged_attention import CpuPagedAttentionBenchmark
from benchmarks.prefix_cache_reuse import PrefixCacheReuseBenchmark
from benchmarks.prefix_cache_concurrent_population import (
    PrefixCacheConcurrentPopulationBenchmark,
)
from benchmarks.request_orchestration import RequestOrchestrationBenchmark
from benchmarks.simple_generation import SimpleGenerationBenchmark
from benchmarks.scheduling_policies import SchedulingPoliciesBenchmark
from benchmarks.speculative_decoding import SpeculativeDecodingBenchmark


BENCHMARKS = {
    "backend_throughput_profiles": BackendThroughputProfilesBenchmark(),
    "continuous_batching": ContinuousBatchingBenchmark(),
    "cpu_activation_dtype": CpuActivationDtypeBenchmark(),
    "cpu_paged_attention_f32": CpuPagedAttentionBenchmark("f32"),
    "cpu_paged_attention_f16": CpuPagedAttentionBenchmark("f16"),
    "prefix_cache_reuse": PrefixCacheReuseBenchmark(),
    "prefix_cache_concurrent_population": PrefixCacheConcurrentPopulationBenchmark(),
    "request_orchestration": RequestOrchestrationBenchmark(),
    "simple_generation": SimpleGenerationBenchmark(),
    "scheduling_policies": SchedulingPoliciesBenchmark(),
    "speculative_decoding_qwen": SpeculativeDecodingBenchmark("qwen"),
    "speculative_decoding_llama": SpeculativeDecodingBenchmark("llama"),
}
