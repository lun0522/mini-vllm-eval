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
from benchmarks.request_routing_policies import RequestRoutingPoliciesBenchmark
from benchmarks.request_orchestration_capacity import (
    RequestOrchestrationCapacityBenchmark,
)
from benchmarks.simple_generation import SimpleGenerationBenchmark
from benchmarks.scheduling_policies import SchedulingPoliciesBenchmark
from benchmarks.speculative_decoding import SpeculativeDecodingBenchmark
from benchmarks.speculative_sampling_tradeoff import (
    SpeculativeSamplingTradeoffBenchmark,
)
from benchmarks.speculative_sampling_proposal_breakeven import (
    SpeculativeSamplingProposalBreakevenBenchmark,
)


BENCHMARKS = {
    "backend_throughput_profiles": BackendThroughputProfilesBenchmark(),
    "continuous_batching": ContinuousBatchingBenchmark(),
    "cpu_activation_dtype": CpuActivationDtypeBenchmark(),
    "cpu_paged_attention_f32": CpuPagedAttentionBenchmark("f32"),
    "cpu_paged_attention_f16": CpuPagedAttentionBenchmark("f16"),
    "prefix_cache_reuse": PrefixCacheReuseBenchmark(),
    "prefix_cache_concurrent_population": PrefixCacheConcurrentPopulationBenchmark(),
    "request_routing_policies": RequestRoutingPoliciesBenchmark(),
    "request_orchestration_capacity": RequestOrchestrationCapacityBenchmark(),
    "simple_generation": SimpleGenerationBenchmark(),
    "scheduling_policies": SchedulingPoliciesBenchmark(),
    "speculative_decoding_qwen": SpeculativeDecodingBenchmark("qwen"),
    "speculative_decoding_llama": SpeculativeDecodingBenchmark("llama"),
    "speculative_sampling_tradeoff": SpeculativeSamplingTradeoffBenchmark(),
    "speculative_sampling_proposal_breakeven": (
        SpeculativeSamplingProposalBreakevenBenchmark()
    ),
}
