"""Available mini-vllm benchmarks."""

from benchmarks.prefix_caching import PrefixCachingBenchmark


BENCHMARKS = {
    "prefix_caching": PrefixCachingBenchmark(),
}
