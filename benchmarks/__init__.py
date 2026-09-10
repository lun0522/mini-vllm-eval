"""Available mini-vllm benchmarks."""

from benchmarks.concurrent_requests import ConcurrentRequestsBenchmark
from benchmarks.prefix_caching import PrefixCachingBenchmark
from benchmarks.simple_generation import SimpleGenerationBenchmark


BENCHMARKS = {
    "concurrent_requests": ConcurrentRequestsBenchmark(),
    "prefix_caching": PrefixCachingBenchmark(),
    "simple_generation": SimpleGenerationBenchmark(),
}
