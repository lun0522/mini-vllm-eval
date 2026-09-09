"""Available mini-vllm benchmarks."""

from benchmarks.concurrent_requests import ConcurrentRequestsBenchmark
from benchmarks.prefix_caching import PrefixCachingBenchmark


BENCHMARKS = {
    "concurrent_requests": ConcurrentRequestsBenchmark(),
    "prefix_caching": PrefixCachingBenchmark(),
}
