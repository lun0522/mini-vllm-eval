"""Benchmark repeated requests with paged prefix caching."""

from __future__ import annotations

from typing import Any

from loguru import logger

from benchmarks.base import Benchmark
from proto_loader import ProtoModules


PROMPT = (
    "Explain in detail how continuous batching improves throughput in an LLM "
    "inference server. Compare it with static batching, describe how requests "
    "enter and leave a running batch, and discuss the key scheduling and KV-cache "
    "challenges an implementation must handle."
)


class PrefixCachingBenchmark(Benchmark):
    def features(self) -> list[str]:
        return ["metal"]

    def server_flags(self) -> list[str]:
        return ["--kv-cache-type", "paged-prefix:16"]

    def run_benchmark(self, client: Any, proto: ProtoModules) -> None:
        request = proto.request_handler.GenerateText(
            prompt=PROMPT,
            max_new_tokens=1024,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
        )
        for request_number in range(1, 3):
            logger.info("Sending prefix-caching request {} of 2", request_number)
            streamed_word_count = 0
            next_word_milestone = 100
            for response in client.GenerateText(request):
                event = response.WhichOneof("event")
                if event == "text":
                    streamed_word_count += len(response.text.split())
                    while streamed_word_count >= next_word_milestone:
                        logger.info("Received {} streamed words", next_word_milestone)
                        next_word_milestone += 100
                elif event == "stats":
                    stats = response.stats
                    logger.info(
                        "Generated {} tokens (prefill: {} tokens/s, decode: {} tokens/s)",
                        stats.output_token_count,
                        self.format_tokens_per_second(
                            stats.input_token_count,
                            stats.prefill_duration_microseconds,
                        ),
                        self.format_tokens_per_second(
                            max(stats.output_token_count - 1, 0),
                            stats.decode_duration_microseconds,
                        ),
                    )
