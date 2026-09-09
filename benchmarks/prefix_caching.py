"""Benchmark repeated requests with paged prefix caching."""

from __future__ import annotations

from typing import Any

from loguru import logger

from benchmarks.base import Benchmark
from benchmarks.base import EXAMPLE_PROMPT_1
from benchmarks.base import QWEN_LARGE_MODEL
from benchmarks.base import QWEN_SMALL_MODEL
from proto_loader import ProtoModules


class PrefixCachingBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return [
            "--model",
            QWEN_LARGE_MODEL,
            "--draft-model",
            QWEN_SMALL_MODEL,
            "--kv-cache-type",
            "paged-prefix:16",
        ]

    def run_benchmark(self, client: Any, proto: ProtoModules) -> None:
        request = proto.request_handler.GenerateText(
            prompt=EXAMPLE_PROMPT_1,
            max_new_tokens=512,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        for request_number in range(1, 3):
            logger.info("Sending request {} of 2", request_number)
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
                    self.print_generation_stats(response.stats)
