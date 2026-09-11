"""Generate one response and print its streamed text."""

from __future__ import annotations

from typing import Any

from loguru import logger

from benchmarks.base import Benchmark
from benchmarks.base import EXAMPLE_PROMPT_1
from benchmarks.base import QWEN_SMALL_MODEL
from proto_loader import ProtoModules


class SimpleGenerationBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return ["--model", QWEN_SMALL_MODEL]

    def run_benchmark(self, client: Any, proto: ProtoModules) -> None:
        request = proto.request_handler.GenerateText(
            prompt=EXAMPLE_PROMPT_1,
            max_new_tokens=1024,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=False,
        )
        logger.info("Sending request")
        for response in client.GenerateText(request):
            event = response.WhichOneof("event")
            if event == "text":
                print(response.text, end="", flush=True)
            elif event == "stats":
                self.print_generation_stats(response.stats)
