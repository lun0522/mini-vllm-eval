"""Benchmark two concurrently submitted generation requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from benchmarks.base import Benchmark
from benchmarks.base import EXAMPLE_PROMPT_1
from benchmarks.base import EXAMPLE_PROMPT_2
from benchmarks.base import QWEN_LARGE_MODEL
from benchmarks.base import QWEN_SMALL_MODEL
from proto_loader import ProtoModules


class ConcurrentRequestsBenchmark(Benchmark):
    def server_flags(self) -> list[str]:
        return [
            "--model",
            QWEN_LARGE_MODEL,
            "--draft-model",
            QWEN_SMALL_MODEL,
        ]

    def run_benchmark(self, client: Any, proto: ProtoModules) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    self._send_request,
                    client,
                    proto,
                    request_number,
                    prompt,
                )
                for request_number, prompt in enumerate(
                    (EXAMPLE_PROMPT_1, EXAMPLE_PROMPT_2), start=1
                )
            ]
            for future in futures:
                future.result()

    def _send_request(
        self,
        client: Any,
        proto: ProtoModules,
        request_number: int,
        prompt: str,
    ) -> None:
        request = proto.request_handler.GenerateText(
            prompt=prompt,
            max_new_tokens=128,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=True,
            ignore_eos_tokens=True,
        )
        logger.info("Sending concurrent request {}", request_number)
        for response in client.GenerateText(request):
            if response.WhichOneof("event") == "stats":
                logger.info("Concurrent request {} finished", request_number)
                self.print_generation_stats(response.stats)
