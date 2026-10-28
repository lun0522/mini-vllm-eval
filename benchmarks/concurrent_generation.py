"""Generate responses for a concurrent batch of long prompts."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.example_prompts import EXAMPLE_LONG_PROMPTS
from benchmarks.simple_generation import MODEL_VARIATIONS
from proto_loader import ProtoModules


class ConcurrentGenerationBenchmark(Benchmark):
    DEFAULT_NUM_REQUESTS = 8
    DEFAULT_MODEL_VARIATION = "qwen_0_5b"

    def __init__(
        self,
        num_requests: int = DEFAULT_NUM_REQUESTS,
        model_variation: str = DEFAULT_MODEL_VARIATION,
    ) -> None:
        if num_requests < 1:
            raise ValueError("num_requests must be at least 1")
        try:
            self.target_model, self.draft_model = MODEL_VARIATIONS[model_variation]
        except KeyError as error:
            supported = ", ".join(MODEL_VARIATIONS)
            raise ValueError(
                f"unsupported model variation {model_variation!r}; expected one of: {supported}"
            ) from error
        self.num_requests = num_requests

    def server_flags(self) -> list[str]:
        flags = ["--model", self.target_model]
        if self.draft_model is not None:
            flags.extend(
                (
                    "--draft-model",
                    f"model {{ {self.draft_model} }} "
                    "token_count_policy { fixed { draft_token_count: 4 } }",
                )
            )
        flags.extend(
            (
                "--kv-cache-config",
                "kv_cache_type: KV_CACHE_TYPE_PAGED_PREFIX per_page_token_count: 16",
                "--inference-device",
                "gpu",
                "--inference-device",
                "gpu",
            )
        )
        return flags

    def run_benchmark(
        self,
        client: Any,
        proto: ProtoModules,
        _case: BenchmarkCase,
        _process: subprocess.Popen[bytes],
    ) -> None:
        prompts = [
            EXAMPLE_LONG_PROMPTS[index % len(EXAMPLE_LONG_PROMPTS)]
            for index in range(self.num_requests)
        ]
        logger.info("Sending {} requests concurrently", self.num_requests)
        with ThreadPoolExecutor(max_workers=self.num_requests) as executor:
            futures = [
                executor.submit(self._send_request, client, proto, prompt)
                for prompt in prompts
            ]
            outputs = [future.result() for future in futures]

        for index, (output, stats) in enumerate(outputs, start=1):
            logger.info(f"Response {index}:\n{output}")
            if stats is not None:
                self.print_generation_stats(stats)

    @staticmethod
    def _send_request(
        client: Any,
        proto: ProtoModules,
        prompt: str,
    ) -> tuple[str, Any | None]:
        request = proto.request_handler.GenerateText(
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.7,
            repeat_penalty=1.1,
            repeat_last_n=64,
            stream_output=False,
            ignore_eos_tokens=False,
        )
        output = ""
        stats = None
        for response in client.GenerateText(request):
            event = response.WhichOneof("event")
            if event == "text":
                output = response.text
            elif event == "stats":
                stats = response.stats
        return output, stats

    def report_results(
        self,
        _results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        pass
