"""Generate one response and print its streamed text."""

from __future__ import annotations

import subprocess
from typing import Any

from loguru import logger

from benchmarks.base import Benchmark
from benchmarks.base import BenchmarkCase
from benchmarks.base import QWEN_0_5B_MODEL
from benchmarks.base import QWEN_7B_MODEL
from benchmarks.example_prompts import EXAMPLE_SHORT_PROMPT_1
from proto_loader import ProtoModules


MODEL_VARIATIONS = {
    "qwen_0_5b": (QWEN_0_5B_MODEL, None),
    "qwen_7b": (QWEN_7B_MODEL, None),
    "qwen_0_5b_draft_7b_target": (QWEN_7B_MODEL, QWEN_0_5B_MODEL),
}


class SimpleGenerationBenchmark(Benchmark):
    DEFAULT_MODEL_VARIATION = "qwen_0_5b"

    def __init__(self, model_variation: str = DEFAULT_MODEL_VARIATION) -> None:
        try:
            self.target_model, self.draft_model = MODEL_VARIATIONS[model_variation]
        except KeyError as error:
            supported = ", ".join(MODEL_VARIATIONS)
            raise ValueError(
                f"unsupported model variation {model_variation!r}; expected one of: {supported}"
            ) from error

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
        request = proto.request_handler.GenerateText(
            prompt=EXAMPLE_SHORT_PROMPT_1,
            max_new_tokens=1024,
            temperature=0.7,
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

    def report_results(
        self,
        _results: list[tuple[BenchmarkCase, Any]],
    ) -> None:
        pass
