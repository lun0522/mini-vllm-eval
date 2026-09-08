"""Shared benchmark lifecycle."""

from __future__ import annotations

import subprocess
from abc import ABC
from abc import abstractmethod
from typing import Any

from loguru import logger

from mini_vllm import CONTROL_SOCKET
from mini_vllm import REQUEST_SOCKET
from mini_vllm import create_channel
from mini_vllm import send_shutdown
from mini_vllm import wait_for_server
from proto_loader import ProtoModules


class Benchmark(ABC):
    @staticmethod
    def format_tokens_per_second(token_count: int, duration_us: int) -> str:
        if duration_us == 0:
            return "unavailable"
        return f"{token_count * 1_000_000 / duration_us:.2f}"

    def build_server_command(self) -> list[str]:
        command = ["cargo", "run", "--release"]
        features = self.features()
        if features:
            command.extend(["--features", ",".join(features)])
        command.extend(
            [
                "--",
                *self.server_flags(),
                "--control-socket",
                str(CONTROL_SOCKET),
                "--request-socket",
                str(REQUEST_SOCKET),
            ]
        )
        return command

    def run(self, process: subprocess.Popen[bytes], proto: ProtoModules) -> None:
        with create_channel(REQUEST_SOCKET) as channel:
            wait_for_server(process, channel)
            client = proto.request_handler_grpc.RequestHandlerServiceStub(channel)
            try:
                self.run_benchmark(client, proto)
            finally:
                logger.info("Benchmark finished; stopping mini-vllm-rs")
                send_shutdown(proto)

    @abstractmethod
    def features(self) -> list[str]:
        pass

    @abstractmethod
    def server_flags(self) -> list[str]:
        pass

    @abstractmethod
    def run_benchmark(self, client: Any, proto: ProtoModules) -> None:
        pass
