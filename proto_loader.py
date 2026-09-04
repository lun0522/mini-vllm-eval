"""Generate and load Python clients from mini-vllm-rs protocol definitions."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

from grpc_tools import protoc


class ProtoModules(NamedTuple):
    main_process: ModuleType
    main_process_grpc: ModuleType
    request_handler: ModuleType
    request_handler_grpc: ModuleType


def generate_proto_modules(repository: Path, output_directory: Path) -> ProtoModules:
    proto_directory = repository / "proto"
    proto_files = [
        "main_process.proto",
        "model_runner.proto",
        "request_handler.proto",
    ]
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_directory}",
            f"--python_out={output_directory}",
            f"--grpc_python_out={output_directory}",
            *proto_files,
        ]
    )
    if result != 0:
        raise RuntimeError("Could not generate Python clients from mini-vllm-rs")

    sys.path.insert(0, str(output_directory))
    try:
        importlib.import_module("model_runner_pb2")
        return ProtoModules(
            main_process=importlib.import_module("main_process_pb2"),
            main_process_grpc=importlib.import_module("main_process_pb2_grpc"),
            request_handler=importlib.import_module("request_handler_pb2"),
            request_handler_grpc=importlib.import_module("request_handler_pb2_grpc"),
        )
    finally:
        sys.path.pop(0)
