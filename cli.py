"""Command-line arguments for the mini-vllm evaluation launcher."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPOSITORY = Path(__file__).resolve().parent.parent / "mini-vllm-rs"
DEFAULT_PAGE_SIZE = 16


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the mini-vllm server.")
    parser.add_argument(
        "--repo_path",
        type=Path,
        default=DEFAULT_REPOSITORY,
        help=f"where mini-vllm-rs is located (default: {DEFAULT_REPOSITORY})",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="use the GPU for faster model inference",
    )
    parser.add_argument(
        "--cache_type",
        choices=("contiguous", "paged", "paged-prefix"),
        default="contiguous",
        help=(
            "how model memory is organized: contiguous, paged, or paged-prefix "
            "(default: contiguous)"
        ),
    )
    parser.add_argument(
        "--page_size",
        type=positive_integer,
        help=f"number of tokens stored in each cache page (default: {DEFAULT_PAGE_SIZE})",
    )
    args = parser.parse_args()
    if args.cache_type == "contiguous" and args.page_size is not None:
        parser.error("--page_size can only be used with a paged cache type")
    return args
