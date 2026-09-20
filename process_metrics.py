"""Reusable process metrics for benchmarks."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class RssMetrics:
    starting_rss_kib: int
    peak_rss_kib: int

    @property
    def peak_increase_kib(self) -> int:
        return self.peak_rss_kib - self.starting_rss_kib


class ProcessTreeRssSampler:
    def __init__(
        self,
        root_process_id: int,
        sampling_interval_seconds: float = 0.1,
    ) -> None:
        if sampling_interval_seconds <= 0:
            raise ValueError("RSS sampling interval must be positive")
        self._root_process_id = root_process_id
        self._sampling_interval_seconds = sampling_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._starting_rss_kib: int | None = None
        self._peak_rss_kib: int | None = None
        self._sampling_error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RSS sampler has already been started")
        self._starting_rss_kib = self._read_rss_kib()
        self._peak_rss_kib = self._starting_rss_kib
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="process-tree-rss-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> RssMetrics:
        if self._thread is None:
            raise RuntimeError("RSS sampler has not been started")
        self._stop_event.set()
        self._thread.join()
        if self._sampling_error is not None:
            raise RuntimeError("failed to sample process-tree RSS") from self._sampling_error

        self._record_sample(self._read_rss_kib())
        if self._starting_rss_kib is None or self._peak_rss_kib is None:
            raise RuntimeError("RSS sampler did not record any samples")
        return RssMetrics(
            starting_rss_kib=self._starting_rss_kib,
            peak_rss_kib=self._peak_rss_kib,
        )

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self._sampling_interval_seconds):
            try:
                self._record_sample(self._read_rss_kib())
            except Exception as error:
                self._sampling_error = error
                self._stop_event.set()

    def _record_sample(self, rss_kib: int) -> None:
        if self._peak_rss_kib is None or rss_kib > self._peak_rss_kib:
            self._peak_rss_kib = rss_kib

    def _read_rss_kib(self) -> int:
        try:
            root_process = psutil.Process(self._root_process_id)
            processes = (root_process, *root_process.children(recursive=True))
        except psutil.NoSuchProcess as error:
            raise RuntimeError(
                f"root process {self._root_process_id} is not running"
            ) from error

        rss_bytes = 0
        process_count = 0
        for process in processes:
            try:
                rss_bytes += process.memory_info().rss
                process_count += 1
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        if process_count == 0:
            raise RuntimeError(
                f"process tree rooted at {self._root_process_id} has no readable processes"
            )
        return rss_bytes // 1024
