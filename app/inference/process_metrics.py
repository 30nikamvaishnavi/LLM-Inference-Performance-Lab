"""Optional process-level metrics sampled during model generation."""

import importlib.util
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ProcessMetricsMonitor:
    """Sample RSS and process CPU during a timed operation if psutil is present."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._interval = interval_seconds
        self._process: Any | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss: int | None = None
        self._cpu_samples: list[float] = []

    def start(self) -> None:
        if importlib.util.find_spec("psutil") is None:
            return
        try:
            import psutil
        except (ImportError, OSError) as error:
            logger.info("Process metrics are unavailable (%s)", type(error).__name__)
            return

        self._process = psutil.Process()
        self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[int | None, float | None]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(0.1, self._interval * 4))
            self._sample()
            self._thread = None
        average_cpu = sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else None
        return self._peak_rss, average_cpu

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        if self._process is None:
            return
        try:
            self._peak_rss = max(
                self._peak_rss or 0,
                int(self._process.memory_info().rss),
            )
            cpu_percent = float(self._process.cpu_percent(interval=None))
            if cpu_percent > 0:
                self._cpu_samples.append(cpu_percent)
        except (OSError, RuntimeError):
            self._process = None
