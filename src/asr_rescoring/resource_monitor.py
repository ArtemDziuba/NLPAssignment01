"""Sample local inference-service RAM and device-wide GPU memory, with explicit scopes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def gpu_snapshot() -> list[dict[str, Any]]:
    """Device-wide memory includes other applications; it is not per-model allocation."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=True,
        )
        rows = []
        for line in result.stdout.strip().splitlines():
            index, name, used, total, utilization, driver = map(str.strip, line.split(","))
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "used_mib": float(used),
                    "total_mib": float(total),
                    "utilization_percent": float(utilization),
                    "driver": driver,
                }
            )
        return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


class ResourceMonitor:
    """RSS summed over Ollama/llama processes; shared pages can be counted repeatedly."""

    def __init__(self, path: Path, interval: float, gpu_interval: float) -> None:
        import psutil

        self.psutil = psutil
        self.path = path
        self.interval = interval
        self.gpu_interval = gpu_interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.baseline = self.sample()
        self.peak_service_rss = self.baseline["service_rss_bytes"]
        self.peak_client_rss = self.baseline["client_rss_bytes"]
        self.peak_system_used = self.baseline["system_used_bytes"]
        self.peak_gpu_mib = max((g["used_mib"] for g in gpu_snapshot()), default=0)
        self.processes: dict[int, str] = {}
        self.errors: list[str] = []
        self.sample_count = 0

    def sample(self) -> dict[str, Any]:
        processes = []
        for process in self.psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                name = (process.info["name"] or "").lower()
                if name.startswith(("ollama", "llama")):
                    processes.append(
                        {
                            "pid": process.pid,
                            "name": process.info["name"],
                            "rss_bytes": process.info["memory_info"].rss,
                        }
                    )
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        memory = self.psutil.virtual_memory()
        return {
            "monotonic_seconds": time.perf_counter(),
            "service_rss_bytes": sum(p["rss_bytes"] for p in processes),
            "client_rss_bytes": self.psutil.Process().memory_info().rss,
            "system_total_bytes": memory.total,
            "system_used_bytes": memory.total - memory.available,
            "processes": processes,
        }

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def _run(self) -> None:
        last_gpu = 0.0
        with self.path.open("a", encoding="utf-8") as stream:
            while not self.stop_event.is_set():
                try:
                    row = self.sample()
                    if time.perf_counter() - last_gpu >= self.gpu_interval:
                        row["gpus"] = gpu_snapshot()
                        last_gpu = time.perf_counter()
                        self.peak_gpu_mib = max(
                            [self.peak_gpu_mib, *(g["used_mib"] for g in row["gpus"])]
                        )
                    self.peak_service_rss = max(self.peak_service_rss, row["service_rss_bytes"])
                    self.peak_client_rss = max(self.peak_client_rss, row["client_rss_bytes"])
                    self.peak_system_used = max(self.peak_system_used, row["system_used_bytes"])
                    self.processes.update({p["pid"]: p["name"] for p in row["processes"]})
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    self.sample_count += 1
                except Exception as error:
                    self.errors.append(repr(error))
                self.stop_event.wait(self.interval)

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=10)
        return {
            "baseline": self.baseline,
            "peak_service_rss_bytes": self.peak_service_rss,
            "peak_client_rss_bytes": self.peak_client_rss,
            "peak_system_used_bytes": self.peak_system_used,
            "peak_device_gpu_used_mib": self.peak_gpu_mib,
            "processes": self.processes,
            "samples": self.sample_count,
            "errors": self.errors,
            "rss_scope": "sum of Ollama GUI, service and llama runner RSS; shared pages may repeat",
            "gpu_scope": "sampled device-wide usage including other applications; not model-only",
            "ram_interval_seconds": self.interval,
            "gpu_interval_seconds": self.gpu_interval,
        }
