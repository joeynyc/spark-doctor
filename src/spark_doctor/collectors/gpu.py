from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..models import CollectorStatus, MetricSample
from ..shell import run

QUERY_FIELDS = [
    "index",
    "name",
    "uuid",
    "driver_version",
    "temperature.gpu",
    "utilization.gpu",
    "power.draw",
    "clocks.current.graphics",
    "clocks.current.memory",
    "pstate",
]


def _to_float(v: str) -> float | None:
    v = v.strip()
    if not v or v.lower().startswith("[n/a]") or v == "N/A":
        return None
    try:
        return float(v.split()[0])
    except (ValueError, IndexError):
        return None


def _query_once() -> list[dict[str, Any]]:
    fields = ",".join(QUERY_FIELDS)
    r = run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        timeout=8,
    )
    if not r.ok or not r.stdout.strip():
        return []
    rows: list[dict[str, Any]] = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            continue
        row = dict(zip(QUERY_FIELDS, parts))
        rows.append(row)
    return rows


def collect_gpu(sample_seconds: int = 5) -> tuple[dict[str, Any], list[MetricSample], CollectorStatus]:
    status = CollectorStatus(name="gpu", ok=True)
    out: dict[str, Any] = {}
    samples: list[MetricSample] = []

    smi = run(["nvidia-smi"], timeout=8)
    if smi.error == "command_not_found":
        status.ok = False
        status.errors.append("nvidia-smi not found")
        out["available"] = False
        return out, samples, status

    if not smi.ok:
        status.ok = False
        status.errors.append(f"nvidia-smi failed: {smi.error}: {smi.stderr.strip()[:200]}")
        out["available"] = False
        return out, samples, status

    out["available"] = True
    initial = _query_once()
    if initial:
        out["gpus"] = initial
        out["gpu_count"] = len(initial)
        first = initial[0]
        out["driver_version"] = first.get("driver_version")
        out["name"] = first.get("name")

    sample_count = max(1, int(sample_seconds))
    for i in range(sample_count):
        if i > 0:
            time.sleep(1.0)
        rows = _query_once()
        if not rows:
            continue
        r = rows[0]
        samples.append(
            MetricSample(
                timestamp=datetime.utcnow(),
                gpu_utilization_percent=_to_float(r.get("utilization.gpu", "")),
                gpu_power_draw_watts=_to_float(r.get("power.draw", "")),
                gpu_clock_mhz=_to_float(r.get("clocks.current.graphics", "")),
                gpu_memory_clock_mhz=_to_float(r.get("clocks.current.memory", "")),
                gpu_temperature_c=_to_float(r.get("temperature.gpu", "")),
            )
        )

    pmon = run(["nvidia-smi", "pmon", "-c", "1"], timeout=5)
    if pmon.ok:
        out["pmon"] = pmon.stdout

    return out, samples, status
