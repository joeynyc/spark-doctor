from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..models import CollectorStatus, MetricSample
from ..shell import run

# Note: clocks.current.memory omitted — returns [N/A] on GB10.
QUERY_FIELDS = [
    "index",
    "name",
    "uuid",
    "driver_version",
    "temperature.gpu",
    "utilization.gpu",
    "power.draw",
    "clocks.current.graphics",
    "pstate",
]

# dmon -s puc column header → MetricSample field.
DMON_FIELD_MAP = {
    "pwr": "gpu_power_draw_watts",
    "gtemp": "gpu_temperature_c",
    "sm": "gpu_utilization_percent",
    "pclk": "gpu_clock_mhz",
    "mclk": "gpu_memory_clock_mhz",
}


def _to_float(v: str) -> float | None:
    v = v.strip()
    if not v or v == "-" or v.lower().startswith("[n/a]") or v == "N/A":
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
        rows.append(dict(zip(QUERY_FIELDS, parts)))
    return rows


def _parse_dmon(text: str) -> list[MetricSample]:
    """Parse `nvidia-smi dmon -s puc` output. Header-aware."""
    header: list[str] | None = None
    samples: list[MetricSample] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            cols = stripped.lstrip("#").split()
            # First header row lists field names; second row lists units.
            # We want the row whose tokens look like field names (contains 'pwr' or 'sm').
            if any(c in cols for c in ("pwr", "sm", "gtemp")):
                header = cols
            continue
        if header is None:
            continue
        parts = stripped.split()
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        kwargs: dict[str, Any] = {"timestamp": datetime.utcnow()}
        for dmon_col, field in DMON_FIELD_MAP.items():
            if dmon_col in row:
                kwargs[field] = _to_float(row[dmon_col])
        samples.append(MetricSample(**kwargs))
    return samples


def _sample_via_dmon(count: int) -> list[MetricSample]:
    r = run(
        ["nvidia-smi", "dmon", "-s", "puc", "-c", str(count)],
        timeout=count + 10,
    )
    if not r.ok or not r.stdout.strip():
        return []
    return _parse_dmon(r.stdout)


def _sample_via_csv(count: int) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for i in range(count):
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
                gpu_temperature_c=_to_float(r.get("temperature.gpu", "")),
            )
        )
    return samples


def peak_sample(samples: list[MetricSample]) -> MetricSample | None:
    """Return the sample with the highest utilization (ties broken by power)."""
    loaded = [s for s in samples if s.gpu_utilization_percent is not None]
    if not loaded:
        return None
    return max(
        loaded,
        key=lambda s: (
            s.gpu_utilization_percent or 0,
            s.gpu_power_draw_watts or 0,
        ),
    )


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
    samples = _sample_via_dmon(sample_count)
    if not samples:
        samples = _sample_via_csv(sample_count)
        out["sampler"] = "csv"
    else:
        out["sampler"] = "dmon"

    peak = peak_sample(samples)
    if peak:
        out["peak"] = {
            "gpu_utilization_percent": peak.gpu_utilization_percent,
            "gpu_power_draw_watts": peak.gpu_power_draw_watts,
            "gpu_clock_mhz": peak.gpu_clock_mhz,
            "gpu_temperature_c": peak.gpu_temperature_c,
        }

    pmon = run(["nvidia-smi", "pmon", "-c", "1"], timeout=5)
    if pmon.ok:
        out["pmon"] = pmon.stdout

    return out, samples, status
