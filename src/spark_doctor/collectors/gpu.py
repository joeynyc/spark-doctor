from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
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
        value = float(v.split()[0])
        return value if math.isfinite(value) else None
    except (ValueError, IndexError):
        return None


def _query_once(errors: list[str] | None = None) -> list[dict[str, Any]]:
    errors = errors if errors is not None else []
    fields = ",".join(QUERY_FIELDS)
    r = run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        timeout=8,
    )
    if not r.ok:
        errors.append(f"nvidia-smi CSV query: {r.error}: {r.stderr.strip()[:200]}")
        return []
    if not r.stdout.strip():
        errors.append("nvidia-smi CSV query: empty output")
        return []
    rows: list[dict[str, Any]] = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            errors.append("nvidia-smi CSV query: malformed row")
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
        if len(parts) != len(header):
            samples.append(MetricSample())
            continue
        row = dict(zip(header, parts))
        # Spark has one GPU; keep other device rows out of this time series.
        if row.get("gpu", "0") != "0":
            continue
        kwargs: dict[str, Any] = {}
        for dmon_col, field in DMON_FIELD_MAP.items():
            if dmon_col in row:
                kwargs[field] = _to_float(row[dmon_col])
        samples.append(MetricSample(**kwargs))
    end = datetime.now(timezone.utc)
    for i, sample in enumerate(samples):
        sample.timestamp = end - timedelta(seconds=len(samples) - i - 1)
    return samples


def _sample_via_dmon(count: int, errors: list[str]) -> list[MetricSample]:
    r = run(
        ["nvidia-smi", "dmon", "-s", "puc", "-c", str(count)],
        timeout=count + 10,
    )
    if not r.ok:
        errors.append(f"nvidia-smi dmon: {r.error}: {r.stderr.strip()[:200]}")
    samples = _parse_dmon(r.stdout)
    if len(samples) != count:
        errors.append(f"nvidia-smi dmon: received {len(samples)} of {count} requested samples")
    if any(not _complete_sample(s) for s in samples):
        errors.append("nvidia-smi dmon: samples contain missing or invalid utilization, power, clock, or temperature")
    return samples


def _sample_via_csv(count: int, errors: list[str]) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for i in range(count):
        if i > 0:
            time.sleep(1.0)
        rows = _query_once(errors)
        if not rows:
            samples.append(MetricSample(timestamp=datetime.now(timezone.utc)))
            continue
        r = rows[0]
        samples.append(
            MetricSample(
                timestamp=datetime.now(timezone.utc),
                gpu_utilization_percent=_to_float(r.get("utilization.gpu", "")),
                gpu_power_draw_watts=_to_float(r.get("power.draw", "")),
                gpu_clock_mhz=_to_float(r.get("clocks.current.graphics", "")),
                gpu_temperature_c=_to_float(r.get("temperature.gpu", "")),
            )
        )
        if not _complete_sample(samples[-1]):
            errors.append(f"nvidia-smi CSV sample {i + 1}: missing or invalid utilization, power, clock, or temperature")
    return samples


def _complete_sample(sample: MetricSample) -> bool:
    return all(value is not None for value in (
        sample.gpu_utilization_percent, sample.gpu_power_draw_watts,
        sample.gpu_clock_mhz, sample.gpu_temperature_c,
    ))


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
    initial = _query_once(status.errors)
    if initial:
        out["gpus"] = initial
        out["gpu_count"] = len(initial)
        first = initial[0]
        out["driver_version"] = first.get("driver_version")
        out["name"] = first.get("name")

    sample_count = max(1, int(sample_seconds))
    dmon_errors: list[str] = []
    samples = _sample_via_dmon(sample_count, dmon_errors)
    out["sampler"] = "dmon"
    if dmon_errors:
        csv_errors: list[str] = []
        csv_samples = _sample_via_csv(sample_count, csv_errors)
        if not csv_errors or sum(map(_complete_sample, csv_samples)) >= sum(map(_complete_sample, samples)):
            samples = csv_samples
            out["sampler"] = "csv"
        if csv_errors:
            status.errors.extend(dmon_errors)
            status.errors.extend(csv_errors)

    peak = peak_sample(samples)
    if peak:
        out["peak"] = {
            "gpu_utilization_percent": peak.gpu_utilization_percent,
            "gpu_power_draw_watts": peak.gpu_power_draw_watts,
            "gpu_clock_mhz": peak.gpu_clock_mhz,
            "gpu_temperature_c": peak.gpu_temperature_c,
        }

    status.ok = bool(initial or any(_complete_sample(s) for s in samples))
    return out, samples, status
