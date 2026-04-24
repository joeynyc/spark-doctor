from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class MetricSample(BaseModel):
    timestamp: datetime | None = None
    gpu_utilization_percent: float | None = None
    gpu_power_draw_watts: float | None = None
    gpu_clock_mhz: float | None = None
    gpu_memory_clock_mhz: float | None = None
    gpu_temperature_c: float | None = None


class MemorySnapshot(BaseModel):
    mem_total_kb: int | None = None
    mem_available_kb: int | None = None
    swap_total_kb: int | None = None
    swap_free_kb: int | None = None
    psi_memory: dict[str, Any] = Field(default_factory=dict)
    psi_io: dict[str, Any] = Field(default_factory=dict)


class ProcessInfo(BaseModel):
    pid: int
    command: str
    args: str = ""
    rss_kb: int | None = None
    cpu_percent: float | None = None
    mem_percent: float | None = None
    detected_backend: str | None = None


class CollectorStatus(BaseModel):
    name: str
    ok: bool
    errors: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    rule_id: str
    title: str
    severity: Literal["info", "warning", "critical"]
    confidence: Literal["low", "medium", "high"] = "medium"
    evidence: list[str] = Field(default_factory=list)
    explanation: str = ""
    recommended_actions: list[str] = Field(default_factory=list)
    escalation_actions: list[str] = Field(default_factory=list)
    source_note: str = ""


class ScanReport(BaseModel):
    schema_version: str = "0.1"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    spark_doctor_version: str = "0.1.0"
    anonymized: bool = True
    os: dict[str, Any] = Field(default_factory=dict)
    firmware: dict[str, Any] = Field(default_factory=dict)
    gpu: dict[str, Any] = Field(default_factory=dict)
    gpu_samples: list[MetricSample] = Field(default_factory=list)
    memory: MemorySnapshot | None = None
    docker: dict[str, Any] = Field(default_factory=dict)
    network: dict[str, Any] = Field(default_factory=dict)
    processes: list[ProcessInfo] = Field(default_factory=list)
    logs: dict[str, Any] = Field(default_factory=dict)
    collector_statuses: list[CollectorStatus] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    reproduction_notes: str | None = None
