from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from . import __version__

PressureValue = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class MetricSample(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

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
    psi_memory: dict[str, dict[str, PressureValue]] = Field(default_factory=dict)
    psi_io: dict[str, dict[str, PressureValue]] = Field(default_factory=dict)


class ProcessInfo(BaseModel):
    pid: int
    ppid: int | None = None
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
    # Optional collectors feed rules opportunistically; their failures are reported but
    # do not make the scan incomplete.
    optional: bool = False


class Finding(BaseModel):
    rule_id: str
    title: str
    severity: Literal["info", "warning", "critical"]
    confidence: Literal["low", "medium", "high"] = "medium"
    evidence: list[str] = Field(default_factory=list)
    explanation: str = ""
    recommended_actions: list[str] = Field(default_factory=list)
    fix_commands: list[str] = Field(default_factory=list)
    escalation_actions: list[str] = Field(default_factory=list)
    source_note: str = ""


class LogEvidence(BaseModel):
    """One log line that matched a diagnostic signal. Only matched lines are kept so
    log-based findings can be re-derived after the raw logs are discarded."""

    source: Literal["kernel", "journal"]
    signal: Literal["oom_kill", "xid", "thermal", "kv_cache_oom"]
    line: str


class ScanReport(BaseModel):
    schema_version: str = "0.1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    spark_doctor_version: str = __version__
    anonymized: bool = True
    os: dict[str, Any] = Field(default_factory=dict)
    firmware: dict[str, Any] = Field(default_factory=dict)
    gpu: dict[str, Any] = Field(default_factory=dict)
    gpu_samples: list[MetricSample] = Field(default_factory=list)
    memory: MemorySnapshot | None = None
    cuda_env: dict[str, Any] = Field(default_factory=dict)
    docker: dict[str, Any] = Field(default_factory=dict)
    network: dict[str, Any] = Field(default_factory=dict)
    processes: list[ProcessInfo] = Field(default_factory=list)
    logs: dict[str, str] = Field(default_factory=dict)
    # None marks a scan that predates evidence capture or skipped logs; [] means logs
    # were read and nothing matched.
    log_evidence: list[LogEvidence] | None = None
    collector_statuses: list[CollectorStatus] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    reproduction_notes: str | None = None

    @field_validator("os", "gpu", "network", "docker")
    @classmethod
    def validate_nested_objects(cls, value: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        object_fields = {"os": ("os_release",), "gpu": ("peak",), "network": (), "docker": ()}
        list_fields = {"os": (), "gpu": ("gpus",), "network": ("interfaces",), "docker": ("containers",)}
        for key in object_fields[info.field_name]:
            if value.get(key) is not None and not isinstance(value[key], dict):
                raise ValueError(f"{key} must be an object")
        for key in list_fields[info.field_name]:
            if value.get(key) is not None and (
                not isinstance(value[key], list)
                or not all(isinstance(item, dict) for item in value[key])
            ):
                raise ValueError(f"{key} must be a list of objects")
        if info.field_name == "gpu" and value.get("peak") is not None:
            value["peak"] = MetricSample.model_validate(value["peak"]).model_dump(exclude_unset=True)
        return value

    @property
    def incomplete(self) -> bool:
        return any((not s.ok or s.errors) and not s.optional for s in self.collector_statuses) or any(
            f.rule_id.startswith("rule.error.") for f in self.findings
        )
