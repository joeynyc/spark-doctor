from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Hardware(BaseModel):
    nodes: int = 1
    gpus_per_node: int = 1


class Runtime(BaseModel):
    container_image: str | None = None
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    kv_cache_dtype: str | None = None
    command: str | None = None


class Expectations(BaseModel):
    requires_docker: bool = True
    requires_connectx: bool = False
    min_mem_available_gb_before_start: float | None = None


class Recipe(BaseModel):
    name: str
    backend: str
    model: str
    hardware: Hardware = Field(default_factory=Hardware)
    runtime: Runtime = Field(default_factory=Runtime)
    expectations: Expectations = Field(default_factory=Expectations)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        return cls.model_validate(data)
