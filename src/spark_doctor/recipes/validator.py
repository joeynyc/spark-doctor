from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .schema import Recipe

SUPPORTED_BACKENDS = {"vllm", "ollama", "llama.cpp", "sglang", "open-webui", "comfyui"}


@dataclass
class RecipeIssue:
    id: str
    severity: Literal["info", "warning", "critical"]
    title: str
    detail: str
    suggested_fix: str | None = None


@dataclass
class RecipeCheckResult:
    recipe_name: str
    status: Literal["pass", "warn", "fail"]
    issues: list[RecipeIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def _load_registry() -> dict[str, Any]:
    reg_path = Path(__file__).parent / "known_registry.yaml"
    try:
        return yaml.safe_load(reg_path.read_text()) or {}
    except OSError:
        return {}


def load_recipe(path: str | Path) -> Recipe:
    text = Path(path).read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Recipe file {path} does not contain a mapping at top level")
    return Recipe.from_dict(data)


def validate_recipe(
    recipe: Recipe,
    *,
    detected_gpu_count: int = 1,
    detected_arch: str | None = None,
    mem_available_gb: float | None = None,
) -> RecipeCheckResult:
    issues: list[RecipeIssue] = []
    registry = _load_registry()

    # Backend supported
    if recipe.backend.lower() not in SUPPORTED_BACKENDS:
        issues.append(
            RecipeIssue(
                id="recipe.backend_unsupported",
                severity="warning",
                title="Unknown backend",
                detail=f"Backend '{recipe.backend}' is not in the supported list: {sorted(SUPPORTED_BACKENDS)}.",
                suggested_fix="Set backend to a supported value, or add it to the registry.",
            )
        )

    # tensor parallel vs gpu count
    tp = recipe.runtime.tensor_parallel_size
    nodes = recipe.hardware.nodes or 1
    if tp and tp > detected_gpu_count and nodes <= 1:
        issues.append(
            RecipeIssue(
                id="recipe.tensor_parallel_exceeds_gpu_count",
                severity="critical",
                title="tensor_parallel_size exceeds detected GPU count",
                detail=(
                    f"Detected GPUs: {detected_gpu_count}. "
                    f"Recipe tensor_parallel_size: {tp}. Nodes: {nodes}."
                ),
                suggested_fix=(
                    "Use tensor_parallel_size: 1 for a single Spark, or declare a multi-node "
                    "configuration with nodes > 1 and the corresponding networking."
                ),
            )
        )

    # gpu_memory_utilization high
    gmu = recipe.runtime.gpu_memory_utilization
    if gmu is not None and gmu > 0.90:
        issues.append(
            RecipeIssue(
                id="recipe.gpu_memory_utilization_high",
                severity="warning",
                title="gpu_memory_utilization is high",
                detail=f"gpu_memory_utilization={gmu} is aggressive and can cause OOM under unified memory pressure.",
                suggested_fix="Start with 0.80 and raise only after stable runs.",
            )
        )

    # excessive context length relative to memory
    mml = recipe.runtime.max_model_len
    if mml is not None and mml > 131072:
        issues.append(
            RecipeIssue(
                id="recipe.context_length_large",
                severity="warning",
                title="Context length is very large",
                detail=f"max_model_len={mml} will require a large KV cache.",
                suggested_fix="Start smaller (e.g. 32768) and increase only after stable runs.",
            )
        )

    # memory expectations
    expected_min = recipe.expectations.min_mem_available_gb_before_start
    if expected_min is not None and mem_available_gb is not None and mem_available_gb < expected_min:
        issues.append(
            RecipeIssue(
                id="recipe.insufficient_memory_available",
                severity="critical",
                title="Insufficient available memory for recipe",
                detail=f"Recipe needs {expected_min} GB available; detected {mem_available_gb:.1f} GB.",
                suggested_fix="Free memory, stop idle backends, or use a smaller model/context.",
            )
        )

    # container image checks
    image = recipe.runtime.container_image
    if recipe.expectations.requires_docker and not image:
        issues.append(
            RecipeIssue(
                id="recipe.container_missing",
                severity="warning",
                title="No container_image declared",
                detail="Recipe requires Docker but does not declare a container_image.",
                suggested_fix="Add runtime.container_image pointing to a validated image.",
            )
        )
    elif image:
        known = set()
        backends_reg = registry.get("backends") or {}
        entry = backends_reg.get(recipe.backend.lower()) or {}
        for k in entry.get("known_images", []) or []:
            known.add(k)
        if known and not any(image.startswith(k) for k in known):
            issues.append(
                RecipeIssue(
                    id="recipe.container_mismatch_or_unknown",
                    severity="warning",
                    title="Container image not in known registry",
                    detail=f"Image '{image}' is not a known validated image for backend '{recipe.backend}'.",
                    suggested_fix=f"Verify against official playbook. Known prefixes: {sorted(known)}",
                )
            )

    # Arm64 architecture hint
    if detected_arch and detected_arch.lower() in ("aarch64", "arm64"):
        if image and any(hint in image.lower() for hint in ("amd64", "x86_64", "x86-64")):
            issues.append(
                RecipeIssue(
                    id="recipe.arch_incompatible",
                    severity="critical",
                    title="Image tag suggests x86_64 on Arm64 host",
                    detail=f"Detected arch '{detected_arch}' but image '{image}' looks x86-specific.",
                    suggested_fix="Use an arm64/aarch64-compatible image variant.",
                )
            )

    status: Literal["pass", "warn", "fail"] = "pass"
    if any(i.severity == "critical" for i in issues):
        status = "fail"
    elif any(i.severity == "warning" for i in issues):
        status = "warn"

    return RecipeCheckResult(recipe_name=recipe.name, status=status, issues=issues)
