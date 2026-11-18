from __future__ import annotations

import re
import shlex
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


def _matches_known_image(image: str, known: str) -> bool:
    """Match a known repository only at a boundary, so `vllm-evil` does not pass as `vllm`."""
    return image == known or image.startswith((f"{known}:", f"{known}@"))


def _looks_moe(model_name: str) -> bool:
    """Best-effort MoE detection from a model name."""
    m = model_name.lower()
    if any(tok in m for tok in ("moe", "mixtral")):
        return True
    # active-param naming (a3b, a9b, a22b, ...) or expert grid (8x7b, 8x22b).
    if re.search(r"(?:^|[-_/])a\d+b(?:$|[-_/])|\d+x\d+b", m):
        return True
    return False


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
    nodes = recipe.hardware.nodes
    declared_gpu_count = nodes * recipe.hardware.gpus_per_node
    # A local scan cannot establish remote-node capacity; multi-node capacity is declared.
    gpu_capacity = min(declared_gpu_count, detected_gpu_count) if nodes == 1 else declared_gpu_count
    if tp is not None and tp > gpu_capacity:
        issues.append(
            RecipeIssue(
                id="recipe.tensor_parallel_exceeds_gpu_count",
                severity="critical",
                title="tensor_parallel_size exceeds GPU capacity",
                detail=(
                    f"Recipe tensor_parallel_size: {tp}. Declared capacity: {declared_gpu_count} GPUs "
                    f"({nodes} nodes × {recipe.hardware.gpus_per_node} GPUs per node). "
                    f"Detected local GPUs: {detected_gpu_count}. "
                    "Remote GPU counts are not verified by this local scan."
                ),
                suggested_fix=(
                    f"Use tensor_parallel_size no greater than {gpu_capacity}, or correct "
                    "hardware.nodes and hardware.gpus_per_node to match your actual topology."
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

    command = recipe.runtime.command or ""
    try:
        command_args = shlex.split(command)
    except ValueError:
        command_args = command.split()
    if recipe.backend.lower() == "vllm" and "--enforce-eager" in command_args:
        issues.append(
            RecipeIssue(
                id="recipe.vllm_enforce_eager",
                severity="info",
                title="vLLM --enforce-eager disables CUDA graphs",
                detail=(
                    "runtime.command contains --enforce-eager, which disables CUDA graph "
                    "capture. This can reduce throughput, but also saves graph memory and "
                    "can prevent KV-cache allocation failures. The tradeoff depends on the workload."
                ),
                suggested_fix=(
                    "Keep --enforce-eager if needed for memory or compatibility. Otherwise "
                    "benchmark without it and check memory headroom before enabling CUDA graphs."
                ),
            )
        )

    quant = (recipe.runtime.quantization or "").lower()
    if recipe.backend.lower() == "vllm" and "mxfp4" in quant:
        is_moe = recipe.is_moe if recipe.is_moe is not None else _looks_moe(recipe.model or "")
        if is_moe:
            issues.append(
                RecipeIssue(
                    id="recipe.mxfp4_moe_on_blackwell",
                    severity="warning",
                    title="Verify the vLLM MXFP4 MoE backend for GB10 (sm_121)",
                    detail=(
                        f"quantization='{recipe.runtime.quantization}' with a Mixture-of-Experts "
                        f"model ('{recipe.model}'). Support on GB10 depends on the vLLM "
                        "version and selected MoE backend. Some kernel paths target SM_100; "
                        "others support SM_121. This recipe alone cannot establish compatibility."
                    ),
                    suggested_fix=(
                        "Check your vLLM build's SM_121 support and selected MoE backend, "
                        "then test the model. Use a validated container or another supported "
                        "format if the selected backend fails."
                    ),
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
        if known and not any(_matches_known_image(image, k) for k in known):
            issues.append(
                RecipeIssue(
                    id="recipe.container_mismatch_or_unknown",
                    severity="warning",
                    title="Container image not in known registry",
                    detail=f"Image '{image}' is not a known validated image for backend '{recipe.backend}'.",
                    suggested_fix=f"Verify against official playbook. Known repositories: {sorted(known)}",
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
