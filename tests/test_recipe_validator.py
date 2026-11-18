from pathlib import Path

import pytest
from pydantic import ValidationError

from spark_doctor.recipes.schema import Recipe
from spark_doctor.recipes.validator import _looks_moe, load_recipe, validate_recipe

FIXTURES = Path(__file__).parent / "fixtures"


def test_tp_exceeds_gpu_count_fails():
    recipe = load_recipe(FIXTURES / "recipe_tp_too_high.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert result.status == "fail"
    ids = [i.id for i in result.issues]
    assert "recipe.tensor_parallel_exceeds_gpu_count" in ids


def test_ok_recipe_passes_or_warns():
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1, detected_arch="aarch64")
    assert result.status in ("pass", "warn")
    assert not any(i.id == "recipe.tensor_parallel_exceeds_gpu_count" for i in result.issues)


def test_arm_arch_incompatibility():
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    # Override image with amd64 hint
    recipe.runtime.container_image = "docker.io/example/vllm-amd64:latest"
    result = validate_recipe(recipe, detected_gpu_count=1, detected_arch="aarch64")
    assert result.status == "fail"
    assert any(i.id == "recipe.arch_incompatible" for i in result.issues)


def test_insufficient_memory():
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    recipe.expectations.min_mem_available_gb_before_start = 64
    result = validate_recipe(recipe, detected_gpu_count=1, mem_available_gb=10)
    assert result.status == "fail"


def _ids(result) -> list:
    return [i.id for i in result.issues]


def test_enforce_eager_flagged():
    recipe = load_recipe(FIXTURES / "recipe_enforce_eager.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert "recipe.vllm_enforce_eager" in _ids(result)
    issue = next(i for i in result.issues if i.id == "recipe.vllm_enforce_eager")
    assert issue.severity == "info"
    assert "KV-cache allocation failures" in issue.detail
    assert "Keep --enforce-eager" in issue.suggested_fix
    assert "2.6x" not in issue.detail


def test_enforce_eager_absent_no_finding():
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert "recipe.vllm_enforce_eager" not in _ids(result)


def test_mxfp4_moe_explicit_is_advisory():
    recipe = load_recipe(FIXTURES / "recipe_mxfp4_moe.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1)
    ids = _ids(result)
    assert "recipe.mxfp4_moe_on_blackwell" in ids
    assert result.status == "warn"
    issue = next(i for i in result.issues if i.id == "recipe.mxfp4_moe_on_blackwell")
    assert issue.severity == "warning"
    assert "cannot establish compatibility" in issue.detail


def test_mxfp4_moe_inferred_from_model_name():
    # No explicit is_moe; the a3b active-param name should trip _looks_moe.
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    recipe.is_moe = None
    recipe.runtime.quantization = "mxfp4"
    assert "a3b" in recipe.model.lower()
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert "recipe.mxfp4_moe_on_blackwell" in _ids(result)


def test_mxfp4_dense_model_no_finding():
    # MXFP4 on a dense (non-MoE) model must NOT flag.
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    recipe.model = "meta-llama/Llama-3.1-8B"
    recipe.is_moe = None
    recipe.runtime.quantization = "mxfp4"
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert "recipe.mxfp4_moe_on_blackwell" not in _ids(result)


def test_looks_moe_detection():
    assert _looks_moe("Qwen/Qwen3.6-35B-A3B")
    assert _looks_moe("mistralai/Mixtral-8x7B")
    assert _looks_moe("some-moe-model")
    assert _looks_moe("foo-8x22b")
    assert _looks_moe("Qwen_Qwen3.6_35B_A3B_MXFP4")
    assert not _looks_moe("meta-llama/Llama-3.1-8B")
    assert not _looks_moe("google/gemma-2-27b")


@pytest.mark.parametrize("backend", ["llama.cpp", "ollama", "sglang"])
def test_mxfp4_other_backend_does_not_get_vllm_warning(backend: str) -> None:
    recipe = load_recipe(FIXTURES / "recipe_mxfp4_moe.yaml")
    recipe.backend = backend
    result = validate_recipe(recipe)
    assert "recipe.mxfp4_moe_on_blackwell" not in _ids(result)
    assert result.status != "fail"


@pytest.mark.parametrize(
    "command",
    [
        "vllm serve model --enforce-eager=False",
        "vllm serve model --no-enforce-eager",
        "vllm serve model --served-model-name=example--enforce-eager",
        "vllm serve 'unterminated",
    ],
)
def test_eager_match_requires_an_enabled_flag(command: str) -> None:
    recipe = load_recipe(FIXTURES / "recipe_enforce_eager.yaml")
    recipe.runtime.command = command
    assert "recipe.vllm_enforce_eager" not in _ids(validate_recipe(recipe))


def test_eager_flag_still_detected_when_command_has_unbalanced_quote() -> None:
    recipe = load_recipe(FIXTURES / "recipe_enforce_eager.yaml")
    recipe.runtime.command = "vllm serve 'unterminated --enforce-eager"
    assert "recipe.vllm_enforce_eager" in _ids(validate_recipe(recipe))


@pytest.mark.parametrize(
    ("backend", "image", "known"),
    [
        ("vllm", "nvcr.io/nvidia/vllm-evil:1", False),
        ("vllm", "nvcr.io/nvidia/vllm/evil:1", False),
        ("vllm", "nvcr.io/nvidia/vllm", True),
        ("vllm", "nvcr.io/nvidia/vllm:25.09-py3", True),
        ("vllm", "nvcr.io/nvidia/vllm@sha256:" + "a" * 64, True),
        ("llama.cpp", "ghcr.io/ggml-org/llama.cpp:server-cuda", True),
    ],
)
def test_known_image_matches_only_at_repository_boundary(backend: str, image: str, known: bool) -> None:
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    recipe.backend = backend
    recipe.runtime.container_image = image
    assert ("recipe.container_mismatch_or_unknown" not in _ids(validate_recipe(recipe))) is known


def test_explicit_dense_model_overrides_name_heuristic() -> None:
    recipe = load_recipe(FIXTURES / "recipe_mxfp4_moe.yaml")
    recipe.model = "example/some-moe-model"
    recipe.is_moe = False
    assert "recipe.mxfp4_moe_on_blackwell" not in _ids(validate_recipe(recipe))


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("hardware", "nodes", 0),
        ("hardware", "nodes", -1),
        ("hardware", "gpus_per_node", 0),
        ("hardware", "gpus_per_node", -1),
        ("runtime", "tensor_parallel_size", 0),
        ("runtime", "tensor_parallel_size", -1),
        ("runtime", "max_model_len", 0),
        ("runtime", "max_model_len", -1),
        ("runtime", "gpu_memory_utilization", 0),
        ("runtime", "gpu_memory_utilization", -0.5),
        ("runtime", "gpu_memory_utilization", 1.01),
        ("runtime", "gpu_memory_utilization", float("nan")),
        ("runtime", "gpu_memory_utilization", float("inf")),
        ("expectations", "min_mem_available_gb_before_start", -1),
        ("expectations", "min_mem_available_gb_before_start", float("nan")),
        ("expectations", "min_mem_available_gb_before_start", float("inf")),
    ],
)
def test_invalid_numeric_settings_rejected(section: str, field: str, value: float) -> None:
    data = load_recipe(FIXTURES / "recipe_ok.yaml").model_dump()
    data[section][field] = value
    with pytest.raises(ValidationError) as exc:
        Recipe.from_dict(data)
    assert (section, field) in [error["loc"] for error in exc.value.errors()]


@pytest.mark.parametrize("section", [None, "hardware", "runtime", "expectations"])
def test_unknown_recipe_keys_rejected(section: str | None) -> None:
    data = load_recipe(FIXTURES / "recipe_ok.yaml").model_dump()
    target = data if section is None else data[section]
    target["misspelled_setting"] = 1
    with pytest.raises(ValidationError) as exc:
        Recipe.from_dict(data)
    assert any(error["type"] == "extra_forbidden" for error in exc.value.errors())


def test_numeric_boundary_values_accepted() -> None:
    data = load_recipe(FIXTURES / "recipe_ok.yaml").model_dump()
    data["runtime"]["gpu_memory_utilization"] = 1
    data["expectations"]["min_mem_available_gb_before_start"] = 0
    recipe = Recipe.from_dict(data)
    assert recipe.runtime.gpu_memory_utilization == 1
    assert recipe.expectations.min_mem_available_gb_before_start == 0


def test_tp_cannot_exceed_declared_multinode_capacity() -> None:
    recipe = load_recipe(FIXTURES / "recipe_multinode_tp_too_high.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert result.status == "fail"
    assert "recipe.tensor_parallel_exceeds_gpu_count" in _ids(result)


@pytest.mark.parametrize(
    ("nodes", "gpus_per_node", "detected_gpus", "tp", "valid"),
    [
        (2, 1, 1, 2, True),
        (2, 2, 2, 4, True),
        (2, 2, 2, 5, False),
        (1, 4, 1, 2, False),
        (1, 1, 2, 2, False),
        (1, 1, 0, 1, False),
    ],
)
def test_tensor_parallel_uses_declared_topology_and_local_single_node_limit(
    nodes: int, gpus_per_node: int, detected_gpus: int, tp: int, valid: bool
) -> None:
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    recipe.hardware.nodes = nodes
    recipe.hardware.gpus_per_node = gpus_per_node
    recipe.runtime.tensor_parallel_size = tp
    result = validate_recipe(recipe, detected_gpu_count=detected_gpus)
    assert ("recipe.tensor_parallel_exceeds_gpu_count" not in _ids(result)) is valid
