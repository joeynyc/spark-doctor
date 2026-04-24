from pathlib import Path

from spark_doctor.recipes.validator import load_recipe, validate_recipe

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
