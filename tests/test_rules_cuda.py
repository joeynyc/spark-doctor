import json
from pathlib import Path

import pytest

from spark_doctor.collectors.cuda_env import (
    parse_ldconfig_libcudart,
    parse_nvcc_release,
    parse_smi_cuda_version,
)
from spark_doctor.models import ScanReport
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> ScanReport:
    data = json.loads((FIXTURES / name).read_text())
    return ScanReport.model_validate(data)


def _cuda_findings(report: ScanReport) -> dict:
    return {f.rule_id: f for f in run_rules(report) if f.rule_id.startswith("cuda.")}


def test_cu12_wheel_on_cuda13_is_critical():
    ids = _cuda_findings(_load("cuda_wheel_mismatch.json"))
    assert "cuda.torch_cu12_wheel" in ids
    assert ids["cuda.torch_cu12_wheel"].severity == "critical"


def test_cu12_wheel_fixture_also_flags_nvcc_without_duplicate_arch_warning():
    ids = _cuda_findings(_load("cuda_wheel_mismatch.json"))
    assert "cuda.nvcc_toolkit_mismatch" in ids
    assert ids["cuda.nvcc_toolkit_mismatch"].severity == "warning"
    assert "cuda.sm121_not_in_arch_list" not in ids


# Arch lists and outcomes observed on a real GB10: cu126 kernels fail, cu128 kernels run.
@pytest.mark.parametrize(("cuda", "arch_list", "flagged"), [
    ("12.6", ["sm_80", "sm_90"], True),
    ("12.8", ["sm_80", "sm_90", "sm_100", "sm_120"], False),
    ("12.4", None, True),
    ("12.8", None, False),
])
def test_cu12_wheel_flags_only_builds_without_gb10_kernels(
    cuda: str, arch_list: list[str] | None, flagged: bool,
) -> None:
    report = _load("cuda_wheel_mismatch.json")
    report.cuda_env["python"]["torch_cuda_version"] = cuda
    report.cuda_env["python"]["torch_arch_list"] = arch_list
    assert ("cuda.torch_cu12_wheel" in _cuda_findings(report)) is flagged


def test_cu12_wheel_is_not_flagged_on_other_gpus():
    report = _load("cuda_wheel_mismatch.json")
    report.gpu = {"name": "NVIDIA GeForce RTX 5090", "gpus": [{"name": "NVIDIA GeForce RTX 5090"}]}
    assert "cuda.torch_cu12_wheel" not in _cuda_findings(report)


def test_libcudart_import_failure_is_critical():
    ids = _cuda_findings(_load("cuda_libcudart_failure.json"))
    assert "cuda.libcudart_missing" in ids
    assert ids["cuda.libcudart_missing"].severity == "critical"
    assert "cuda.torch_cu12_wheel" not in ids


def test_healthy_cuda13_env_no_findings():
    assert _cuda_findings(_load("cuda_env_healthy.json")) == {}


def test_missing_cuda_env_stays_silent():
    assert _cuda_findings(_load("healthy_minimal.json")) == {}


def test_cpu_only_torch_in_selected_python_warns_without_blaming_containers() -> None:
    report = _load("cuda_torch_cpu_only.json")
    findings = _cuda_findings(report)

    assert set(findings) == {"cuda.torch_cpu_only"}
    finding = findings["cuda.torch_cpu_only"]
    assert finding.severity == "warning"
    assert finding.confidence == "high"
    assert "/usr/bin/python3" in " ".join(finding.evidence)
    assert "2.13.0+cpu" in " ".join(finding.evidence)
    assert "selected Python environment" in finding.explanation
    assert "container" in finding.explanation
    actions = " ".join(finding.recommended_actions)
    assert "--python" in actions
    assert "intentional" in actions


@pytest.mark.parametrize("version", ["2.13.0+cpu", "2.13.0a0+cpu.nightly"])
def test_explicit_cpu_build_tag_triggers_warning(version: str) -> None:
    report = _load("cuda_torch_cpu_only.json")
    report.cuda_env["python"]["torch_version"] = version
    assert "cuda.torch_cpu_only" in _cuda_findings(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("torch_import_ok", False),
        ("torch_import_ok", None),
        ("torch_import_ok", 1),
        ("torch_cuda_version", "13.0"),
        ("torch_cuda_version", ""),
        ("torch_cuda_available", True),
        ("torch_cuda_available", None),
        ("torch_cuda_available", 0),
        ("torch_version", "2.13.0"),
        ("torch_version", "2.13.0+rocm7.0"),
        ("torch_version", "2.13.0+cu130"),
        ("torch_version", "2.13.0+cpuish"),
        ("torch_version", None),
    ],
)
def test_cpu_only_torch_requires_explicit_consistent_build_evidence(field: str, value: object) -> None:
    report = _load("cuda_torch_cpu_only.json")
    report.cuda_env["python"][field] = value
    assert "cuda.torch_cpu_only" not in _cuda_findings(report)


@pytest.mark.parametrize(
    "field",
    ["torch_import_ok", "torch_cuda_version", "torch_cuda_available", "torch_version"],
)
def test_cpu_only_torch_does_not_infer_from_missing_probe_fields(field: str) -> None:
    report = _load("cuda_torch_cpu_only.json")
    del report.cuda_env["python"][field]
    assert "cuda.torch_cpu_only" not in _cuda_findings(report)


def test_cpu_only_torch_uses_requested_interpreter_when_resolved_path_is_missing() -> None:
    report = _load("cuda_torch_cpu_only.json")
    del report.cuda_env["python"]["executable"]
    report.cuda_env["python_executable"] = "/opt/workload/bin/python"
    finding = _cuda_findings(report)["cuda.torch_cpu_only"]
    assert "/opt/workload/bin/python" in " ".join(finding.evidence)


def test_unavailable_cuda_device_does_not_imply_cpu_only_torch() -> None:
    report = _load("cuda_env_healthy.json")
    report.cuda_env["python"]["torch_cuda_available"] = False
    report.cuda_env["python"]["torch_arch_list"] = []
    assert "cuda.torch_cpu_only" not in _cuda_findings(report)


@pytest.mark.parametrize(("arch_list", "flagged"), [
    # Official cu130 wheels as observed on a real Spark: sm_120 cubins run on sm_121.
    (["sm_80", "sm_90", "sm_100", "sm_110", "sm_120"], False),
    (["sm_90", "sm_121a"], False),
    (["sm_90", "sm_100", "sm_120a"], True),
    (["sm_80", "sm_86", "sm_90", "compute_90"], True),
])
def test_sm121_arch_check_accepts_binary_compatible_builds(arch_list: list[str], flagged: bool) -> None:
    report = _load("cuda_env_healthy.json")
    report.cuda_env["python"]["torch_arch_list"] = arch_list
    assert ("cuda.sm121_not_in_arch_list" in _cuda_findings(report)) is flagged


def test_parse_smi_cuda_version():
    banner = "| NVIDIA-SMI 580.95.05    Driver Version: 580.95.05    CUDA Version: 13.0 |"
    assert parse_smi_cuda_version(banner) == "13.0"
    assert parse_smi_cuda_version("no match here") is None


def test_parse_nvcc_release():
    text = "Cuda compilation tools, release 13.0, V13.0.48"
    assert parse_nvcc_release(text) == "13.0"


def test_parse_ldconfig_libcudart():
    text = (
        "\tlibcudart.so.13 (libc6,AArch64) => /usr/local/cuda/lib64/libcudart.so.13\n"
        "\tlibcudart.so.13 (libc6,AArch64) => /opt/other/libcudart.so.13\n"
        "\tlibcuda.so.1 (libc6,AArch64) => /usr/lib/libcuda.so.1\n"
    )
    assert parse_ldconfig_libcudart(text) == ["libcudart.so.13"]
