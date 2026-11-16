from __future__ import annotations

import pytest

from spark_doctor.collectors import docker_runtime
from spark_doctor.collectors.docker_runtime import _finalize_gpu_readiness
from spark_doctor.models import CollectorStatus, ScanReport
from spark_doctor.rules import run_rules
from spark_doctor.shell import ShellResult


def _ready(**keys) -> bool:
    out: dict = {"docker_installed": True, "daemon_reachable": True}
    out.update(keys)
    _finalize_gpu_readiness(out)
    return out["gpu_docker_ready"]


def test_ready_via_named_runtime():
    assert _ready(nvidia_runtime_available=True) is True


def test_ready_via_hook_main_field_name():
    # Reads main's key `nvidia_hook_installed`, not the branch's old name.
    assert _ready(nvidia_hook_installed=True) is True


def test_ready_via_cdi_main_field_name():
    assert _ready(cdi_specs_present=True) is True


def test_not_ready_when_no_path():
    assert _ready(
        nvidia_runtime_available=False,
        nvidia_hook_installed=False,
        cdi_specs_present=False,
    ) is False


def test_not_ready_ignores_stale_branch_keys():
    # Old branch keys must NOT satisfy readiness on their own.
    assert _ready(nvidia_runtime_hook_present=True, cdi_available=True) is False


@pytest.mark.parametrize("prerequisite", ["docker_installed", "daemon_reachable"])
def test_not_ready_without_docker_prerequisite(prerequisite: str) -> None:
    assert not _ready(nvidia_hook_installed=True, **{prerequisite: False})


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("nvidia.com/gpu=0\nnvidia.com/gpu=all\n", "", 2),
        ("nvidia.com/gpu=all\nnvidia.com/gpu=all\n", "", 1),
        ("amd.com/gpu=all\n", "Found 1 CDI devices", 0),
        ("", "Found 3 CDI devices", 0),
        ("garbage\nnvidia.com/gpu=\n", "", 0),
    ],
)
def test_cdi_counts_only_nvidia_device_names(
    monkeypatch: pytest.MonkeyPatch, stdout: str, stderr: str, expected: int
) -> None:
    def fake_run(args: list[str], *, timeout: float) -> ShellResult:
        assert args == ["nvidia-ctk", "cdi", "list"]
        assert timeout == 5
        return ShellResult("ctk", True, 0, stdout, stderr)

    monkeypatch.setattr(docker_runtime, "run", fake_run)
    status = CollectorStatus(name="docker", ok=True)
    assert docker_runtime._cdi_list_device_count(status) == expected
    assert status.errors == []


@pytest.mark.parametrize("error", ["timeout", "command_not_found", "nonzero_exit"])
def test_failed_cdi_probe_does_not_claim_readiness(
    monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    monkeypatch.setattr(
        docker_runtime, "run",
        lambda *args, **kwargs: ShellResult(
            "ctk", False, 1, "nvidia.com/gpu=all\n", "Found 1 CDI devices", error
        ),
    )
    status = CollectorStatus(name="docker", ok=True)
    assert docker_runtime._cdi_list_device_count(status) == 0
    assert error in status.errors[0]


@pytest.mark.parametrize("device", ["nvidia.com/gpu=all", "amd.com/gpu=all"])
def test_collected_cdi_evidence_reaches_runtime_rule(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    monkeypatch.setenv("DOCKER_CONTEXT", "default")
    monkeypatch.setattr(
        docker_runtime, "which",
        lambda cmd: f"/usr/bin/{cmd}" if cmd in ("docker", "nvidia-ctk") else None,
    )
    monkeypatch.setattr(docker_runtime, "_cdi_specs_present", lambda: False)

    def fake_run(args: list[str], *, timeout: float, env: dict[str, str] | None = None) -> ShellResult:
        if args == ["nvidia-ctk", "cdi", "list"]:
            return ShellResult("ctk", True, 0, device + "\n", "Found 1 CDI devices")
        assert args[:3] == ["docker", "--host", "unix:///var/run/docker.sock"]
        output = '{"Runtimes":{"runc":{}}}' if args[3] == "info" else "{}"
        if args[3] == "ps":
            output = ""
        return ShellResult("docker", True, 0, output, "")

    monkeypatch.setattr(docker_runtime, "run", fake_run)
    data, status = docker_runtime.collect_docker()
    report = ScanReport(docker=data)
    is_nvidia = device.startswith("nvidia.com/")
    assert status.ok
    assert data["gpu_docker_ready"] is is_nvidia
    assert data["cdi_device_count"] == int(is_nvidia)
    runtime = [f for f in run_rules(report) if f.rule_id == "runtime.docker_unhealthy"]
    assert bool(runtime) is not is_nvidia


def test_missing_docker_is_a_complete_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_runtime, "which", lambda cmd: None)
    monkeypatch.setattr(docker_runtime, "_cdi_specs_present", lambda: False)
    data, status = docker_runtime.collect_docker()
    assert data["docker_installed"] is False
    assert status.ok and not status.errors
    report = ScanReport(docker=data, collector_statuses=[status])
    assert not report.incomplete
    assert any(f.rule_id == "runtime.docker_unhealthy" for f in run_rules(report))
