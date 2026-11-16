from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from spark_doctor.collectors import cuda_env
from spark_doctor.models import ScanReport
from spark_doctor.rules import run_rules
from spark_doctor.shell import ShellResult


@pytest.mark.parametrize(
    ("stdout", "error"),
    [
        ("", "nonzero_exit"),
        ("", "timeout"),
        ("", "command_not_found"),
        ("", None),
        ("not json", None),
        ("null", None),
        ("[]", None),
        ('{"optional_gpu_packages": []}', None),
    ],
)
def test_optional_probe_failure_preserves_core_data(
    monkeypatch: pytest.MonkeyPatch, stdout: str, error: str | None
) -> None:
    core = {"torch_import_ok": True, "torch_cuda_version": "13.0"}
    probes: list[str] = []

    def fake_run(args: list[str], *, timeout: float, **kwargs: object) -> ShellResult:
        if args[0] == "test-python":
            assert timeout == 60
            probes.append(args[-1])
            if args[-1] == cuda_env._PY_PROBE:
                return ShellResult("python", True, 0, json.dumps(core), "")
            assert args[-1] == cuda_env._OPTIONAL_PKG_PROBE
            return ShellResult("python", error is None, -11 if error else 0, stdout, "", error)
        return ShellResult(args[0], False, None, "", "", "command_not_found")

    monkeypatch.setattr(cuda_env, "run", fake_run)
    data, status = cuda_env.collect_cuda_env("test-python")
    assert data["python"] == core
    assert status.ok
    assert any("optional-package probe:" in e for e in status.errors)
    assert probes == [cuda_env._PY_PROBE, cuda_env._OPTIONAL_PKG_PROBE]


def test_optional_probe_merges_package_results(monkeypatch: pytest.MonkeyPatch) -> None:
    optional = {"flash_attn": {"import_ok": False, "import_error": "not installed"}}

    def fake_run(args: list[str], *, timeout: float, **kwargs: object) -> ShellResult:
        if args[0] != "test-python":
            output = {
                "nvidia-smi": "CUDA Version: 13.0",
                "nvcc": "release 13.0",
                "ldconfig": "",
            }[args[0]]
            return ShellResult(args[0], True, 0, output, "")
        result = (
            {"torch_import_ok": True}
            if args[-1] == cuda_env._PY_PROBE
            else {"optional_gpu_packages": optional}
        )
        return ShellResult("python", True, 0, "import notice\n" + json.dumps(result), "")

    monkeypatch.setattr(cuda_env, "run", fake_run)
    data, status = cuda_env.collect_cuda_env("test-python")
    assert data["python"]["torch_import_ok"] is True
    assert data["python"]["optional_gpu_packages"] == optional
    assert status.errors == []


@pytest.mark.parametrize(
    ("legacy_flag", "modern_flag", "expected"),
    [
        (False, None, False),
        (True, None, True),
        (None, False, False),
        (None, True, True),
        (None, None, None),
        (None, "unknown", None),
    ],
)
def test_actual_probe_detects_legacy_and_modern_bitsandbytes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    legacy_flag: bool | None,
    modern_flag: bool | str | None,
    expected: bool | None,
) -> None:
    bnb = ModuleType("bitsandbytes")
    bnb.__version__ = "test"
    if legacy_flag is not None:
        bnb.COMPILED_WITH_CUDA = legacy_flag
    extension = ModuleType("bitsandbytes.cextension")
    extension.lib = SimpleNamespace(compiled_with_cuda=modern_flag)
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.cextension", extension)
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))

    exec(cuda_env._OPTIONAL_PKG_PROBE, {})
    result = json.loads(capsys.readouterr().out)
    entry = result["optional_gpu_packages"]["bitsandbytes"]
    assert entry["import_ok"] is True
    if expected is None:
        assert "cuda_build" not in entry
    else:
        assert entry["cuda_build"] is expected
    report = ScanReport(
        os={"arch": "aarch64"}, gpu={"name": "NVIDIA GB10"}, cuda_env={"python": result}
    )
    wheel_gap = [f for f in run_rules(report) if f.rule_id == "cuda.aarch64_prebuilt_wheel_gap"]
    assert bool(wheel_gap) is (expected is False)


def test_probe_ignores_modules_shadowed_by_caller_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "shadow-executed"
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "platform.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    monkeypatch.chdir(workdir)
    data, _ = cuda_env.collect_cuda_env(python_executable=sys.executable)
    assert not marker.exists()
    assert data["python"]["python_version"]
