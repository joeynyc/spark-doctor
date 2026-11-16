from __future__ import annotations

import json
import os
import shlex
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from spark_doctor.collectors import cuda_env, firmware, gpu, logs, memory, network, os_info, processes
from spark_doctor.models import ScanReport
from spark_doctor.privacy import redact_report
from spark_doctor.shell import ShellResult


def result(stdout: str = "", *, error: str | None = None) -> ShellResult:
    return ShellResult("probe", error is None, 1 if error else 0, stdout, "", error)


@pytest.mark.parametrize("use_sudo", [False, True])
@pytest.mark.parametrize("error", [
    None, "command_not_found", "nonzero_exit", "timeout", "exception:PermissionError:denied",
])
def test_firmware_probes_use_private_c_locale_and_preserve_failures(
    monkeypatch: pytest.MonkeyPatch, use_sudo: bool, error: str | None,
) -> None:
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
    monkeypatch.setenv("LANGUAGE", "de:en")
    monkeypatch.setenv("PATH", "/test/workload/bin:/usr/bin")
    monkeypatch.setenv("SPARK_DOCTOR_TEST_KEEP", "unchanged")
    original_environment = dict(os.environ)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(args: list[str], **kwargs: Any) -> ShellResult:
        calls.append((args, kwargs))
        return result("" if error else "probe output", error=error)

    monkeypatch.setattr(firmware, "run", run)
    data, status = firmware.collect_firmware(use_sudo=use_sudo)
    assert [args[0] for args, _ in calls] == (["sudo"] if use_sudo else []) + ["fwupdmgr", "mokutil"]
    expected_environment = {**original_environment, "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C"}
    for _, kwargs in calls:
        assert isinstance(kwargs.get("env"), dict)
        # Keep inherited credentials out of pytest assertion diffs.
        assert bool(kwargs["env"] == expected_environment)
        assert kwargs["env"] is not os.environ
    assert bool(dict(os.environ) == original_environment)
    if error:
        assert not data and not status.ok
        assert len(status.errors) == len(calls)
        assert all(error in message for message in status.errors)
        assert not ScanReport(firmware=data, collector_statuses=[status]).incomplete
    else:
        assert set(data) == {"fwupdmgr", "secure_boot"} | ({"dmidecode"} if use_sudo else set())
        assert status.ok and not status.errors


def test_firmware_locale_keeps_partial_success_after_failed_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(args: list[str], **kwargs: Any) -> ShellResult:
        return result(error="timeout") if args[0] == "fwupdmgr" else result("probe output")

    monkeypatch.setattr(firmware, "run", run)
    data, status = firmware.collect_firmware(use_sudo=True)
    assert set(data) == {"dmidecode", "secure_boot"}
    assert status.errors == ["fwupdmgr: timeout"]
    assert not ScanReport(firmware=data, collector_statuses=[status]).incomplete


@pytest.mark.parametrize("error", ["command_not_found", "nonzero_exit", "timeout"])
def test_log_failures_are_recorded_without_shell_pipelines(monkeypatch: pytest.MonkeyPatch, error: str) -> None:
    calls = []

    def run(args: list[str], **kwargs: Any) -> ShellResult:
        calls.append(args)
        return result(error=error)

    monkeypatch.setattr(logs, "run", run)
    data, status = logs.collect_logs()
    assert all(isinstance(args, list) for args in calls)
    assert not data
    assert len(status.errors) == 2
    assert all(error in value for value in status.errors)


def test_log_tails_are_bounded_and_partial_success_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        logs, "run", lambda args, **kw: result("\n".join(map(str, range(1000))))
        if args[0] == "dmesg" else result(error="timeout")
    )
    data, status = logs.collect_logs()
    assert data["dmesg_tail"].splitlines() == list(map(str, range(800, 1000)))
    assert status.errors and "journalctl" in status.errors[0]


def test_network_failed_ip_probes_are_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(network, "run", lambda *a, **kw: result(error="command_not_found"))
    monkeypatch.setattr(network.glob, "glob", lambda pattern: [])
    data, status = network.collect_network()
    assert data["interfaces"] == []
    assert any("ip -br addr" in e for e in status.errors)
    assert any("ip -br link" in e for e in status.errors)


@pytest.mark.parametrize("wireless", [False, True])
def test_network_requires_speed_only_for_active_wired_interfaces(
    monkeypatch: pytest.MonkeyPatch, wireless: bool,
) -> None:
    interface = "/sys/class/net/wlan0" if wireless else "/sys/class/net/eth0"
    monkeypatch.setattr(network.glob, "glob", lambda _: [interface])
    monkeypatch.setattr(network, "run", lambda *args, **kw: result("ok"))
    monkeypatch.setattr(network, "read_text", lambda path: "up" if path.endswith("/operstate") else None)
    monkeypatch.setattr(network.os, "readlink", lambda _: "/drivers/mt7925e" if wireless else "/drivers/r8169")
    monkeypatch.setattr(network.os.path, "isdir", lambda path: wireless and path == interface + "/wireless")
    data, status = network.collect_network()
    assert data["interfaces"][0]["operstate"] == "up"
    assert "speed_mbps" not in data["interfaces"][0]
    assert bool(status.errors) is (not wireless)
    assert ScanReport(network=data, collector_statuses=[status]).incomplete is (not wireless)


def test_memory_missing_available_and_psi_are_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "read_text", lambda path: "MemTotal: 100000 kB\n" if path == "/proc/meminfo" else None)
    data, status = memory.collect_memory()
    assert data.mem_total_kb == 100000
    assert data.mem_available_kb is None
    assert any("MemAvailable" in e for e in status.errors)
    assert any("pressure/memory" in e for e in status.errors)


def test_os_malformed_loadavg_preserves_other_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os_info, "read_text", lambda path: "bad input data" if path == "/proc/loadavg" else None)
    monkeypatch.setattr(os_info, "run", lambda *a, **kw: result("aarch64"))
    data, status = os_info.collect_os()
    assert data["arch"] == "aarch64"
    assert any("loadavg" in e for e in status.errors)


@pytest.mark.parametrize(("error", "recorded"), [("command_not_found", False), ("timeout", True)])
def test_os_missing_dpkg_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, error: str, recorded: bool,
) -> None:
    monkeypatch.setattr(os_info, "read_text", lambda path: "1.0 0.5 0.2 1/100 1" if path != "/etc/os-release" else "ID=fedora\n")
    monkeypatch.setattr(
        os_info, "run", lambda args, **kw: result(error=error) if args[0] == "dpkg-query" else result("aarch64")
    )
    _, status = os_info.collect_os()
    assert any("dpkg-query" in e for e in status.errors) is recorded
    assert ScanReport(collector_statuses=[status]).incomplete is recorded


@pytest.mark.parametrize("output", ["", "null", "[]", "not json"])
def test_core_python_bad_output_is_reported(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    def run(args: list[str], **kwargs: Any) -> ShellResult:
        if args[0] == "chosen-python":
            return result(output)
        return result(error="command_not_found")

    monkeypatch.setattr(cuda_env, "run", run)
    data, status = cuda_env.collect_cuda_env("chosen-python")
    assert "python" not in data
    assert any("python probe" in e for e in status.errors)


def test_cuda_native_probe_failures_are_recorded_with_core_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(args: list[str], **kwargs: Any) -> ShellResult:
        if args[0] == "chosen-python":
            return result(json.dumps({"torch_import_ok": False, "torch_import_error": "ModuleNotFoundError: No module named 'torch'"})
                          if args[-1] == cuda_env._PY_PROBE else '{"optional_gpu_packages": {}}')
        return result(error="timeout")

    monkeypatch.setattr(cuda_env, "run", run)
    data, status = cuda_env.collect_cuda_env("chosen-python")
    assert data["python"]["torch_import_ok"] is False
    assert any("nvcc" in e for e in status.errors)
    assert any("ldconfig" in e for e in status.errors)
    assert not any("torch" in e for e in status.errors)


CSV_ROW = "0, NVIDIA GB10, GPU-example, 580.0, 45, 90, 14, 500, P0\n"
DMON_HEADER = "# gpu pwr gtemp sm pclk\n"


def test_gpu_csv_gap_breaks_sample_sequence_and_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    queries = 0

    def run(args: list[str], **kwargs: Any) -> ShellResult:
        nonlocal queries
        if any(arg.startswith("--query-gpu") for arg in args):
            queries += 1
            return result(error="timeout") if queries == 3 else result(CSV_ROW)
        if "dmon" in args:
            return result(error="nonzero_exit")
        return result("ok")

    monkeypatch.setattr(gpu, "run", run)
    monkeypatch.setattr(gpu.time, "sleep", lambda _: None)
    data, samples, status = gpu.collect_gpu(3)
    assert data["sampler"] == "csv"
    assert len(samples) == 3
    assert samples[1].gpu_utilization_percent is None
    assert status.errors and any("timeout" in e for e in status.errors)


def test_gpu_complete_csv_fallback_does_not_claim_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(args: list[str], **kwargs: Any) -> ShellResult:
        if any(arg.startswith("--query-gpu") for arg in args):
            return result(CSV_ROW)
        return result(error="nonzero_exit") if "dmon" in args else result("ok")

    monkeypatch.setattr(gpu, "run", run)
    monkeypatch.setattr(gpu.time, "sleep", lambda _: None)
    data, samples, status = gpu.collect_gpu(3)
    assert data["sampler"] == "csv"
    assert len(samples) == 3
    assert status.ok and not status.errors


def test_gpu_collection_does_not_depend_on_pmon(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(args: list[str], **kwargs: Any) -> ShellResult:
        if "pmon" in args:
            return result(error="nonzero_exit")
        if any(arg.startswith("--query-gpu") for arg in args):
            return result(CSV_ROW)
        return result(error="nonzero_exit") if "dmon" in args else result("ok")

    monkeypatch.setattr(gpu, "run", run)
    monkeypatch.setattr(gpu.time, "sleep", lambda _: None)
    data, _, status = gpu.collect_gpu(3)
    assert status.ok and not status.errors
    assert not ScanReport(gpu=data, collector_statuses=[status]).incomplete


@pytest.mark.parametrize("dmon_output", [DMON_HEADER + "0 - - - -\n", DMON_HEADER + "0 14 40 90 500\n"])
def test_gpu_unusable_or_short_dmon_is_not_complete(monkeypatch: pytest.MonkeyPatch, dmon_output: str) -> None:
    queries = 0

    def run(args: list[str], **kwargs: Any) -> ShellResult:
        nonlocal queries
        if any(arg.startswith("--query-gpu") for arg in args):
            queries += 1
            return result(CSV_ROW) if queries == 1 else result(error="timeout")
        return result(dmon_output if "dmon" in args else "ok")

    monkeypatch.setattr(gpu, "run", run)
    monkeypatch.setattr(gpu.time, "sleep", lambda _: None)
    _, _, status = gpu.collect_gpu(3)
    assert status.errors


def test_gpu_malformed_dmon_row_remains_a_gap() -> None:
    samples = gpu._parse_dmon(DMON_HEADER + "0 14 40 90 500\nbroken\n0 14 40 90 500\n")
    assert len(samples) == 3
    assert samples[1].gpu_utilization_percent is None
    assert (samples[2].timestamp - samples[0].timestamp).total_seconds() == 2


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "broken"])
def test_native_memory_invalid_psi_keeps_available_and_survives_redaction(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    meminfo = "MemTotal: 100000 kB\nMemAvailable: 1000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
    psi = f"some avg10={value} avg60=0 avg300=0 total=0\nfull avg10=0 avg60=0 avg300=0 total=0\n"
    monkeypatch.setattr(memory, "read_text", lambda path: meminfo if path == "/proc/meminfo" else psi)
    data, status = memory.collect_memory()
    assert data.mem_available_kb == 1000
    assert "avg10" not in data.psi_memory["some"]
    assert status.errors
    report = redact_report(ScanReport(memory=data, collector_statuses=[status]))
    assert report.memory.mem_available_kb == 1000


def test_native_memory_missing_full_pressure_is_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "read_text", lambda _: "some avg10=0 avg60=0 avg300=0 total=0\n")
    _, status = memory.collect_memory()
    assert any("pressure/memory" in e for e in status.errors)


def test_process_argument_boundaries_survive_collection_and_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    argv = ["python", "-m", "vllm.entrypoints", "--api-key", "two word secret", "--password", "other secret"]
    info = {"pid": 1, "name": "python", "cmdline": argv, "memory_info": None}
    monkeypatch.setattr(psutil, "process_iter", lambda _, **kw: iter([SimpleNamespace(info=info)]))
    data, status = processes.collect_processes()
    assert shlex.split(data[0].args) == argv
    assert data[0].detected_backend == "vllm"
    assert status.ok and not status.errors
    redacted = redact_report(ScanReport(processes=data)).model_dump_json()
    assert "two word secret" not in redacted
    assert "other secret" not in redacted


def test_process_enumeration_permission_error_keeps_partial_data(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    def iter_processes(_: list[str], **kw: object) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(info={"pid": 1, "name": "python", "cmdline": ["python"]})
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil, "process_iter", iter_processes)
    data, status = processes.collect_processes()
    assert len(data) == 1
    assert any("AccessDenied" in e for e in status.errors)


def test_process_unavailable_attributes_mark_report_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    def iter_processes(attrs: list[str], ad_value: object = None) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(info={
            "pid": 42, "name": "python", "cmdline": ad_value, "memory_info": ad_value,
            "cpu_percent": 0.0, "memory_percent": ad_value,
        })

    monkeypatch.setattr(psutil, "process_iter", iter_processes)
    data, status = processes.collect_processes()
    assert len(data) == 1 and data[0].pid == 42
    assert data[0].args == "" and data[0].rss_kb is None
    assert status.errors
    assert ScanReport(processes=data, collector_statuses=[status]).incomplete


@pytest.mark.parametrize("rdma_ok", [True, False])
def test_missing_ibstat_on_connectx_only_matters_without_rdma(
    monkeypatch: pytest.MonkeyPatch, rdma_ok: bool,
) -> None:
    # DGX OS ships iproute2's `rdma` but not infiniband-diags' `ibstat`.
    def fake_run(args: list[str], **kw: Any) -> ShellResult:
        if args[0] == "ibstat" or (args[0] == "rdma" and not rdma_ok):
            return result(error="command_not_found")
        return result("link rocep1s0f1/1 state ACTIVE physical_state LINK_UP netdev enp1s0f1np1")

    interface = "/sys/class/net/enp1s0f1np1"
    monkeypatch.setattr(network.glob, "glob", lambda _: [interface])
    monkeypatch.setattr(network, "run", fake_run)
    monkeypatch.setattr(network, "read_text", lambda path: "200000" if path.endswith("/speed") else "up")
    monkeypatch.setattr(network.os, "readlink", lambda _: "/drivers/mlx5_core")
    data, status = network.collect_network()
    assert data["interfaces"][0]["connectx_like"] is True
    assert any("ibstat" in e for e in status.errors) is (not rdma_ok)
    assert ScanReport(network=data, collector_statuses=[status]).incomplete is (not rdma_ok)
