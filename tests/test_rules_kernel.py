import json
from pathlib import Path

from spark_doctor.log_evidence import kernel_evidence
from spark_doctor.models import ScanReport
from spark_doctor.rules import run_rules
from spark_doctor.rules.kernel import rule_gpu_xid_error, rule_memory_oom_killer

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture() -> ScanReport:
    return ScanReport.model_validate(json.loads((FIXTURES / "kernel_oom_and_xid.json").read_text()))


def _logs(text: str) -> ScanReport:
    return ScanReport(log_evidence=kernel_evidence([text]))


def test_host_oom_kill_is_critical_and_deduplicated_across_sources() -> None:
    finding = rule_memory_oom_killer.fn(_fixture())[0]
    assert finding.severity == "critical"
    assert finding.evidence == ["Kernel OOM killer killed python3 (pid 4242)"]


def test_cgroup_oom_kill_is_a_container_limit_warning() -> None:
    report = _logs("kernel: Memory cgroup out of memory: Killed process 77 (vllm) total-vm:1kB")
    finding = rule_memory_oom_killer.fn(report)[0]
    assert finding.severity == "warning"
    assert "limit" in finding.explanation


def test_hardware_xid_is_critical_with_reboot_advice() -> None:
    finding = rule_gpu_xid_error.fn(_fixture())[0]
    assert finding.severity == "critical"
    assert len(finding.evidence) == 1
    assert finding.evidence[0].startswith("Xid 79")
    assert any("reboot" in a.lower() for a in finding.recommended_actions)


def test_application_xid_is_warning_without_reboot_advice() -> None:
    report = _logs("kernel: NVRM: Xid (PCI:000f:01:00): 13, pid=4242, name=python3, Graphics Exception")
    finding = rule_gpu_xid_error.fn(report)[0]
    assert finding.severity == "warning"
    assert not any("reboot" in a.lower() for a in finding.recommended_actions)
    assert not finding.escalation_actions


def test_clean_logs_produce_no_kernel_findings() -> None:
    ids = {f.rule_id for f in run_rules(_logs("kernel: usb 1-1: new high-speed USB device"))}
    assert not ids & {"memory.oom_killer", "gpu.xid_error"}


def test_fixture_registered_rules_fire() -> None:
    ids = {f.rule_id for f in run_rules(_fixture())}
    assert {"memory.oom_killer", "gpu.xid_error"} <= ids
