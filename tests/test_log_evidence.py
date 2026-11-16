from __future__ import annotations

import json
from pathlib import Path

import pytest

from spark_doctor.collectors import logs
from spark_doctor.log_evidence import CARRIED_OVER_NOTE
from spark_doctor.models import Finding, ScanReport
from spark_doctor.rules import run_rules
from spark_doctor.shell import ShellResult

FIXTURES = Path(__file__).parent / "fixtures"
LOG_RULES = {"memory.oom_killer", "gpu.xid_error", "thermal.shutdown_risk", "backend.kv_cache_oom"}
KERNEL_RULES = {"memory.oom_killer", "gpu.xid_error", "thermal.shutdown_risk"}

FORGED_JOURNAL = "\n".join([
    "Nov 16 10:00:00 spark sshd[9]: Invalid user overheat from 10.0.0.5",
    "Nov 16 10:00:01 spark sshd[9]: Invalid user NVRM: Xid (PCI:0000:01:00): 79, x",
    "Nov 16 10:00:02 spark sshd[9]: Invalid user Out of memory: Killed process 1 (x) from 10.0.0.5",
    # `logger -t kernel` output is indistinguishable from a kernel line in the general journal.
    "Nov 16 10:00:03 spark kernel: thermal thermal_zone0: critical temperature reached, shutting down",
    "Nov 16 10:00:04 spark kernel: NVRM: Xid (PCI:000f:01:00): 79, pid=1, GPU has fallen off the bus.",
])

GENUINE = {
    "memory.oom_killer": "Out of memory: Killed process 4242 (python3) total-vm:98765432kB",
    "gpu.xid_error": "NVRM: Xid (PCI:000f:01:00): 79, pid=1, GPU has fallen off the bus.",
    "thermal.shutdown_risk": "thermal thermal_zone0: critical temperature reached (105 C), shutting down",
}


def _shell(stdout: str = "", *, error: str | None = None) -> ShellResult:
    return ShellResult("probe", error is None, 1 if error else 0, stdout, "", error)


def _collect(
    monkeypatch: pytest.MonkeyPatch, *, dmesg: str | None = None, kernel: str | None = None, journal: str = "",
) -> ScanReport:
    def run(args: list[str], **kw: object) -> ShellResult:
        if args[0] == "dmesg":
            return _shell(dmesg) if dmesg is not None else _shell(error="nonzero_exit")
        if "-k" in args:
            return _shell(kernel) if kernel is not None else _shell("-- No entries --")
        return _shell(journal)

    monkeypatch.setattr(logs, "run", run)
    data, status = logs.collect_logs()
    evidence = data.pop("evidence", [])
    return ScanReport(logs=data, log_evidence=evidence, collector_statuses=[status])


def _rule_ids(report: ScanReport) -> set[str]:
    return {f.rule_id for f in run_rules(report)}


def test_general_journal_cannot_forge_kernel_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _collect(monkeypatch, dmesg="[  1.0] usb 1-1: new device", journal=FORGED_JOURNAL)
    assert not _rule_ids(report) & KERNEL_RULES


@pytest.mark.parametrize("rule_id", sorted(GENUINE))
@pytest.mark.parametrize("source", ["dmesg", "journalctl -k"])
def test_genuine_kernel_lines_produce_findings(
    monkeypatch: pytest.MonkeyPatch, rule_id: str, source: str,
) -> None:
    message = GENUINE[rule_id]
    if source == "dmesg":
        report = _collect(monkeypatch, dmesg=f"[Mon Nov 16 10:00:00 2026] {message}")
    else:
        report = _collect(monkeypatch, kernel=f"Nov 16 10:00:00 spark kernel: {message}")
    findings = [f for f in run_rules(report) if f.rule_id == rule_id]
    assert findings and findings[0].severity == "critical"


def test_thermal_evidence_quotes_the_matching_line(monkeypatch: pytest.MonkeyPatch) -> None:
    message = GENUINE["thermal.shutdown_risk"]
    report = _collect(monkeypatch, kernel=f"Nov 16 10:00:00 spark kernel: {message}")
    finding = next(f for f in run_rules(report) if f.rule_id == "thermal.shutdown_risk")
    assert any(message in e for e in finding.evidence)


def test_same_event_in_dmesg_and_kernel_journal_is_reported_once(monkeypatch: pytest.MonkeyPatch) -> None:
    message = GENUINE["memory.oom_killer"]
    report = _collect(
        monkeypatch,
        dmesg=f"[Mon Nov 16 10:00:00 2026] {message}",
        kernel=f"Nov 16 10:00:00 spark kernel: {message}",
    )
    assert len(report.log_evidence or []) == 1


def test_kernel_events_beyond_the_saved_tail_are_found(monkeypatch: pytest.MonkeyPatch) -> None:
    filler = "\n".join(f"[{i}.0] eth0: link up" for i in range(5000))
    report = _collect(monkeypatch, dmesg=f"[0.5] {GENUINE['gpu.xid_error']}\n{filler}")
    assert "NVRM" not in report.logs["dmesg_tail"]
    assert "gpu.xid_error" in _rule_ids(report)


def test_unreadable_kernel_log_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _collect(monkeypatch, journal="Nov 16 10:00:00 spark systemd[1]: Started x.")
    assert any("kernel log unreadable" in e for e in report.collector_statuses[0].errors)


@pytest.mark.parametrize("line,fires", [
    ("Nov 16 10:00:00 spark python3[812]: vllm ERROR No available memory for the cache blocks.", True),
    ("Nov 16 10:00:00 spark python3[812]: ERROR No available memory for the cache blocks.", False),
])
def test_kv_cache_signature_requires_vllm_on_the_line(
    monkeypatch: pytest.MonkeyPatch, line: str, fires: bool,
) -> None:
    report = _collect(monkeypatch, journal=line)
    findings = [f for f in run_rules(report) if f.rule_id == "backend.kv_cache_oom"]
    assert bool(findings) is fires
    if fires:
        assert any(line in e and "journal" in e.lower() for e in findings[0].evidence)


@pytest.mark.parametrize("line,fires", [
    ("Oct 04 13:01:10 spark kernel: NVRM: Xid (PCI:000f:01:00): 79, GPU has fallen off the bus.", True),
    ("Oct 04 13:01:10 spark sshd[9]: Invalid user NVRM: Xid (PCI:0000:01:00): 79, x", False),
])
def test_legacy_journal_tail_counts_only_syslog_kernel_lines(line: str, fires: bool) -> None:
    assert ("gpu.xid_error" in _rule_ids(ScanReport(logs={"journal_tail": line}))) is fires


def test_legacy_scan_without_logs_keeps_stored_log_findings() -> None:
    data = json.loads((FIXTURES / "kernel_oom_and_xid.json").read_text())
    original = ScanReport.model_validate(data)
    stored = run_rules(original)
    stored.append(Finding(
        rule_id="thermal.shutdown_risk", title="Thermal", severity="critical",
        evidence=["dmesg_tail contains 'overheat'"],
    ))
    legacy = original.model_copy(update={"logs": {}, "findings": stored})

    first = run_rules(legacy)
    assert {(f.rule_id, f.severity) for f in first if f.rule_id in LOG_RULES} == {
        (f.rule_id, f.severity) for f in stored if f.rule_id in LOG_RULES
    }
    assert all(f.evidence.count(CARRIED_OVER_NOTE) == 1 for f in first if f.rule_id in LOG_RULES)
    again = run_rules(legacy.model_copy(update={"findings": first}))
    assert all(f.evidence.count(CARRIED_OVER_NOTE) == 1 for f in again if f.rule_id in LOG_RULES)


def test_temperature_only_thermal_finding_is_recomputed_not_carried() -> None:
    stored = Finding(
        rule_id="thermal.shutdown_risk", title="Thermal", severity="critical",
        evidence=["Peak GPU temperature 95.0 C"],
    )
    assert "thermal.shutdown_risk" not in _rule_ids(ScanReport(findings=[stored]))


def test_scan_with_evidence_does_not_carry_stale_findings() -> None:
    stored = Finding(rule_id="gpu.xid_error", title="Xid", severity="critical")
    assert "gpu.xid_error" not in _rule_ids(ScanReport(log_evidence=[], findings=[stored]))
