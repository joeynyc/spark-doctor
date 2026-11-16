import json
from pathlib import Path

import pytest

from spark_doctor.models import MemorySnapshot, ScanReport
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def test_memory_pressure_critical():
    data = json.loads((FIXTURES / "memory_pressure.json").read_text())
    report = ScanReport.model_validate(data)
    findings = run_rules(report)
    mem = [f for f in findings if f.rule_id == "memory.uma_pressure"]
    assert mem
    assert mem[0].severity == "critical"


def test_healthy_no_memory_finding():
    data = json.loads((FIXTURES / "healthy_minimal.json").read_text())
    report = ScanReport.model_validate(data)
    findings = run_rules(report)
    assert not any(f.rule_id == "memory.uma_pressure" for f in findings)


def _psi_report(some: float, full: float) -> ScanReport:
    # 100 GB of 128 GB free: only PSI can trigger the rule.
    return ScanReport(
        memory=MemorySnapshot(
            mem_total_kb=128 * 1024 * 1024,
            mem_available_kb=100 * 1024 * 1024,
            swap_total_kb=0,
            swap_free_kb=0,
            psi_memory={"some": {"avg10": some}, "full": {"avg10": full}},
        )
    )


@pytest.mark.parametrize(
    ("some", "full", "severity"),
    [
        (0.6, 0.3, None),
        (9.9, 0.99, None),
        (12.0, 0.0, "warning"),
        (2.0, 1.5, "warning"),
        (8.0, 6.0, "critical"),
    ],
)
def test_psi_thresholds_treat_avg10_as_percent(some: float, full: float, severity: str | None) -> None:
    findings = [f for f in run_rules(_psi_report(some, full)) if f.rule_id == "memory.uma_pressure"]
    assert (findings[0].severity if findings else None) == severity


def test_psi_some_scope_reported_when_it_triggers() -> None:
    finding = next(f for f in run_rules(_psi_report(12.0, 0.0)) if f.rule_id == "memory.uma_pressure")
    assert any("some" in line and "12.0%" in line for line in finding.evidence)
