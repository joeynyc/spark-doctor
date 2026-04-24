import json
from pathlib import Path

from spark_doctor.models import ScanReport
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
