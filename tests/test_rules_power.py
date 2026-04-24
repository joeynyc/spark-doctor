import json
from pathlib import Path

from spark_doctor.models import ScanReport
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> ScanReport:
    data = json.loads((FIXTURES / name).read_text())
    return ScanReport.model_validate(data)


def test_power_low_draw_triggers_critical():
    report = _load("power_limited_14w.json")
    findings = run_rules(report)
    ids = {f.rule_id: f for f in findings}
    assert "power.low_draw_under_load" in ids
    assert ids["power.low_draw_under_load"].severity == "critical"


def test_healthy_no_power_finding():
    report = _load("healthy_minimal.json")
    findings = run_rules(report)
    assert not any(f.rule_id == "power.low_draw_under_load" for f in findings)


def test_thermal_fixture_triggers_critical():
    report = _load("thermal_shutdown_like.json")
    findings = run_rules(report)
    thermal = [f for f in findings if f.rule_id == "thermal.shutdown_risk"]
    assert thermal, "expected thermal.shutdown_risk finding"
    assert thermal[0].severity == "critical"
