import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from spark_doctor.models import MetricSample, ScanReport
from spark_doctor.rules import run_rules
from spark_doctor.rules.power import rule_power_low_draw_under_load

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


def _low_sample(timestamp: datetime | None = None, clock: float | None = 611) -> MetricSample:
    return MetricSample(
        timestamp=timestamp,
        gpu_utilization_percent=95,
        gpu_power_draw_watts=14,
        gpu_clock_mhz=clock,
    )


@pytest.mark.parametrize(
    "break_sample",
    [
        MetricSample(gpu_utilization_percent=95, gpu_power_draw_watts=80, gpu_clock_mhz=1500),
        MetricSample(gpu_utilization_percent=10, gpu_power_draw_watts=14, gpu_clock_mhz=611),
        MetricSample(),
    ],
)
def test_interrupted_low_power_is_warning_with_resampling_advice(break_sample: MetricSample) -> None:
    report = ScanReport(gpu_samples=[_low_sample(), _low_sample(), break_sample, _low_sample()])
    finding = rule_power_low_draw_under_load.fn(report)[0]
    assert finding.severity == "warning"
    assert finding.confidence == "low"
    advice = " ".join(finding.recommended_actions).lower()
    assert "sample" in advice and "load" in advice
    assert "unplug" not in advice and "shut down" not in advice
    assert "cause" in finding.explanation.lower()


def test_three_legacy_consecutive_samples_remain_critical() -> None:
    report = ScanReport(gpu_samples=[_low_sample(), _low_sample(), _low_sample()])
    finding = rule_power_low_draw_under_load.fn(report)[0]
    assert finding.severity == "critical"


@pytest.mark.parametrize("seconds", [(0, 1, 60), (0, 0, 0), (2, 1, 0)])
def test_gaps_or_non_increasing_timestamps_cannot_establish_persistence(seconds: tuple[int, ...]) -> None:
    start = datetime(2026, 4, 24, tzinfo=timezone.utc)
    report = ScanReport(gpu_samples=[_low_sample(start + timedelta(seconds=s)) for s in seconds])
    finding = rule_power_low_draw_under_load.fn(report)[0]
    assert finding.severity == "warning"
    assert finding.confidence == "low"


def test_mixed_timestamp_awareness_does_not_raise_or_claim_persistence() -> None:
    start = datetime(2026, 4, 24)
    report = ScanReport(
        gpu_samples=[
            _low_sample(start),
            _low_sample((start + timedelta(seconds=1)).replace(tzinfo=timezone.utc)),
            _low_sample(start + timedelta(seconds=2)),
        ]
    )
    assert rule_power_low_draw_under_load.fn(report)[0].severity == "warning"


def test_sustained_evidence_comes_from_consecutive_run() -> None:
    isolated = _low_sample()
    isolated.gpu_utilization_percent = 81
    report = ScanReport(
        gpu_samples=[isolated, MetricSample(), _low_sample(), _low_sample(), _low_sample()]
    )
    finding = rule_power_low_draw_under_load.fn(report)[0]
    assert finding.severity == "critical"
    assert finding.confidence == "high"
    assert not any("81%" in line for line in finding.evidence)
    assert any("3 consecutive" in line for line in finding.evidence)


def test_low_draw_with_healthy_clock_is_info_without_power_cycle_advice() -> None:
    report = ScanReport(gpu_samples=[_low_sample(clock=1982) for _ in range(5)])
    finding = rule_power_low_draw_under_load.fn(report)[0]
    assert finding.severity == "info"
    advice = " ".join(finding.recommended_actions).lower()
    assert "unplug" not in advice and "shut down" not in advice
    assert not finding.escalation_actions


def test_low_draw_with_low_clock_stays_critical() -> None:
    report = ScanReport(gpu_samples=[_low_sample(clock=799) for _ in range(3)])
    finding = rule_power_low_draw_under_load.fn(report)[0]
    assert finding.severity == "critical"
    assert finding.confidence == "high"


@pytest.mark.parametrize(
    "clocks",
    [
        [2400, 2400, 790],
        [None, None, None],
        [611, None, 611],
        [900, 900, 900],
    ],
)
def test_sustained_low_draw_without_all_low_clocks_is_unconfirmed_warning(
    clocks: list[float | None],
) -> None:
    report = ScanReport(gpu_samples=[_low_sample(clock=c) for c in clocks])
    finding = rule_power_low_draw_under_load.fn(report)[0]
    assert finding.severity == "warning"
    assert finding.confidence != "high"
    advice = " ".join(finding.recommended_actions).lower()
    assert "sample" in advice
    assert "unplug" not in advice and "shut down" not in advice
    assert not finding.escalation_actions
