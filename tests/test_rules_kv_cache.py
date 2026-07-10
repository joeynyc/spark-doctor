import json
from pathlib import Path

from spark_doctor.models import ScanReport
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _findings(name: str):
    data = json.loads((FIXTURES / name).read_text())
    report = ScanReport.model_validate(data)
    return run_rules(report)


def test_kv_cache_oom_detected_critical():
    findings = _findings("kv_cache_oom.json")
    kv = [f for f in findings if f.rule_id == "backend.kv_cache_oom"]
    assert kv
    assert kv[0].severity == "critical"
    assert any("enforce-eager" in a for a in kv[0].recommended_actions)


def test_kv_cache_oom_is_not_reported_as_host_memory_pressure():
    # The fixture has ample MemAvailable and zero PSI, so the host-memory rule must
    # stay silent — this failure is GPU KV/graph memory, a different diagnosis.
    findings = _findings("kv_cache_oom.json")
    assert not any(f.rule_id == "memory.uma_pressure" for f in findings)


def test_healthy_no_kv_cache_finding():
    findings = _findings("healthy_minimal.json")
    assert not any(f.rule_id == "backend.kv_cache_oom" for f in findings)
