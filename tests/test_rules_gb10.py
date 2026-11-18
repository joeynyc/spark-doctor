import json
from pathlib import Path

import pytest

from spark_doctor.models import ScanReport
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> ScanReport:
    data = json.loads((FIXTURES / name).read_text())
    return ScanReport.model_validate(data)


def _ids(report: ScanReport) -> dict:
    return {f.rule_id: f for f in run_rules(report)}


# --- nemotron_v3 reasoning-parser --------------------------------------------

def test_nemotron_v3_on_0221_flags():
    f = _ids(_load("nemotron_v3_0221.json"))
    assert "backend.nemotron_v3_discards_primed_reasoning" in f
    assert f["backend.nemotron_v3_discards_primed_reasoning"].severity == "warning"


def test_nemotron_v3_on_0221_local_build_flags():
    # 0.22.1+cu128 must still match (prefix-match on the part before '+').
    f = _ids(_load("nemotron_v3_0221_cu128.json"))
    assert "backend.nemotron_v3_discards_primed_reasoning" in f


def test_nemotron_v3_other_version_no_finding():
    f = _ids(_load("nemotron_v3_other_version.json"))
    assert "backend.nemotron_v3_discards_primed_reasoning" not in f


@pytest.mark.parametrize(
    "args",
    [
        "/other/venv/bin/vllm serve nvidia/model --reasoning-parser nemotron_v3",
        "vllm serve nvidia/model --reasoning-parser nemotron_v3",
    ],
)
def test_nemotron_v3_process_outside_inspected_env_no_finding(args: str) -> None:
    data = json.loads((FIXTURES / "nemotron_v3_0221.json").read_text())
    data["processes"][0]["args"] = args
    f = _ids(ScanReport.model_validate(data))
    assert "backend.nemotron_v3_discards_primed_reasoning" not in f


def test_nemotron_v3_links_via_selected_python_executable() -> None:
    data = json.loads((FIXTURES / "nemotron_v3_0221.json").read_text())
    del data["cuda_env"]["python"]["executable"]
    data["cuda_env"]["python_executable"] = "/opt/venv/bin/python3"
    data["processes"][0]["args"] = (
        "/opt/venv/bin/python3 -m vllm.entrypoints.openai.api_server --reasoning-parser nemotron_v3"
    )
    f = _ids(ScanReport.model_validate(data))
    assert "backend.nemotron_v3_discards_primed_reasoning" in f


# --- aarch64 + GB10 prebuilt-wheel gap ---------------------------------------

def test_aarch64_wheel_missing_flags():
    f = _ids(_load("aarch64_wheel_gap_missing.json"))
    assert "cuda.aarch64_prebuilt_wheel_gap" in f
    finding = f["cuda.aarch64_prebuilt_wheel_gap"]
    assert finding.severity == "info"
    assert any("flash_attn" in e for e in finding.evidence)


def test_aarch64_wheel_cpu_only_flags():
    f = _ids(_load("aarch64_wheel_gap_cpu_only.json"))
    assert "cuda.aarch64_prebuilt_wheel_gap" in f
    finding = f["cuda.aarch64_prebuilt_wheel_gap"]
    assert any("CPU-only" in e for e in finding.evidence)


def test_aarch64_wheel_gap_non_gb10_silent():
    # Same optional-package signals but not a GB10 GPU → no finding.
    data = json.loads((FIXTURES / "aarch64_wheel_gap_missing.json").read_text())
    data["gpu"] = {"available": True, "gpu_count": 1, "name": "NVIDIA RTX 4090"}
    report = ScanReport.model_validate(data)
    assert "cuda.aarch64_prebuilt_wheel_gap" not in _ids(report)


# --- split preserves isolation ------------------------------------------------

def test_split_into_separate_rules():
    from spark_doctor.rules.engine import ALL_RULES

    ids = {r.id for r in ALL_RULES}
    assert "gb10.local_inference" not in ids
    assert {"gb10.nemotron_v3_parser", "gb10.aarch64_wheel_gap"} <= ids
