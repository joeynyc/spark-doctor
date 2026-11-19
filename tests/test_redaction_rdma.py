from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from spark_doctor.cli import app
from spark_doctor.models import Finding, ScanReport
from spark_doctor.privacy import redact_obj, redact_report, redact_text


@pytest.mark.parametrize("label", ["Node GUID", "Port GUID", "System image GUID"])
@pytest.mark.parametrize("value", ["0x0123456789abcdef", "0123:4567:89ab:cdef"])
@pytest.mark.parametrize("include_network", [False, True])
def test_rdma_guids_remain_private_with_network_opt_in(
    label: str, value: str, include_network: bool,
) -> None:
    text = f"\t{label}: {value}\n\tState: Active\n\tRate: 200\n"
    result = redact_text(text, include_network_identifiers=include_network)
    assert result == f"\t{label}: <redacted:hardware_id>\n\tState: Active\n\tRate: 200\n"
    assert redact_text(result, include_network_identifiers=include_network) == result


@pytest.mark.parametrize("key", ["node_guid", "port-guid", "system_image_guid", "Node GUID"])
@pytest.mark.parametrize("include_network", [False, True])
def test_structured_rdma_guids_are_hardware_identifiers(key: str, include_network: bool) -> None:
    source = {"network": {key: "0x0123456789abcdef", "state": "Active", "rate": 200}}
    result = redact_obj(source, include_network_identifiers=include_network)
    assert result == {"network": {key: "<redacted:hardware_id>", "state": "Active", "rate": 200}}
    assert source["network"][key] == "0x0123456789abcdef"


def test_device_class_guids_and_diagnostic_hex_values_are_preserved() -> None:
    text = "GUID: 01234567-89ab-cdef-0123-456789abcdef\nCapability mask: 0x00010000"
    assert redact_text(text) == text
    assert redact_obj({"guid": "device-class-guid", "capability_mask": "0x00010000"}) == {
        "guid": "device-class-guid", "capability_mask": "0x00010000",
    }


@pytest.mark.parametrize("include_network", [False, True])
def test_network_report_redaction_is_idempotent(include_network: bool) -> None:
    report = ScanReport(
        anonymized=False,
        network={"ibstat": "Node GUID: 0x0123456789abcdef\nPort GUID: 0xfedcba9876543210"},
    )
    result = redact_report(report, include_network_identifiers=include_network)
    assert result.network["ibstat"].count("<redacted:hardware_id>") == 2
    assert result.anonymized
    assert redact_report(result, include_network_identifiers=include_network) == result
    assert "0x0123456789abcdef" in report.network["ibstat"]


def test_imported_rdma_guids_are_redacted_in_json_and_report_exports(tmp_path: Path) -> None:
    guid = "0x0123456789abcdef"
    report = ScanReport(
        anonymized=False,
        network={"ibstat": f"Node GUID: {guid}\nState: Active"},
        findings=[Finding(
            rule_id="test.network", title="Synthetic network evidence", severity="info",
            evidence=[f"Port GUID: {guid}"],
        )],
    )
    source = tmp_path / "source.json"
    source.write_text(report.model_dump_json())
    output = tmp_path / "redacted.json"
    runner = CliRunner()
    result = runner.invoke(app, ["anonymize", str(source), "--out", str(output)])
    assert result.exit_code == 0, result.output
    serialized = output.read_text()
    assert guid not in serialized
    assert "<redacted:hardware_id>" in serialized
    for format in ("markdown", "forum", "github"):
        result = runner.invoke(app, ["report", "--from", str(source), "--format", format])
        assert result.exit_code == 0, result.output
        assert guid not in result.output
        assert "<redacted:hardware_id>" in result.output
