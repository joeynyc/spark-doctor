from __future__ import annotations

import re

from .. import log_evidence
from ..models import Finding, ScanReport
from .engine import Rule

# Evidence format written by spark-doctor versions that matched thermal keywords in raw tails.
_LEGACY_LOG_EVIDENCE = re.compile(r"^(?:dmesg_tail|journal_tail) contains '")


def _had_log_evidence(finding: Finding) -> bool:
    return any(_LEGACY_LOG_EVIDENCE.match(e) for e in finding.evidence)


def _evaluate(report: ScanReport) -> list[Finding]:
    carried = log_evidence.carried_over(report, "thermal.shutdown_risk", _had_log_evidence)
    if carried:
        return carried

    max_temp = None
    for s in report.gpu_samples:
        if s.gpu_temperature_c is not None:
            if max_temp is None or s.gpu_temperature_c > max_temp:
                max_temp = s.gpu_temperature_c

    log_hits = [f"Kernel log: {e.line[:200]}" for e in log_evidence.lines_for(report, "thermal")]

    if max_temp is None and not log_hits:
        return []

    warn = max_temp is not None and max_temp >= 85
    critical = (max_temp is not None and max_temp >= 90) or bool(log_hits)

    if not (warn or critical):
        return []

    severity = "critical" if critical else "warning"
    evidence: list[str] = []
    if max_temp is not None:
        evidence.append(f"Peak GPU temperature {max_temp:.1f} C")
    evidence.extend(log_hits[:5])
    if len(log_hits) > 5:
        evidence.append(f"{len(log_hits) - 5} more thermal kernel message(s) since boot")

    return [
        Finding(
            rule_id="thermal.shutdown_risk",
            title="Thermal risk or shutdown-like behavior",
            severity=severity,
            confidence="medium",
            evidence=evidence,
            explanation=(
                "The GPU temperature is high or logs suggest a thermal event. "
                "Sustained high temperatures can lead to throttling or automatic shutdown."
            ),
            recommended_actions=[
                "Stop or reduce the current workload.",
                "Improve airflow and clearance around the device.",
                "Confirm ambient temperature is within supported range.",
                "Re-run `spark-doctor scan` under load after cooling.",
            ],
            escalation_actions=[
                "If shutdowns persist, run NVIDIA Field Diagnostics and consider an RMA or support ticket.",
            ],
            source_note="DGX Spark forum report of temperature-triggered shutdowns.",
        )
    ]


rule_thermal_shutdown_risk = Rule(
    id="thermal.shutdown_risk",
    title="Thermal risk or shutdown-like behavior",
    fn=_evaluate,
)
