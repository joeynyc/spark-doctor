from __future__ import annotations

from ..models import Finding, ScanReport
from .engine import Rule

THERMAL_LOG_KEYWORDS = ("thermal shutdown", "over temperature", "overheat", "thermal event", "critical temperature")


def _evaluate(report: ScanReport) -> list[Finding]:
    samples = report.gpu_samples
    if not samples:
        # Still check logs even without samples
        pass

    max_temp = None
    for s in samples:
        if s.gpu_temperature_c is not None:
            if max_temp is None or s.gpu_temperature_c > max_temp:
                max_temp = s.gpu_temperature_c

    log_hits: list[str] = []
    for key in ("dmesg_tail", "journal_tail"):
        text = report.logs.get(key) if isinstance(report.logs, dict) else None
        if not text:
            continue
        low = text.lower()
        for kw in THERMAL_LOG_KEYWORDS:
            if kw in low:
                log_hits.append(f"{key} contains '{kw}'")

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
