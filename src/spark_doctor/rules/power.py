from __future__ import annotations

from ..models import Finding, ScanReport
from .engine import Rule


def _evaluate(report: ScanReport) -> list[Finding]:
    samples = report.gpu_samples
    if not samples:
        return []

    matching = [
        s for s in samples
        if (s.gpu_utilization_percent is not None and s.gpu_utilization_percent >= 80)
        and (s.gpu_power_draw_watts is not None and s.gpu_power_draw_watts <= 25)
    ]
    if not matching:
        return []

    low_clock = any(
        (s.gpu_clock_mhz is not None and s.gpu_clock_mhz <= 800) for s in matching
    )

    sustained = len(matching) >= 3
    severity = "critical" if sustained else "warning"
    confidence = "high" if (sustained and low_clock) else ("medium" if sustained else "low")

    evidence = []
    for s in matching[:5]:
        util = s.gpu_utilization_percent
        power = s.gpu_power_draw_watts
        clock = s.gpu_clock_mhz
        evidence.append(
            f"GPU util {util:.0f}%, power {power:.1f} W, clock {clock if clock is not None else 'n/a'} MHz"
        )

    return [
        Finding(
            rule_id="power.low_draw_under_load",
            title="Possible GPU low-power state",
            severity=severity,
            confidence=confidence,
            evidence=evidence,
            explanation=(
                "Your GPU looks busy but is drawing unusually low power. This often means the "
                "system is not entering the expected performance state."
            ),
            recommended_actions=[
                "Save all running work.",
                "Shut down the Spark.",
                "Unplug the power brick from the wall and from the Spark.",
                "Wait 60 seconds.",
                "Plug in and boot.",
                "Run `spark-doctor scan` again.",
                "Check DGX Dashboard for updates.",
            ],
            escalation_actions=[
                "If this repeats after updates, run NVIDIA Field Diagnostics and include this Spark Doctor report in a forum or support post.",
            ],
            source_note="DGX Spark forum reports of 14 W power cap and low-power states under load.",
        )
    ]


rule_power_low_draw_under_load = Rule(
    id="power.low_draw_under_load",
    title="Possible GPU low-power state",
    fn=_evaluate,
)
