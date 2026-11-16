from __future__ import annotations

from ..models import Finding, MetricSample, ScanReport
from .engine import Rule


RULE_ID = "power.low_draw_under_load"
LOW_CLOCK_MHZ = 800
# A stuck unit pins the clock near 800 MHz; memory-bound decode on a healthy GB10 sits near
# boost clock while drawing the same ~15-20 W. Only the clock separates the two.
HEALTHY_CLOCK_MHZ = 1000


def _consecutive(previous: MetricSample, current: MetricSample) -> bool:
    if previous.timestamp is None or current.timestamp is None:
        return previous.timestamp is None and current.timestamp is None
    try:
        gap = (current.timestamp - previous.timestamp).total_seconds()
    except TypeError:
        return False
    # Collectors poll at one-second intervals; long gaps cannot establish persistence.
    return 0 < gap <= 5


def _run_evidence(run: list[MetricSample]) -> list[str]:
    evidence = [f"Longest qualifying run: {len(run)} consecutive samples."]
    for s in run[:5]:
        clock = f"{s.gpu_clock_mhz:.0f}" if s.gpu_clock_mhz is not None else "n/a"
        evidence.append(
            f"GPU util {s.gpu_utilization_percent:.0f}%, power {s.gpu_power_draw_watts:.1f} W, "
            f"clock {clock} MHz"
        )
    return evidence


def _healthy_clock_finding(evidence: list[str]) -> Finding:
    return Finding(
        rule_id=RULE_ID,
        title="Low GPU power draw with a normal clock",
        severity="info",
        confidence="medium",
        evidence=evidence,
        explanation=(
            "Utilization is high and power draw is low, but the GPU clock is at a normal level. "
            "That is expected for memory-bound work such as single-request LLM decode, and does "
            "not look like the stuck low-power state, which pins the clock low."
        ),
        recommended_actions=[
            "No action needed if generation speed (tokens/s) matches what you normally see.",
            "If speed is below your known baseline, re-run `spark-doctor scan --sample-seconds 30` under load and check whether the clock drops.",
        ],
        source_note="Healthy GB10 decode draws ~15-20 W at boost clock; the stuck state shows the same draw at ~800 MHz.",
    )


def _evaluate(report: ScanReport) -> list[Finding]:
    current_run: list[MetricSample] = []
    longest_run: list[MetricSample] = []
    for sample in report.gpu_samples:
        if not (
            sample.gpu_utilization_percent is not None and sample.gpu_utilization_percent >= 80
            and sample.gpu_power_draw_watts is not None and sample.gpu_power_draw_watts <= 25
        ):
            current_run = []
            continue
        if current_run and not _consecutive(current_run[-1], sample):
            current_run = []
        current_run.append(sample)
        if len(current_run) > len(longest_run):
            longest_run = current_run
    if not longest_run:
        return []

    clocks = [s.gpu_clock_mhz for s in longest_run]
    low_clock = all(c is not None and c <= LOW_CLOCK_MHZ for c in clocks)
    healthy_clock = all(c is not None and c > HEALTHY_CLOCK_MHZ for c in clocks)
    evidence = _run_evidence(longest_run)

    if healthy_clock:
        return [_healthy_clock_finding(evidence)]

    sustained = len(longest_run) >= 3
    confirmed = sustained and low_clock

    missing = sum(c is None for c in clocks)
    if missing:
        evidence.insert(
            1,
            f"GPU clock was not reported for {missing} of {len(clocks)} samples, so a low-power "
            "state cannot be confirmed or ruled out.",
        )
    if sustained and not low_clock:
        not_low = sum(c is not None and c > LOW_CLOCK_MHZ for c in clocks)
        if not_low:
            evidence.insert(
                1,
                f"GPU clock was above {LOW_CLOCK_MHZ} MHz in {not_low} of {len(clocks)} samples; "
                "the stuck low-power state pins every sample at or below that, so it is not confirmed.",
            )

    if confirmed:
        explanation = (
            "Your GPU reports high utilization, unusually low power, and a clock pinned at or below "
            f"{LOW_CLOCK_MHZ} MHz across consecutive samples. This matches the reported low-power "
            "state, but does not establish the cause."
        )
    elif sustained:
        explanation = (
            "Your GPU reports high utilization and unusually low power across consecutive samples, "
            "but the GPU clock was missing or not consistently low, so these readings do not "
            "confirm a stuck low-power state or its cause."
        )
    else:
        explanation = (
            "Your GPU reports high utilization and unusually low power in brief or separated "
            "samples. These readings do not establish a sustained low-power state or its cause."
        )

    return [
        Finding(
            rule_id=RULE_ID,
            title="Possible GPU low-power state",
            severity="critical" if confirmed else "warning",
            confidence="high" if confirmed else ("medium" if sustained else "low"),
            evidence=evidence,
            explanation=explanation,
            recommended_actions=[
                "Confirm the readings repeat during a sustained GPU workload before attempting recovery.",
                "Save all running work.",
                "Shut down the Spark.",
                "Unplug the power brick from the wall and from the Spark.",
                "Wait 60 seconds.",
                "Plug in and boot.",
                "Run `spark-doctor scan` again.",
                "Check DGX Dashboard for updates.",
            ] if confirmed else [
                "Confirm a sustained GPU workload is running; short or intermittent load can be misleading.",
                "Collect more samples with `spark-doctor scan --sample-seconds 30` while the workload runs.",
                "Compare power and clock readings across the samples before deciding on recovery steps.",
            ],
            escalation_actions=[
                "If this repeats after updates, run NVIDIA Field Diagnostics and include this Spark Doctor report in a forum or support post.",
            ] if confirmed else [],
            source_note="DGX Spark forum reports of 14 W power cap and low-power states under load.",
        )
    ]


rule_power_low_draw_under_load = Rule(
    id=RULE_ID,
    title="Possible GPU low-power state",
    fn=_evaluate,
)
