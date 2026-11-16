from __future__ import annotations

import re
from collections.abc import Iterator

from .. import log_evidence
from ..log_evidence import OOM_KILL, XID, Signal
from ..models import Finding, ScanReport
from .engine import Rule

# Xids NVIDIA documents as driver, firmware or hardware faults. The rest (13, 31, 43, 45, ...)
# are usually raised by an application bug or its teardown, so they warrant less alarm.
_HARDWARE_XIDS = {48, 61, 62, 63, 64, 74, 79, 92, 94, 95, 119, 120}


def _log_matches(report: ScanReport, signal: Signal, pattern: re.Pattern[str]) -> Iterator[re.Match[str]]:
    """Yield each distinct kernel event once: dmesg and the kernel journal carry the same
    event with different prefixes."""
    seen: set[str] = set()
    for evidence in log_evidence.lines_for(report, signal):
        match = pattern.search(evidence.line)
        if match and match.group(0) not in seen:
            seen.add(match.group(0))
            yield match


def _evaluate_oom(report: ScanReport) -> list[Finding]:
    kills = list(_log_matches(report, "oom_kill", OOM_KILL))
    if not kills:
        return log_evidence.carried_over(report, "memory.oom_killer")

    host_kills = [m for m in kills if not m.group("cgroup")]
    evidence = [
        f"{'Container memory limit' if m.group('cgroup') else 'Kernel OOM killer'} killed "
        f"{m.group('name')} (pid {m.group('pid')})"
        for m in kills[:5]
    ]
    if len(kills) > 5:
        evidence.append(f"{len(kills) - 5} more OOM kill(s) since boot")

    if host_kills:
        explanation = (
            "The Linux kernel ran out of memory and killed a process to recover. On DGX Spark "
            "the CPU and GPU share one 128 GB memory pool, so a model server, its KV cache and "
            "everything else on the box all draw from the same budget."
        )
        actions = [
            "Check whether the killed process was your model server; if so, it is no longer running.",
            "Lower vLLM --gpu-memory-utilization or --max-model-len, or use a smaller model.",
            "Stop other models or heavy processes before starting the server again.",
            "Re-run `spark-doctor scan` while the workload runs to see current memory pressure.",
        ]
    else:
        explanation = (
            "A container or service hit its own memory limit (a cgroup limit, such as "
            "`docker run --memory`) and the kernel killed a process inside it. The host as a "
            "whole may still have free memory."
        )
        actions = [
            "Raise or remove the memory limit on the container or service that was killed.",
            "Or reduce the workload's memory use (model size, context length, batch size).",
        ]

    return [
        Finding(
            rule_id="memory.oom_killer",
            title="The kernel killed a process for running out of memory",
            severity="critical" if host_kills else "warning",
            confidence="high",
            evidence=evidence,
            explanation=explanation,
            recommended_actions=actions,
            source_note="Kernel 'Out of memory: Killed process' messages in this boot's kernel log.",
        )
    ]


def _evaluate_xid(report: ScanReport) -> list[Finding]:
    events = list(_log_matches(report, "xid", XID))
    if not events:
        return log_evidence.carried_over(report, "gpu.xid_error")

    codes = sorted({int(m.group("code")) for m in events})
    hardware = [c for c in codes if c in _HARDWARE_XIDS]
    evidence = [f"Xid {m.group('code')}{m.group('rest').rstrip()[:160]}" for m in events[:5]]
    if len(events) > 5:
        evidence.append(f"{len(events) - 5} more Xid event(s) since boot")

    if hardware:
        explanation = (
            f"The NVIDIA driver reported Xid {', '.join(map(str, hardware))}, which NVIDIA "
            "documents as a driver, firmware or hardware-level GPU fault rather than an "
            "application bug. GPU work may fail until the system is restarted."
        )
        actions = [
            "Save your work and reboot the Spark.",
            "Install pending DGX OS and firmware updates (DGX Dashboard).",
            "Run `spark-doctor scan` again after the reboot and check whether the Xid returns.",
        ]
        escalation = [
            "If the same Xid returns after a reboot and updates, run NVIDIA Field Diagnostics "
            "and include this report in a forum or support post.",
        ]
    else:
        explanation = (
            f"The NVIDIA driver reported Xid {', '.join(map(str, codes))}. These codes are usually "
            "caused by an application error, such as an illegal memory access in a CUDA kernel "
            "or a process killed mid-run, rather than a hardware fault."
        )
        actions = [
            "Check the logs of the program that was using the GPU at that time for a CUDA error.",
            "Retry the workload; if it fails the same way, update the framework or container image.",
        ]
        escalation = []

    return [
        Finding(
            rule_id="gpu.xid_error",
            title="NVIDIA driver reported GPU errors (Xid)",
            severity="critical" if hardware else "warning",
            confidence="high" if hardware else "medium",
            evidence=evidence,
            explanation=explanation,
            recommended_actions=actions,
            escalation_actions=escalation,
            source_note="NVIDIA Xid error catalog; 'NVRM: Xid' messages in this boot's kernel log.",
        )
    ]


rule_memory_oom_killer = Rule(
    id="memory.oom_killer",
    title="Kernel OOM killer fired",
    fn=_evaluate_oom,
)

rule_gpu_xid_error = Rule(
    id="gpu.xid_error",
    title="NVIDIA Xid error in kernel log",
    fn=_evaluate_xid,
)
