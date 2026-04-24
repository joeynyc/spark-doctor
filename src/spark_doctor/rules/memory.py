from __future__ import annotations

from ..models import Finding, ScanReport
from .engine import Rule


def _evaluate(report: ScanReport) -> list[Finding]:
    mem = report.memory
    if mem is None:
        return []

    warn = False
    critical = False
    evidence: list[str] = []

    mem_avail_gb = None
    mem_total_gb = None
    if mem.mem_available_kb is not None:
        mem_avail_gb = mem.mem_available_kb / (1024 * 1024)
        evidence.append(f"MemAvailable {mem_avail_gb:.1f} GB")
    if mem.mem_total_kb is not None:
        mem_total_gb = mem.mem_total_kb / (1024 * 1024)
        evidence.append(f"MemTotal {mem_total_gb:.1f} GB")

    psi_full_avg10 = None
    if isinstance(mem.psi_memory, dict):
        full = mem.psi_memory.get("full")
        if isinstance(full, dict):
            psi_full_avg10 = full.get("avg10")
            if psi_full_avg10 is not None:
                evidence.append(f"memory PSI full avg10 {psi_full_avg10:.2f}")

    swap_used_gb = None
    if mem.swap_total_kb is not None and mem.swap_free_kb is not None:
        swap_used_kb = mem.swap_total_kb - mem.swap_free_kb
        swap_used_gb = swap_used_kb / (1024 * 1024)
        if swap_used_gb > 0.1:
            evidence.append(f"swap used {swap_used_gb:.1f} GB")

    if mem_avail_gb is not None:
        if mem_avail_gb < 8:
            critical = True
        elif mem_avail_gb < 16:
            warn = True
        if mem_total_gb and mem_avail_gb / mem_total_gb < 0.15:
            warn = True

    if psi_full_avg10 is not None:
        if psi_full_avg10 > 0.25:
            critical = True
        elif psi_full_avg10 > 0.10:
            warn = True

    if swap_used_gb is not None and swap_used_gb > 8:
        warn = True

    if not (warn or critical):
        return []

    return [
        Finding(
            rule_id="memory.uma_pressure",
            title="Unified memory pressure elevated",
            severity="critical" if critical else "warning",
            confidence="medium",
            evidence=evidence,
            explanation=(
                "Available memory is low or the kernel reports memory pressure. "
                "On DGX Spark, unified memory is shared between CPU and GPU, so pressure here "
                "can stall inference workloads."
            ),
            recommended_actions=[
                "Reduce vLLM gpu_memory_utilization (start around 0.80).",
                "Reduce context length or max_model_len.",
                "Use smaller KV cache settings where appropriate.",
                "Unload idle models.",
                "Avoid running multiple heavy backends simultaneously.",
                "Treat swap as a safety net, not a performance fix.",
            ],
            source_note="DGX Spark unified memory behavior, Linux PSI signals.",
        )
    ]


rule_memory_uma_pressure = Rule(
    id="memory.uma_pressure",
    title="Unified memory pressure elevated",
    fn=_evaluate,
)
