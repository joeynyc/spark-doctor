from __future__ import annotations

from ..models import Finding, ProcessInfo, ScanReport
from .engine import Rule

HEAVY_BACKENDS = {"vllm", "sglang", "llama.cpp", "ollama", "comfyui"}
HEAVY_RSS_KB = 1_000_000  # ~1 GB


def _instance_root(proc: ProcessInfo, by_pid: dict[int, ProcessInfo]) -> int:
    """Pid of the topmost listed ancestor sharing proc's backend (proc itself if none)."""
    root = proc.pid
    seen = {proc.pid}
    ppid = proc.ppid
    while ppid is not None and ppid not in seen and ppid in by_pid:
        seen.add(ppid)
        parent = by_pid[ppid]
        # Ancestors of any RSS count: a small vLLM API server still owns its heavy EngineCore children.
        if parent.detected_backend == proc.detected_backend:
            root = parent.pid
        ppid = parent.ppid
    return root


def _evaluate(report: ScanReport) -> list[Finding]:
    heavy = [
        p for p in report.processes
        if p.detected_backend in HEAVY_BACKENDS and (p.rss_kb or 0) >= HEAVY_RSS_KB
    ]
    # Scans saved before ppid was collected cannot tell workers from separate servers.
    legacy = all(p.ppid is None for p in report.processes)
    by_pid = {p.pid: p for p in report.processes}

    groups: dict[str, dict[int, list[ProcessInfo]]] = {}
    for p in heavy:
        root = 0 if legacy else _instance_root(p, by_pid)
        groups.setdefault(p.detected_backend or "", {}).setdefault(root, []).append(p)
    if sum(len(instances) for instances in groups.values()) < 2:
        return []

    evidence = []
    total_rss_gb = 0.0
    for name, instances in groups.items():
        procs = [p for members in instances.values() for p in members]
        rss_gb = sum((p.rss_kb or 0) for p in procs) / (1024 * 1024)
        total_rss_gb += rss_gb
        instance_text = "" if legacy else f"{len(instances)} instance(s), "
        evidence.append(f"{name}: {instance_text}{len(procs)} process(es), {rss_gb:.1f} GB RSS")
    evidence.append(f"Total heavy backend RSS: {total_rss_gb:.1f} GB")

    return [
        Finding(
            rule_id="backend.multiple_heavy_models",
            title="Multiple heavy model servers running",
            severity="warning",
            confidence="medium",
            evidence=evidence,
            explanation=(
                "Two or more heavy model servers are running at the same time, either different "
                "backends or separate instances of the same one. "
                "They will compete for unified memory and GPU time."
            ),
            recommended_actions=[
                "Stop idle model servers.",
                "Use a model swapper or router instead of keeping all models resident.",
                "Re-run `spark-doctor scan` with only one workload active.",
            ],
        )
    ]


rule_backend_multiple_heavy_models = Rule(
    id="backend.multiple_heavy_models",
    title="Multiple heavy model servers running",
    fn=_evaluate,
)
