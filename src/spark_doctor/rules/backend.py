from __future__ import annotations

from ..models import Finding, ScanReport
from .engine import Rule

HEAVY_BACKENDS = {"vllm", "sglang", "llama.cpp", "ollama", "comfyui"}


def _evaluate(report: ScanReport) -> list[Finding]:
    heavy = [
        p for p in report.processes
        if p.detected_backend in HEAVY_BACKENDS
        and (p.rss_kb or 0) >= 1_000_000  # >= ~1 GB RSS
    ]
    # collapse by backend name
    seen: dict[str, list] = {}
    for p in heavy:
        seen.setdefault(p.detected_backend or "", []).append(p)
    if len(seen) < 2:
        return []

    evidence = []
    total_rss_gb = 0.0
    for name, procs in seen.items():
        rss_gb = sum((p.rss_kb or 0) for p in procs) / (1024 * 1024)
        total_rss_gb += rss_gb
        evidence.append(f"{name}: {len(procs)} process(es), {rss_gb:.1f} GB RSS")
    evidence.append(f"Total heavy backend RSS: {total_rss_gb:.1f} GB")

    return [
        Finding(
            rule_id="backend.multiple_heavy_models",
            title="Multiple heavy model backends running",
            severity="warning",
            confidence="medium",
            evidence=evidence,
            explanation=(
                "Two or more heavy model backends are running simultaneously. "
                "They will compete for unified memory and GPU time."
            ),
            recommended_actions=[
                "Stop idle backends.",
                "Use a model swapper or router instead of keeping all models resident.",
                "Re-run `spark-doctor scan` with only one workload active.",
            ],
        )
    ]


rule_backend_multiple_heavy_models = Rule(
    id="backend.multiple_heavy_models",
    title="Multiple heavy model backends running",
    fn=_evaluate,
)
