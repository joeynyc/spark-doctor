from __future__ import annotations

from .. import log_evidence
from ..models import Finding, ScanReport
from .engine import Rule

# vLLM raises "No available memory for the cache blocks" when, after loading weights
# and reserving activation + CUDA-graph memory, nothing is left for the KV cache.
# On GB10 this is usually the CUDA graphs — captured once at startup and held for the
# model's lifetime, reserved against the KV budget — NOT host memory pressure, and NOT
# something a smaller quant fixes. This is a distinct failure from memory.uma_pressure
# (which is host RAM/PSI) and must not be "fixed" by lowering gpu_memory_utilization,
# which can make it worse.


def _evaluate(report: ScanReport) -> list[Finding]:
    hits = [f"System journal: {e.line[:200]}" for e in log_evidence.lines_for(report, "kv_cache_oom")]
    if not hits:
        return log_evidence.carried_over(report, "backend.kv_cache_oom")

    return [
        Finding(
            rule_id="backend.kv_cache_oom",
            title="vLLM could not allocate a KV cache (CUDA-graph memory likely)",
            severity="critical",
            confidence="high",
            evidence=hits[:5],
            explanation=(
                "vLLM reported 'No available memory for the cache blocks'. After loading weights "
                "and reserving activation and CUDA-graph memory, nothing was left for the KV "
                "cache, so the engine cannot serve any request. On DGX Spark this is usually the "
                "CUDA graphs, not host-memory pressure: vLLM captures CUDA graphs once at startup "
                "and holds them for the model's lifetime, reserving that space against the KV "
                "budget. Because it is activation/graph memory (not weights), a smaller "
                "quantization (NVFP4/FP8) does NOT free it — and some models capture surprisingly "
                "large graphs (e.g. a multi-resolution vision model can reserve 10+ GB for a ~3B "
                "model). Note this is the opposite situation from host memory pressure: lowering "
                "gpu_memory_utilization can make it worse, not better."
            ),
            recommended_actions=[
                "Add --enforce-eager: disables CUDA-graph capture and frees that reserved memory "
                "so the KV cache fits (typical cost ~10% slower decode). Most reliable fix when "
                "packing several models onto one Spark, and it also removes graph-capture "
                "contention during concurrent model loads.",
                "Or raise --gpu-memory-utilization to give this model a larger slice of the 128 GB "
                "unified pool — only if other resident models leave room.",
                "Or reduce --max-model-len / --max-num-seqs so less KV cache is required.",
                "Read the vLLM startup log lines 'Available KV cache memory' and 'Estimated CUDA "
                "graph memory' to see which is squeezing KV toward zero.",
            ],
            source_note=(
                "vLLM 'No available memory for the cache blocks' on GB10; CUDA-graph memory is "
                "reserved against the KV budget and is not reduced by quantization."
            ),
        )
    ]


rule_backend_kv_cache_oom = Rule(
    id="backend.kv_cache_oom",
    title="vLLM KV cache allocation failure (CUDA-graph memory)",
    fn=_evaluate,
)
