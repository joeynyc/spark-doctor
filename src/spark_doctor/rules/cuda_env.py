from __future__ import annotations

import re
from typing import Any

from ..models import Finding, ScanReport
from .engine import Rule

# GB10 is compute capability 12.1 (SM_121a) and ships with CUDA 13.
# Most pip ML packages still publish cu12.x wheels, which fail on this box —
# the single most reported DGX Spark software issue.

_CU13_TORCH_FIX = [
    "Install a CUDA 13 torch wheel: pip install torch --index-url https://download.pytorch.org/whl/cu130",
    "Or use an NVIDIA NGC container (nvcr.io/nvidia/pytorch) where torch is prebuilt for GB10.",
]


def _cuda_env(report: ScanReport) -> dict[str, Any]:
    return report.cuda_env if isinstance(report.cuda_env, dict) else {}


def _python_info(report: ScanReport) -> dict[str, Any]:
    py = _cuda_env(report).get("python")
    return py if isinstance(py, dict) else {}


def _is_gb10(report: ScanReport) -> bool:
    name = str(report.gpu.get("name", ""))
    if "GB10" in name:
        return True
    gpus = report.gpu.get("gpus")
    if isinstance(gpus, list):
        return any("GB10" in str(g.get("name", "")) for g in gpus if isinstance(g, dict))
    return False


def _major(version: str | None) -> int | None:
    if not version:
        return None
    try:
        return int(str(version).split(".")[0])
    except ValueError:
        return None


def _needs_cuda13(report: ScanReport) -> bool:
    driver_major = _major(_cuda_env(report).get("driver_cuda_version"))
    if driver_major is not None:
        return driver_major >= 13
    return _is_gb10(report)


def _eval_torch_cpu_only(report: ScanReport) -> list[Finding]:
    py = _python_info(report)
    version = py.get("torch_version")
    if py.get("torch_import_ok") is not True or not isinstance(version, str):
        return []
    # A null CUDA version alone can also describe a ROCm or other non-CUDA build.
    if "cpu" not in version.partition("+")[2].lower().split("."):
        return []
    if "torch_cuda_version" not in py or py["torch_cuda_version"] is not None:
        return []
    if py.get("torch_cuda_available") is not False:
        return []

    executable = py.get("executable") or _cuda_env(report).get("python_executable")
    evidence = [
        f"torch {version} imported successfully and is tagged as a CPU-only build",
        "torch.version.cuda is None; torch.cuda.is_available() is False",
    ]
    if executable:
        evidence.insert(0, f"Inspected Python: {executable}")

    return [
        Finding(
            rule_id="cuda.torch_cpu_only",
            title="Selected Python has a CPU-only PyTorch build",
            severity="warning",
            confidence="high",
            evidence=evidence,
            explanation=(
                "PyTorch in the selected Python environment was built without CUDA support, "
                "so it cannot use the NVIDIA GPU. This finding applies only to the inspected "
                "interpreter; it does not establish whether a separate GPU container or "
                "another Python environment is working."
            ),
            recommended_actions=[
                "Confirm which Python environment runs your workload. Re-run spark-doctor "
                "scan --python /path/to/workload/python to inspect that interpreter.",
                "If this environment is intended for GPU work, choose a PyTorch build that "
                "supports CUDA and your hardware, then re-scan the same interpreter.",
                "If the inspected environment is intentionally CPU-only and GPU work runs "
                "elsewhere, this warning can be ignored; check that workload separately.",
            ],
        )
    ]


def _eval_torch_cu12(report: ScanReport) -> list[Finding]:
    py = _python_info(report)
    if not py.get("torch_import_ok"):
        return []
    torch_cuda = py.get("torch_cuda_version")
    if _major(torch_cuda) != 12 or not _needs_cuda13(report):
        return []

    evidence = [f"torch {py.get('torch_version', '?')} built for CUDA {torch_cuda}"]
    driver_cuda = _cuda_env(report).get("driver_cuda_version")
    if driver_cuda:
        evidence.append(f"driver reports CUDA {driver_cuda}")
    if _is_gb10(report):
        evidence.append("GPU is GB10 (SM_121a, requires CUDA 13)")
    if py.get("torch_cuda_available") is False:
        evidence.append("torch.cuda.is_available() is False")

    return [
        Finding(
            rule_id="cuda.torch_cu12_wheel",
            title="PyTorch built for CUDA 12 on a CUDA 13 system",
            severity="critical",
            confidence="high",
            evidence=evidence,
            explanation=(
                "The installed PyTorch wheel targets CUDA 12.x, but DGX Spark (GB10) requires "
                "CUDA 13. Plain `pip install torch` pulls a cu12 wheel that cannot use this GPU — "
                "typically torch.cuda.is_available() returns False or kernels fail to load."
            ),
            recommended_actions=_CU13_TORCH_FIX,
            source_note="GB10 sm_121 requires CUDA 13; PyPI default wheels are cu12.x.",
        )
    ]


def _eval_libcudart_import_failure(report: ScanReport) -> list[Finding]:
    py = _python_info(report)
    if py.get("torch_import_ok") is not False:
        return []
    err = str(py.get("torch_import_error", ""))
    if "libcudart" not in err:
        return []

    evidence = [f"torch import failed: {err[:200]}"]
    sonames = _cuda_env(report).get("libcudart_sonames")
    if isinstance(sonames, list):
        evidence.append(
            f"libcudart on system: {', '.join(sonames) if sonames else 'none found via ldconfig'}"
        )

    return [
        Finding(
            rule_id="cuda.libcudart_missing",
            title="Python package linked against a CUDA runtime that is not installed",
            severity="critical",
            confidence="high",
            evidence=evidence,
            explanation=(
                "A package (torch or one of its dependencies) was built against a CUDA runtime "
                "library this system does not have — the classic symptom of a cu12.x wheel on "
                "DGX Spark's CUDA 13 stack (`libcudart.so.12: cannot open shared object file`)."
            ),
            recommended_actions=[
                "Reinstall the failing package from a CUDA 13 wheel index (e.g. torch from https://download.pytorch.org/whl/cu130).",
                "Check other GPU packages in the same venv (vllm, flash-attn, triton) — they must all target cu13.",
                "If a cu13 wheel does not exist for a package, use the NGC container that bundles it instead.",
            ],
            source_note="Pervasive libcudart.so.12 import failures reported by DGX Spark owners.",
        )
    ]


# Cubins run on later minor revisions of the same major, so sm_120 kernels (the official
# cu130 wheels) run on GB10's sm_121. Arch-specific sm_120a cubins do not.
_GB10_COMPATIBLE_ARCH = re.compile(r"sm_120|sm_121a?|compute_121a?")


def _eval_sm121_arch_missing(report: ScanReport) -> list[Finding]:
    py = _python_info(report)
    if not py.get("torch_import_ok") or not _is_gb10(report):
        return []
    arch_list = py.get("torch_arch_list")
    if not isinstance(arch_list, list) or not arch_list:
        return []
    if any(_GB10_COMPATIBLE_ARCH.fullmatch(str(a)) for a in arch_list):
        return []

    return [
        Finding(
            rule_id="cuda.sm121_not_in_arch_list",
            title="PyTorch build does not include SM_121 (GB10) kernels",
            severity="warning",
            confidence="medium",
            evidence=[
                f"torch {py.get('torch_version', '?')} arch list: {', '.join(str(a) for a in arch_list)}",
                "GPU is GB10 (compute capability 12.1 / SM_121a)",
            ],
            explanation=(
                "This PyTorch build ships no kernels for the GB10's SM_121a architecture. It may "
                "still run via PTX JIT compilation (slow first run, and some fused kernels are "
                "skipped), or fail with 'no kernel image is available' errors."
            ),
            recommended_actions=[
                "Prefer a cu130 wheel or NGC PyTorch container that includes sm_121 support.",
                "If building from source, set TORCH_CUDA_ARCH_LIST to include 12.1.",
            ],
            source_note="'SM_121a architecture not recognized' is a documented DGX Spark build issue.",
        )
    ]


def _eval_nvcc_driver_mismatch(report: ScanReport) -> list[Finding]:
    env = _cuda_env(report)
    nvcc_major = _major(env.get("nvcc_release"))
    if nvcc_major is None or nvcc_major >= 13 or not _needs_cuda13(report):
        return []

    evidence = [f"nvcc release {env.get('nvcc_release')}"]
    driver_cuda = env.get("driver_cuda_version")
    if driver_cuda:
        evidence.append(f"driver reports CUDA {driver_cuda}")

    return [
        Finding(
            rule_id="cuda.nvcc_toolkit_mismatch",
            title="CUDA toolkit (nvcc) is older than the driver's CUDA version",
            severity="warning",
            confidence="medium",
            evidence=evidence,
            explanation=(
                "nvcc on PATH is a CUDA 12.x toolkit while this system runs CUDA 13. Source builds "
                "(vllm, flash-attn, custom kernels) will compile against the wrong toolkit and "
                "either fail on SM_121a or produce binaries that cannot load."
            ),
            recommended_actions=[
                "Install the CUDA 13 toolkit, or check PATH order if both toolkits are installed (which nvcc).",
                "For source builds, prefer the NGC CUDA 13 development containers.",
            ],
            source_note="Source builds on DGX Spark require the CUDA 13 toolkit for SM_121a.",
        )
    ]


def _evaluate(report: ScanReport) -> list[Finding]:
    if not _cuda_env(report):
        return []
    findings: list[Finding] = []
    findings.extend(_eval_torch_cu12(report))
    findings.extend(_eval_libcudart_import_failure(report))
    findings.extend(_eval_sm121_arch_missing(report))
    findings.extend(_eval_nvcc_driver_mismatch(report))
    return findings


rule_cuda_env_mismatch = Rule(
    id="cuda.env_mismatch",
    title="CUDA 13 / SM_121 environment mismatches",
    fn=_evaluate,
)

rule_torch_cpu_only = Rule(
    id="cuda.torch_cpu_only",
    title="CPU-only PyTorch in the selected Python environment",
    fn=_eval_torch_cpu_only,
)
