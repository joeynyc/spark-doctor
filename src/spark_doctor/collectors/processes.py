from __future__ import annotations

from ..models import CollectorStatus, ProcessInfo

BACKEND_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("vllm", ("vllm",)),
    ("ollama", ("ollama",)),
    ("llama.cpp", ("llama-server", "llama.cpp", "llama_cpp")),
    ("sglang", ("sglang",)),
    ("litellm", ("litellm",)),
    ("open-webui", ("open-webui", "open_webui")),
    ("comfyui", ("comfyui", "ComfyUI")),
]


def _detect_backend(cmd: str, args: str) -> str | None:
    hay = (cmd + " " + args).lower()
    for name, keys in BACKEND_KEYWORDS:
        for k in keys:
            if k.lower() in hay:
                return name
    if "python" in hay and any(x in hay for x in ("serve", "--model", "--port", "transformers", "vllm.entrypoints")):
        return "python-serve"
    return None


def collect_processes(limit: int = 50) -> tuple[list[ProcessInfo], CollectorStatus]:
    status = CollectorStatus(name="processes", ok=True)
    out: list[ProcessInfo] = []
    try:
        import psutil
    except ImportError:
        status.ok = False
        status.errors.append("psutil not installed")
        return out, status

    procs: list[tuple[int, dict]] = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent", "memory_percent"]):
        try:
            info = p.info
            rss = info["memory_info"].rss if info.get("memory_info") else None
            procs.append((rss or 0, info))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    procs.sort(key=lambda x: x[0], reverse=True)
    for rss, info in procs[:limit]:
        cmdline = info.get("cmdline") or []
        args_str = " ".join(cmdline)
        name = info.get("name") or (cmdline[0] if cmdline else "")
        backend = _detect_backend(name, args_str)
        out.append(
            ProcessInfo(
                pid=info["pid"],
                command=name or "",
                args=args_str,
                rss_kb=(rss // 1024) if rss else None,
                cpu_percent=info.get("cpu_percent"),
                mem_percent=info.get("memory_percent"),
                detected_backend=backend,
            )
        )
    return out, status
