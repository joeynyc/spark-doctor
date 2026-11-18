from __future__ import annotations

import re
import shlex

from ..models import CollectorStatus, ProcessInfo

# Keys are normalized by _norm(): basename, lowercase, ".py" stripped, "-" -> "_".
_EXECUTABLES: dict[str, str] = {
    "vllm": "vllm",
    "ollama": "ollama",
    "ollama_llama_server": "ollama",
    "llama_server": "llama.cpp",
    "llama_cli": "llama.cpp",
    "sglang": "sglang",
    "litellm": "litellm",
    "open_webui": "open-webui",
    "comfyui": "comfyui",
}
# Matched against `-m` module roots and python script path components.
_PYTHON_PACKAGES: dict[str, str] = {
    "vllm": "vllm",
    "sglang": "sglang",
    "llama_cpp": "llama.cpp",
    "litellm": "litellm",
    "open_webui": "open-webui",
    "comfyui": "comfyui",
}
_ASGI_RUNNERS = {"uvicorn", "gunicorn", "hypercorn"}
_PYTHON_RE = re.compile(r"python(\d+(\.\d+)*)?")
_PY_OPTS_WITH_VALUE = {"-W", "-X", "--check-hash-based-pycs"}
_APP_SPEC_RE = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")
_SERVE_WORDS = {"serve", "server", "serving"}


def _norm(token: str) -> str:
    base = token.rsplit("/", 1)[-1].lower()
    if base.endswith(".py"):
        base = base[:-3]
    return base.replace("-", "_")


def _proc_title_backend(token: str) -> str | None:
    # vLLM/SGLang workers rename themselves via setproctitle, e.g. "VLLM::EngineCore".
    if "::" not in token:
        return None
    return _PYTHON_PACKAGES.get(_norm(token.split("::", 1)[0]))


def _python_target(args: list[str]) -> tuple[str, str, list[str]] | None:
    """Return (kind, target, remaining args) for what a python interpreter runs."""
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("-m") and not a.startswith("--"):
            if len(a) > 2:
                return "module", a[2:], args[i + 1:]
            return ("module", args[i + 1], args[i + 2:]) if i + 1 < len(args) else None
        if a == "-" or (a.startswith("-c") and not a.startswith("--")):
            return None
        if a in _PY_OPTS_WITH_VALUE:
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        return "script", a, args[i + 1:]
    return None


def _asgi_app_backend(args: list[str]) -> str | None:
    for a in args:
        if _APP_SPEC_RE.fullmatch(a):
            return _PYTHON_PACKAGES.get(_norm(a.split(":", 1)[0].split(".", 1)[0]))
    return None


def _is_model_server(names: list[str], args: list[str]) -> bool:
    words = {w for n in names for w in re.split(r"[_.\-]", n)}
    if not words & _SERVE_WORDS:
        return False
    if names and names[0] == "transformers":
        return True
    return any(a.startswith("--model") for a in args)


def _detect_backend(name: str, argv: list[str]) -> str | None:
    exe = argv[0] if argv else name
    for token in (name, exe):
        backend = _proc_title_backend(token)
        if backend:
            return backend
    # Shebang scripts run as [python, /venv/bin/vllm, ...] but keep the script as comm/name.
    for token in (exe, name):
        backend = _EXECUTABLES.get(_norm(token))
        if backend:
            return backend

    exe_name = _norm(exe)
    rest = argv[1:]
    if exe_name in _ASGI_RUNNERS:
        return _asgi_app_backend(rest)
    if exe_name in ("transformers", "transformers_cli"):
        return "python-serve" if rest[:1] == ["serve"] else None
    if not _PYTHON_RE.fullmatch(exe_name):
        return None

    target = _python_target(rest)
    if target is None:
        return None
    kind, value, after = target
    if kind == "module":
        names = [_norm(part) for part in value.split(".")]
    else:
        names = [_norm(part) for part in value.split("/") if part]
        names.reverse()
    if not names:
        return None
    for n in names if kind == "script" else names[:1]:
        if n in _PYTHON_PACKAGES:
            return _PYTHON_PACKAGES[n]
    if names[0] in _ASGI_RUNNERS:
        return _asgi_app_backend(after)
    serve_names = names if kind == "module" else names[:1]
    if _is_model_server(serve_names, after):
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
    denied = 0
    unavailable = object()
    try:
        for p in psutil.process_iter(
            ["pid", "ppid", "name", "cmdline", "memory_info", "memory_percent"],
            ad_value=unavailable,
        ):
            try:
                if any(value is unavailable for value in p.info.values()):
                    denied += 1
                info = {key: None if value is unavailable else value for key, value in p.info.items()}
                rss = info["memory_info"].rss if info.get("memory_info") else None
                procs.append((rss or 0, info))
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                denied += 1
    except (psutil.Error, OSError) as exc:
        status.errors.append(f"process enumeration: {type(exc).__name__}")
    if denied:
        status.errors.append(f"process enumeration: attributes unavailable for {denied} processes (access denied or exited)")

    procs.sort(key=lambda x: x[0], reverse=True)
    for rss, info in procs[:limit]:
        cmdline = info.get("cmdline") or []
        args_str = shlex.join(cmdline)
        name = info.get("name") or (cmdline[0] if cmdline else "")
        backend = _detect_backend(name or "", cmdline)
        out.append(
            ProcessInfo(
                pid=info["pid"],
                ppid=info.get("ppid"),
                command=name or "",
                args=args_str,
                rss_kb=(rss // 1024) if rss else None,
                mem_percent=info.get("memory_percent"),
                detected_backend=backend,
            )
        )
    status.ok = bool(out) or not status.errors
    return out, status
