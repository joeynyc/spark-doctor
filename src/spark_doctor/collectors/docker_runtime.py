from __future__ import annotations

import json
from typing import Any

from ..models import CollectorStatus
from ..shell import run, which


def collect_docker() -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="docker", ok=True)
    out: dict[str, Any] = {
        "docker_installed": which("docker") is not None,
        "nvidia_container_runtime_installed": which("nvidia-container-runtime") is not None,
        "nvidia_ctk_installed": which("nvidia-ctk") is not None,
        "daemon_reachable": False,
        "socket_accessible": False,
        "nvidia_runtime_available": False,
        "runtimes": [],
        "containers": [],
    }

    if not out["docker_installed"]:
        status.errors.append("docker not installed")
        return out, status

    ver = run(["docker", "version", "--format", "{{json .}}"], timeout=5)
    if ver.ok:
        out["daemon_reachable"] = True
        out["socket_accessible"] = True
        try:
            out["version"] = json.loads(ver.stdout)
        except json.JSONDecodeError:
            out["version_raw"] = ver.stdout.strip()
    else:
        status.errors.append(f"docker version: {ver.stderr.strip()[:200] or ver.error}")

    info = run(["docker", "info", "--format", "{{json .}}"], timeout=5)
    if info.ok:
        try:
            d = json.loads(info.stdout)
            runtimes = list((d.get("Runtimes") or {}).keys())
            out["runtimes"] = runtimes
            out["nvidia_runtime_available"] = any("nvidia" in r.lower() for r in runtimes)
            out["default_runtime"] = d.get("DefaultRuntime")
            out["server_version"] = d.get("ServerVersion")
        except json.JSONDecodeError:
            pass

    ps = run(["docker", "ps", "--format", "{{json .}}"], timeout=5)
    if ps.ok and ps.stdout.strip():
        containers: list[dict[str, Any]] = []
        for line in ps.stdout.strip().splitlines():
            try:
                containers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        out["containers"] = containers

    if status.errors and not out["daemon_reachable"]:
        status.ok = False
    return out, status
