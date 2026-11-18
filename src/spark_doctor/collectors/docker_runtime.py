from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..models import CollectorStatus
from ..shell import run, which

CDI_DIRS = ("/etc/cdi", "/var/run/cdi")
CDI_SPEC_SUFFIXES = (".yaml", ".yml", ".json")
CDI_SPEC_READ_LIMIT = 64 * 1024
_YAML_NVIDIA_GPU_KIND = re.compile(r"^\s*kind\s*:\s*['\"]?nvidia\.com/gpu['\"]?\s*(?:#.*)?$", re.MULTILINE)
_JSON_NVIDIA_GPU_KIND = re.compile(r'"kind"\s*:\s*"nvidia\.com/gpu"')


def _local_endpoint(status: CollectorStatus) -> tuple[str | None, dict[str, str]]:
    env = os.environ.copy()
    context = env.get("DOCKER_CONTEXT")
    endpoint = env.get("DOCKER_HOST") if not context else None
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        env.pop(key, None)
    if not context and not endpoint:
        config_dir = Path(env.get("DOCKER_CONFIG") or Path.home() / ".docker")
        try:
            config = json.loads((config_dir / "config.json").read_text())
            if not isinstance(config, dict):
                raise ValueError("expected an object")
            context = config.get("currentContext")
            if context is not None and not isinstance(context, str):
                raise ValueError("invalid currentContext")
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            status.errors.append(f"docker config: cannot resolve local endpoint ({type(exc).__name__})")
            return None, env
    if context and context != "default":
        # Context inspection reads local metadata; it does not contact the daemon.
        metadata = run(
            ["docker", "context", "inspect", context, "--format", "{{json .Endpoints.docker.Host}}"],
            timeout=5,
            env=env,
        )
        if not metadata.ok:
            status.errors.append(f"docker context inspect: {metadata.error}: {metadata.stderr.strip()[:200]}")
            return None, env
        try:
            endpoint = json.loads(metadata.stdout)
            if not isinstance(endpoint, str) or not endpoint:
                raise ValueError("missing endpoint")
        except ValueError:
            status.errors.append("docker context inspect: invalid endpoint metadata")
            return None, env
    endpoint = endpoint or "unix:///var/run/docker.sock"
    try:
        parsed = urlsplit(endpoint)
        if (
            not endpoint.startswith("unix:///") or any(ord(c) < 32 for c in endpoint)
            or parsed.scheme != "unix" or parsed.netloc or not parsed.path.startswith("/")
            or parsed.path == "/" or parsed.query or parsed.fragment
        ):
            raise ValueError("not a local Unix socket")
    except ValueError:
        status.errors.append("docker endpoint skipped: only a local unix:/// socket is allowed; select a local Docker context")
        return None, env
    return endpoint, env


def _is_nvidia_cdi_spec(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(CDI_SPEC_READ_LIMIT).decode("utf-8", errors="replace")
    except OSError:
        return False
    return bool(_YAML_NVIDIA_GPU_KIND.search(head) or _JSON_NVIDIA_GPU_KIND.search(head))


def _cdi_specs_present() -> bool:
    """True only when a CDI spec file declares NVIDIA GPU devices; stray files do not count."""
    for d in CDI_DIRS:
        try:
            with os.scandir(d) as entries:
                # is_file() excludes FIFOs and device nodes, whose reads could block.
                specs = sorted(
                    e.path for e in entries
                    if e.name.lower().endswith(CDI_SPEC_SUFFIXES) and e.is_file()
                )
        except OSError:
            continue
        if any(_is_nvidia_cdi_spec(p) for p in specs):
            return True
    return False


def _cdi_list_device_count(status: CollectorStatus) -> int:
    """Count NVIDIA GPU device names, independent of toolkit log formatting."""
    cdi = run(["nvidia-ctk", "cdi", "list"], timeout=5)
    if not cdi.ok:
        status.errors.append(f"nvidia-ctk cdi list: {cdi.error or 'probe failed'}")
        return 0
    return len({
        line.strip()
        for line in cdi.stdout.splitlines()
        if line.strip().startswith("nvidia.com/gpu=")
        and line.strip().removeprefix("nvidia.com/gpu=")
        and len(line.split()) == 1
    })


def _finalize_gpu_readiness(out: dict[str, Any]) -> None:
    """Require a reachable Docker daemon and a detected GPU-injection path."""
    out["gpu_docker_ready"] = bool(
        out.get("docker_installed")
        and out.get("daemon_reachable")
        and (
            out.get("nvidia_runtime_available")
            or out.get("cdi_specs_present")
            or out.get("nvidia_hook_installed")
        )
    )


def collect_docker() -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="docker", ok=True)
    out: dict[str, Any] = {
        "docker_installed": which("docker") is not None,
        "nvidia_container_runtime_installed": which("nvidia-container-runtime") is not None,
        "nvidia_ctk_installed": which("nvidia-ctk") is not None,
        "nvidia_hook_installed": which("nvidia-container-runtime-hook") is not None,
        "cdi_specs_present": _cdi_specs_present(),
        "cdi_device_count": 0,
        "daemon_reachable": False,
        "socket_accessible": False,
        "nvidia_runtime_available": False,
        "gpu_container_running": False,
        "gpu_docker_ready": False,
        "runtimes": [],
        "containers": [],
    }

    # `nvidia-ctk cdi list` reports generated device specs even when the /etc/cdi
    # dir scan misses them; either signal means a usable CDI spec exists.
    if out["nvidia_ctk_installed"]:
        out["cdi_device_count"] = _cdi_list_device_count(status)
        if out["cdi_device_count"] > 0:
            out["cdi_specs_present"] = True

    # Missing Docker is itself the diagnosis (runtime rule reads docker_installed), not a collector failure.
    if not out["docker_installed"]:
        _finalize_gpu_readiness(out)
        return out, status

    endpoint, env = _local_endpoint(status)
    if endpoint is None:
        out["daemon_check_skipped"] = True
        status.ok = False
        return out, status
    out["endpoint"] = endpoint
    command = ["docker", "--host", endpoint]

    ver = run([*command, "version", "--format", "{{json .}}"], timeout=5, env=env)
    if ver.ok:
        out["daemon_reachable"] = True
        out["socket_accessible"] = True
        try:
            parsed_version = json.loads(ver.stdout)
            if not isinstance(parsed_version, dict):
                raise ValueError("expected an object")
            out["version"] = parsed_version
        except ValueError:
            out["version_raw"] = ver.stdout.strip()
            status.errors.append("docker version: unparseable output")
    else:
        status.errors.append(f"docker version: {ver.stderr.strip()[:200] or ver.error}")

    info = run([*command, "info", "--format", "{{json .}}"], timeout=5, env=env)
    if info.ok:
        try:
            d = json.loads(info.stdout)
            if not isinstance(d, dict) or not isinstance(d.get("Runtimes"), dict):
                raise ValueError("missing runtimes")
            out["daemon_reachable"] = True
            out["socket_accessible"] = True
            runtimes = list(d["Runtimes"])
            out["runtimes"] = runtimes
            out["nvidia_runtime_available"] = any("nvidia" in r.lower() for r in runtimes)
            out["default_runtime"] = d.get("DefaultRuntime")
            out["server_version"] = d.get("ServerVersion")
        except ValueError:
            status.errors.append("docker info: unparseable output or missing runtimes")
    else:
        status.errors.append(f"docker info: {info.error}: {info.stderr.strip()[:200]}")

    ps = run([*command, "ps", "--format", "{{json .}}"], timeout=5, env=env)
    if ps.ok and ps.stdout.strip():
        containers: list[dict[str, Any]] = []
        for line in ps.stdout.strip().splitlines():
            try:
                container = json.loads(line)
                if not isinstance(container, dict) or not isinstance(container.get("ID"), str):
                    raise ValueError("missing container ID")
                containers.append(container)
            except ValueError:
                status.errors.append("docker ps: unparseable container row")
                continue
        out["containers"] = containers

        for c in containers:
            cid = c.get("ID")
            if not cid:
                continue
            inspect = run(
                [*command, "inspect", "--format", "{{json .HostConfig.DeviceRequests}}", cid],
                timeout=5,
                env=env,
            )
            if inspect.ok:
                try:
                    requests = json.loads(inspect.stdout)
                    if requests is not None and (
                        not isinstance(requests, list) or not all(isinstance(r, dict) for r in requests)
                    ):
                        raise ValueError("invalid device requests")
                except ValueError:
                    status.errors.append(f"docker inspect {cid}: unparseable device requests")
                    requests = None
                if requests:
                    out["gpu_container_running"] = True
            else:
                status.errors.append(f"docker inspect {cid}: {inspect.error}: {inspect.stderr.strip()[:200]}")
    elif not ps.ok:
        status.errors.append(f"docker ps: {ps.error}: {ps.stderr.strip()[:200]}")

    _finalize_gpu_readiness(out)

    if status.errors and not out["daemon_reachable"]:
        status.ok = False
    return out, status
