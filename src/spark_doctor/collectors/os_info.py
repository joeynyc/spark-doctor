from __future__ import annotations

from typing import Any

from ..models import CollectorStatus
from ..shell import read_text, run


def _parse_os_release(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        data[k.strip()] = v.strip().strip('"')
    return data


def collect_os() -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="os", ok=True)
    out: dict[str, Any] = {}

    os_release = read_text("/etc/os-release")
    if os_release:
        out["os_release"] = _parse_os_release(os_release)
    else:
        status.errors.append("missing:/etc/os-release")

    uname = run(["uname", "-a"])
    if uname.ok:
        out["uname"] = uname.stdout.strip()
    else:
        status.errors.append(f"uname -a: {uname.error}")

    arch = run(["uname", "-m"])
    if arch.ok:
        out["arch"] = arch.stdout.strip()

    uptime = read_text("/proc/uptime")
    if uptime:
        try:
            out["uptime_seconds"] = float(uptime.split()[0])
        except (ValueError, IndexError):
            pass

    loadavg = read_text("/proc/loadavg")
    if loadavg:
        parts = loadavg.split()
        if len(parts) >= 3:
            out["loadavg"] = [float(parts[0]), float(parts[1]), float(parts[2])]

    pkg = run(
        ["dpkg-query", "-W", "-f=${Package} ${Version}\n",
         "*nvidia*", "*cuda*", "*dgx*", "docker*"],
        timeout=8,
    )
    if pkg.ok and pkg.stdout:
        packages: dict[str, str] = {}
        for line in pkg.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[1]:
                packages[parts[0]] = parts[1]
        out["packages"] = packages
    elif pkg.error and pkg.error != "command_not_found":
        status.errors.append(f"dpkg-query: {pkg.error}")

    if not status.errors:
        status.ok = True
    else:
        status.ok = bool(out)
    return out, status
