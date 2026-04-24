from __future__ import annotations

from typing import Any

from ..models import CollectorStatus
from ..shell import run


def collect_firmware(use_sudo: bool = False) -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="firmware", ok=True)
    out: dict[str, Any] = {}

    if use_sudo:
        dmi = run(["sudo", "-n", "dmidecode", "-t", "system", "-t", "bios", "-t", "baseboard"], timeout=10)
        if dmi.ok:
            out["dmidecode"] = dmi.stdout
        else:
            status.errors.append(f"dmidecode: {dmi.error}")

    fwup = run(["fwupdmgr", "get-devices"], timeout=15)
    if fwup.ok:
        out["fwupdmgr"] = fwup.stdout
    elif fwup.error != "command_not_found":
        status.errors.append(f"fwupdmgr: {fwup.error}")

    sb = run(["mokutil", "--sb-state"], timeout=5)
    if sb.ok:
        out["secure_boot"] = sb.stdout.strip()

    status.ok = bool(out) or not status.errors
    return out, status
