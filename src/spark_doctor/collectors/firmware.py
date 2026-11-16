from __future__ import annotations

import os
from typing import Any

from ..models import CollectorStatus
from ..shell import run


def collect_firmware(use_sudo: bool = False) -> tuple[dict[str, Any], CollectorStatus]:
    # Firmware data feeds no rule, so probe failures are notes rather than an incomplete scan.
    status = CollectorStatus(name="firmware", ok=True, optional=True)
    out: dict[str, Any] = {}
    # Privacy matching depends on stable labels, including with gettext LANGUAGE set.
    probe_environment = {**os.environ, "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C"}

    if use_sudo:
        dmi = run(
            ["sudo", "-n", "dmidecode", "-t", "system", "-t", "bios", "-t", "baseboard"],
            timeout=10, env=probe_environment,
        )
        if dmi.ok:
            out["dmidecode"] = dmi.stdout
        else:
            status.errors.append(f"dmidecode: {dmi.error}")

    fwup = run(["fwupdmgr", "get-devices"], timeout=15, env=probe_environment)
    if fwup.ok:
        out["fwupdmgr"] = fwup.stdout
    else:
        status.errors.append(f"fwupdmgr: {fwup.error}")

    sb = run(["mokutil", "--sb-state"], timeout=5, env=probe_environment)
    if sb.ok:
        out["secure_boot"] = sb.stdout.strip()
    else:
        status.errors.append(f"mokutil --sb-state: {sb.error}: {sb.stderr.strip()[:200]}")

    status.ok = bool(out) or not status.errors
    return out, status
