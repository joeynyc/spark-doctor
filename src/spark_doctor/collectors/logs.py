from __future__ import annotations

from typing import Any

from ..models import CollectorStatus
from ..shell import run


def collect_logs() -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="logs", ok=True)
    out: dict[str, Any] = {}

    dmesg = run("dmesg -T 2>/dev/null | tail -200", timeout=8)
    if dmesg.ok and dmesg.stdout:
        out["dmesg_tail"] = dmesg.stdout
    elif dmesg.error and dmesg.error != "nonzero_exit":
        status.errors.append(f"dmesg: {dmesg.error}")

    jctl = run("journalctl -b --no-pager 2>/dev/null | tail -300", timeout=10)
    if jctl.ok and jctl.stdout:
        out["journal_tail"] = jctl.stdout

    return out, status
