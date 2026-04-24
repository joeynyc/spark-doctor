from __future__ import annotations

import glob
import os
from typing import Any

from ..models import CollectorStatus
from ..shell import read_text, run


def collect_network() -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="network", ok=True)
    out: dict[str, Any] = {"interfaces": []}

    addr = run(["ip", "-br", "addr"], timeout=5)
    if addr.ok:
        out["ip_br_addr"] = addr.stdout
    link = run(["ip", "-br", "link"], timeout=5)
    if link.ok:
        out["ip_br_link"] = link.stdout

    interfaces: list[dict[str, Any]] = []
    for path in sorted(glob.glob("/sys/class/net/*")):
        name = os.path.basename(path)
        if name == "lo":
            continue
        entry: dict[str, Any] = {"name": name}
        speed = read_text(os.path.join(path, "speed"))
        if speed and speed.strip() and speed.strip() != "-1":
            try:
                entry["speed_mbps"] = int(speed.strip())
            except ValueError:
                pass
        operstate = read_text(os.path.join(path, "operstate"))
        if operstate:
            entry["operstate"] = operstate.strip()
        # detect mellanox/connectx by driver path
        driver_link = os.path.join(path, "device", "driver")
        try:
            driver = os.path.basename(os.readlink(driver_link))
            entry["driver"] = driver
            if "mlx" in driver.lower():
                entry["connectx_like"] = True
        except OSError:
            pass
        interfaces.append(entry)
    out["interfaces"] = interfaces

    rdma = run(["rdma", "link"], timeout=5)
    if rdma.ok:
        out["rdma_link"] = rdma.stdout
    ib = run(["ibstat"], timeout=5)
    if ib.ok:
        out["ibstat"] = ib.stdout

    return out, status
