from __future__ import annotations

from ..models import ScanReport
from .console import overall_status
from .markdown import render_markdown


def render_github(report: ScanReport) -> str:
    status = overall_status(report)
    header: list[str] = []
    header.append(f"### Environment\n")
    header.append(f"- Overall status: **{status}**")
    header.append(f"- Spark Doctor: `{report.spark_doctor_version}`")
    osr = report.os.get("os_release") if report.os else None
    if osr and osr.get("PRETTY_NAME"):
        header.append(f"- OS: `{osr['PRETTY_NAME']}`")
    if report.os.get("arch"):
        header.append(f"- arch: `{report.os['arch']}`")
    if report.gpu.get("name"):
        header.append(f"- GPU: `{report.gpu['name']}`")
    if report.gpu.get("driver_version"):
        header.append(f"- driver: `{report.gpu['driver_version']}`")

    steps = ["\n### Steps to reproduce\n"]
    if report.reproduction_notes:
        steps.append(report.reproduction_notes)
    else:
        steps.append("_(fill in)_")

    expected = ["\n### Expected vs actual\n", "_(fill in)_"]

    body = render_markdown(report).replace("# Spark Doctor Report\n", "", 1)

    return "\n".join(header) + "\n" + "\n".join(steps) + "\n" + "\n".join(expected) + "\n\n" + body
