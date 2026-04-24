from __future__ import annotations

from ..models import ScanReport
from .console import overall_status


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n" if body.strip() else ""


def _list(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items)


def render_markdown(report: ScanReport) -> str:
    lines: list[str] = []
    status = overall_status(report)
    lines.append(f"# Spark Doctor Report\n")
    lines.append(f"**Overall: {status}**  ")
    lines.append(f"Schema version: {report.schema_version}  ")
    lines.append(f"Spark Doctor version: {report.spark_doctor_version}  ")
    lines.append(f"Created at: {report.created_at.isoformat()}  ")
    lines.append(f"Anonymized: {report.anonymized}\n")

    # Summary of findings
    if report.findings:
        lines.append("## Findings\n")
        for i, f in enumerate(report.findings, start=1):
            lines.append(f"### {i}. {f.title} — `{f.severity}` ({f.rule_id})\n")
            if f.explanation:
                lines.append(f.explanation + "\n")
            if f.evidence:
                lines.append("**Evidence:**\n")
                lines.append(_list(f.evidence) + "\n")
            if f.recommended_actions:
                lines.append("**Next steps:**\n")
                lines.append(_list(f.recommended_actions) + "\n")
            if f.escalation_actions:
                lines.append("**Escalation:**\n")
                lines.append(_list(f.escalation_actions) + "\n")
    else:
        lines.append("## Findings\n\nNo issues detected.\n")

    # Hardware / OS
    os_body = []
    osr = report.os.get("os_release") if report.os else None
    if osr:
        for k in ("PRETTY_NAME", "NAME", "VERSION", "ID"):
            if osr.get(k):
                os_body.append(f"- {k}: {osr[k]}")
    if report.os.get("uname"):
        os_body.append(f"- uname: `{report.os['uname']}`")
    if report.os.get("arch"):
        os_body.append(f"- arch: `{report.os['arch']}`")
    if os_body:
        lines.append(_section("Hardware and OS", "\n".join(os_body)))

    # GPU
    gpu_body = []
    if report.gpu:
        gpu_body.append(f"- available: {report.gpu.get('available')}")
        if report.gpu.get("gpu_count") is not None:
            gpu_body.append(f"- gpu_count: {report.gpu.get('gpu_count')}")
        if report.gpu.get("name"):
            gpu_body.append(f"- name: {report.gpu['name']}")
        if report.gpu.get("driver_version"):
            gpu_body.append(f"- driver_version: {report.gpu['driver_version']}")
    if report.gpu_samples:
        gpu_body.append("\n**GPU samples:**")
        for s in report.gpu_samples:
            gpu_body.append(
                f"- util={s.gpu_utilization_percent} %, "
                f"power={s.gpu_power_draw_watts} W, "
                f"clock={s.gpu_clock_mhz} MHz, "
                f"temp={s.gpu_temperature_c} C"
            )
    if gpu_body:
        lines.append(_section("GPU telemetry", "\n".join(gpu_body)))

    # Memory
    if report.memory:
        m = report.memory
        mem_body = []
        if m.mem_total_kb:
            mem_body.append(f"- MemTotal: {m.mem_total_kb / 1024 / 1024:.1f} GB")
        if m.mem_available_kb:
            mem_body.append(f"- MemAvailable: {m.mem_available_kb / 1024 / 1024:.1f} GB")
        if m.swap_total_kb is not None and m.swap_free_kb is not None:
            used = (m.swap_total_kb - m.swap_free_kb) / 1024 / 1024
            mem_body.append(f"- Swap used: {used:.1f} GB")
        if m.psi_memory:
            mem_body.append(f"- memory PSI: `{m.psi_memory}`")
        if m.psi_io:
            mem_body.append(f"- io PSI: `{m.psi_io}`")
        if mem_body:
            lines.append(_section("Memory pressure", "\n".join(mem_body)))

    # Docker / runtime
    if report.docker:
        d = report.docker
        dk = [
            f"- docker_installed: {d.get('docker_installed')}",
            f"- daemon_reachable: {d.get('daemon_reachable')}",
            f"- nvidia_runtime_available: {d.get('nvidia_runtime_available')}",
            f"- runtimes: {d.get('runtimes')}",
        ]
        if d.get("containers"):
            dk.append(f"- running containers: {len(d['containers'])}")
        lines.append(_section("Docker / runtime state", "\n".join(dk)))

    # Processes
    model_procs = [p for p in report.processes if p.detected_backend]
    if model_procs:
        body = []
        for p in model_procs[:20]:
            rss_gb = (p.rss_kb or 0) / 1024 / 1024
            body.append(f"- pid={p.pid} backend={p.detected_backend} rss={rss_gb:.1f} GB cmd=`{p.command}`")
        lines.append(_section("Running model processes", "\n".join(body)))

    # Network
    if report.network and report.network.get("interfaces"):
        body = []
        for iface in report.network["interfaces"]:
            entry = f"- {iface.get('name')}: state={iface.get('operstate')}"
            if iface.get("speed_mbps"):
                entry += f", speed={iface['speed_mbps']} Mbps"
            if iface.get("driver"):
                entry += f", driver={iface['driver']}"
            if iface.get("connectx_like"):
                entry += " (ConnectX-like)"
            body.append(entry)
        lines.append(_section("Network", "\n".join(body)))

    # Logs (only if present, already redacted if anonymized)
    if report.logs:
        body = []
        for k, v in report.logs.items():
            if not v:
                continue
            snippet = v if len(v) < 2000 else v[-2000:]
            body.append(f"### {k}\n\n```\n{snippet}\n```")
        if body:
            lines.append(_section("Recent logs (redacted)", "\n\n".join(body)))

    # Collector statuses
    errs = [s for s in report.collector_statuses if not s.ok or s.errors]
    if errs:
        body = []
        for s in errs:
            body.append(f"- {s.name}: ok={s.ok}; errors={s.errors}")
        lines.append(_section("Collector notes", "\n".join(body)))

    if report.reproduction_notes:
        lines.append(_section("Reproduction notes", report.reproduction_notes))

    return "\n".join(lines).rstrip() + "\n"
