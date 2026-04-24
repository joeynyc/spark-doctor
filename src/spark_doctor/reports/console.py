from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..models import ScanReport

SEVERITY_COLOR = {"info": "cyan", "warning": "yellow", "critical": "red"}
SEVERITY_LABEL = {"info": "[info]", "warning": "[warning]", "critical": "[critical]"}


def overall_status(report: ScanReport) -> str:
    sev = {f.severity for f in report.findings}
    if "critical" in sev:
        return "Critical"
    if "warning" in sev:
        return "Warning"
    if "info" in sev:
        return "Info"
    return "OK"


def render_console(report: ScanReport, console: Console | None = None) -> None:
    console = console or Console()
    status = overall_status(report)
    color = {"OK": "green", "Info": "cyan", "Warning": "yellow", "Critical": "red"}[status]
    console.print()
    console.print(Panel.fit(Text(f"Spark Doctor Scan — Overall: {status}", style=color)))

    peak = (report.gpu or {}).get("peak") if isinstance(report.gpu, dict) else None
    if peak:
        util = peak.get("gpu_utilization_percent")
        pwr = peak.get("gpu_power_draw_watts")
        clk = peak.get("gpu_clock_mhz")
        tmp = peak.get("gpu_temperature_c")
        console.print(
            f"[dim]GPU peak during scan: util={util}%, power={pwr} W, "
            f"clock={clk} MHz, temp={tmp} C  "
            f"(samples={len(report.gpu_samples)}, sampler={report.gpu.get('sampler')})[/]"
        )

    if not report.findings:
        console.print("[green]No issues detected by MVP rule set.[/]")
    else:
        console.print("\n[bold]Findings:[/]")
        for i, f in enumerate(report.findings, start=1):
            c = SEVERITY_COLOR.get(f.severity, "white")
            label = SEVERITY_LABEL.get(f.severity, f"[{f.severity}]")
            console.print(f"\n[bold]{i}. {f.title}[/]  [{c}]{label}[/]  ({f.rule_id})")
            if f.evidence:
                console.print("   Evidence:")
                for e in f.evidence:
                    console.print(f"     - {e}")
            if f.explanation:
                console.print(f"   {f.explanation}")
            if f.recommended_actions:
                console.print("   Next steps:")
                for a in f.recommended_actions:
                    console.print(f"     - {a}")
            if f.escalation_actions:
                console.print("   Escalation:")
                for a in f.escalation_actions:
                    console.print(f"     - {a}")

    errs = [s for s in report.collector_statuses if not s.ok or s.errors]
    if errs:
        console.print("\n[dim]Collector notes:[/]")
        for s in errs:
            console.print(f"  [dim]- {s.name}: ok={s.ok}; {', '.join(s.errors) if s.errors else ''}[/]")
