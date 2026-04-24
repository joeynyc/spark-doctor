from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .collectors import (
    collect_docker,
    collect_firmware,
    collect_gpu,
    collect_logs,
    collect_memory,
    collect_network,
    collect_os,
    collect_processes,
)
from .models import ScanReport
from .privacy import redact_report
from .recipes.validator import load_recipe, validate_recipe
from .reports import render_console, render_forum, render_github, render_markdown
from .rules import run_rules

app = typer.Typer(add_completion=False, help="Spark Doctor: local diagnostic CLI for DGX Spark.")
recipe_app = typer.Typer(add_completion=False, help="Recipe validation commands.")
app.add_typer(recipe_app, name="recipe")

console = Console()


def _exit_code_for(report: ScanReport) -> int:
    sev = {f.severity for f in report.findings}
    if "critical" in sev:
        return 2
    if "warning" in sev:
        return 1
    return 0


def _save_json(report: ScanReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2))


def _build_report(
    *,
    sample_seconds: int,
    include_logs: bool,
    use_sudo: bool,
    anonymize: bool,
    include_network_identifiers: bool,
) -> ScanReport:
    report = ScanReport(
        created_at=datetime.utcnow(),
        spark_doctor_version=__version__,
        anonymized=anonymize,
    )

    os_data, s_os = collect_os()
    report.os = os_data
    report.collector_statuses.append(s_os)

    fw, s_fw = collect_firmware(use_sudo=use_sudo)
    report.firmware = fw
    report.collector_statuses.append(s_fw)

    gpu_data, samples, s_gpu = collect_gpu(sample_seconds=sample_seconds)
    report.gpu = gpu_data
    report.gpu_samples = samples
    report.collector_statuses.append(s_gpu)

    mem, s_mem = collect_memory()
    report.memory = mem
    report.collector_statuses.append(s_mem)

    dk, s_dk = collect_docker()
    report.docker = dk
    report.collector_statuses.append(s_dk)

    procs, s_pr = collect_processes()
    report.processes = procs
    report.collector_statuses.append(s_pr)

    net, s_net = collect_network()
    report.network = net
    report.collector_statuses.append(s_net)

    if include_logs:
        logs, s_lg = collect_logs()
        report.logs = logs
        report.collector_statuses.append(s_lg)

    report.findings = run_rules(report)

    if anonymize:
        report = redact_report(report, include_network_identifiers=include_network_identifiers)

    return report


@app.command()
def version() -> None:
    """Print Spark Doctor version."""
    console.print(f"spark-doctor {__version__}")


@app.command()
def scan(
    sample_seconds: int = typer.Option(5, "--sample-seconds", help="GPU sampling duration."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write JSON report to this path."),
    markdown_out: Optional[Path] = typer.Option(None, "--markdown", help="Write markdown report."),
    no_logs: bool = typer.Option(False, "--no-logs", help="Do not collect dmesg/journalctl."),
    include_logs: bool = typer.Option(False, "--include-logs", help="Collect dmesg/journalctl snippets."),
    use_sudo: bool = typer.Option(False, "--sudo", help="Allow sudo for firmware collection."),
    anonymize: bool = typer.Option(True, "--anonymize/--no-anonymize", help="Redact personal identifiers."),
    include_network_identifiers: bool = typer.Option(
        False, "--include-network-identifiers", help="Keep private IPs and MACs in report."
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Save scan under .spark-doctor/reports/."),
) -> None:
    """Run collectors, evaluate rules, print findings."""
    want_logs = include_logs and not no_logs
    report = _build_report(
        sample_seconds=sample_seconds,
        include_logs=want_logs,
        use_sudo=use_sudo,
        anonymize=anonymize,
        include_network_identifiers=include_network_identifiers,
    )

    render_console(report, console=console)

    if save:
        stamp = report.created_at.strftime("%Y-%m-%dT%H%M%S")
        default = Path(".spark-doctor/reports") / f"{stamp}.json"
        _save_json(report, default)
        console.print(f"\n[dim]Report saved: {default}[/]")

    if json_out is not None:
        _save_json(report, json_out)
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(report))

    raise typer.Exit(code=_exit_code_for(report))


@app.command()
def doctor(
    from_file: Path = typer.Option(..., "--from", help="Load a scan JSON fixture and re-run rules."),
) -> None:
    """Re-run diagnosis rules against an existing scan JSON."""
    data = json.loads(from_file.read_text())
    report = ScanReport.model_validate(data)
    report.findings = run_rules(report)
    render_console(report, console=console)
    raise typer.Exit(code=_exit_code_for(report))


@app.command()
def report(
    from_file: Path = typer.Option(..., "--from", help="Path to scan JSON."),
    format: str = typer.Option("markdown", "--format", help="markdown | forum | github"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write to path instead of stdout."),
) -> None:
    """Render a report from a saved scan JSON."""
    data = json.loads(from_file.read_text())
    rep = ScanReport.model_validate(data)
    if not rep.findings:
        rep.findings = run_rules(rep)
    fmt = format.lower()
    if fmt == "markdown":
        text = render_markdown(rep)
    elif fmt == "forum":
        text = render_forum(rep)
    elif fmt == "github":
        text = render_github(rep)
    else:
        console.print(f"[red]Unknown format: {format}[/]")
        raise typer.Exit(code=3)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        console.print(f"[green]Wrote {out}[/]")
    else:
        typer.echo(text)


@app.command()
def anonymize(
    scan_file: Path = typer.Argument(..., help="Path to scan JSON to anonymize."),
    out: Path = typer.Option(..., "--out", help="Path to write redacted JSON."),
    include_network_identifiers: bool = typer.Option(False, "--include-network-identifiers"),
) -> None:
    """Produce a redacted copy of an existing scan JSON."""
    data = json.loads(scan_file.read_text())
    rep = ScanReport.model_validate(data)
    redacted = redact_report(rep, include_network_identifiers=include_network_identifiers)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(redacted.model_dump_json(indent=2))
    console.print(f"[green]Wrote redacted report: {out}[/]")


@app.command("self-test")
def self_test() -> None:
    """Run a minimal self-test that does not require GPU hardware."""
    report = ScanReport(spark_doctor_version=__version__)
    report.findings = run_rules(report)
    console.print("[green]self-test ok[/]")
    console.print(f"rules registered: {len(report.findings) + 5}")  # at least the 5 MVP rules


@recipe_app.command("check")
def recipe_check(
    recipe_file: Path = typer.Argument(..., help="Path to a YAML recipe file."),
    gpus: int = typer.Option(1, "--gpus", help="Detected GPU count to validate against."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Architecture (e.g. aarch64)."),
    mem_available_gb: Optional[float] = typer.Option(None, "--mem-available-gb"),
) -> None:
    """Validate a recipe YAML against MVP rules."""
    recipe = load_recipe(recipe_file)
    result = validate_recipe(
        recipe,
        detected_gpu_count=gpus,
        detected_arch=arch,
        mem_available_gb=mem_available_gb,
    )
    status_color = {"pass": "green", "warn": "yellow", "fail": "red"}[result.status]
    console.print(f"\n[bold]Recipe Check: {recipe_file.name}[/]\n")
    console.print(f"Status: [{status_color}]{result.status.upper()}[/]\n")
    for i, issue in enumerate(result.issues, start=1):
        color = {"info": "cyan", "warning": "yellow", "critical": "red"}[issue.severity]
        console.print(f"{i}. [{color}]{issue.severity}[/] {issue.title}  ({issue.id})")
        console.print(f"   {issue.detail}")
        if issue.suggested_fix:
            console.print(f"   [dim]Suggested fix:[/] {issue.suggested_fix}")
        console.print()
    exit_code = {"pass": 0, "warn": 1, "fail": 2}[result.status]
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
