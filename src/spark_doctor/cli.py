from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, NoReturn, Optional, TypeVar

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console

from . import __version__
from .collectors import (
    collect_cuda_env,
    collect_docker,
    collect_firmware,
    collect_gpu,
    collect_logs,
    collect_memory,
    collect_network,
    collect_os,
    collect_processes,
)
from .models import CollectorStatus, MetricSample, ScanReport
from .privacy import redact_report, redact_text
from .recipes.validator import load_recipe, validate_recipe
from .reports import render_console, render_forum, render_github, render_markdown
from .rules import run_rules

app = typer.Typer(help="Spark Doctor: local diagnostic CLI for DGX Spark.")
recipe_app = typer.Typer(add_completion=False, help="Recipe validation commands.")
app.add_typer(recipe_app, name="recipe")

console = Console()
Collected = TypeVar("Collected")

REPORT_FORMATS = ("markdown", "forum", "github")


def _exit_code_for(report: ScanReport) -> int:
    if report.incomplete:
        return 3
    sev = {f.severity for f in report.findings}
    if "critical" in sev:
        return 2
    if "warning" in sev:
        return 1
    return 0


def _input_error(kind: str, error: Exception, code: int) -> NoReturn:
    if isinstance(error, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'document'}: {e['msg']}"
            for e in error.errors(include_input=False, include_url=False)
        )
    elif isinstance(error, OSError):
        details = "Cannot read the input file; check its path and permissions."
    else:
        details = "Expected a valid JSON scan." if kind == "scan" else "Expected a valid YAML recipe mapping."
    typer.echo(redact_text(f"Invalid {kind}: {details}"), err=True)
    raise typer.Exit(code=code)


def _load_report(path: Path) -> ScanReport:
    try:
        return ScanReport.model_validate_json(path.read_text())
    except (OSError, ValueError) as error:
        _input_error("scan", error, 3)


def _collect(
    report: ScanReport,
    name: str,
    collector: Callable[[], tuple[Collected, CollectorStatus]],
    fallback: Collected,
) -> Collected:
    try:
        data, status = collector()
    except Exception as error:  # noqa: BLE001
        data = fallback
        status = CollectorStatus(name=name, ok=False, errors=[f"{type(error).__name__}: {error}"])
    report.collector_statuses.append(status)
    return data


def _save_json(report: ScanReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2))


def _build_report(
    *,
    sample_seconds: int,
    read_logs: bool,
    include_logs: bool,
    use_sudo: bool,
    anonymize: bool,
    include_network_identifiers: bool,
    python_executable: str = "python3",
) -> ScanReport:
    report = ScanReport(
        created_at=datetime.now(timezone.utc),
        spark_doctor_version=__version__,
        anonymized=anonymize,
    )

    def gpu_snapshot() -> tuple[tuple[dict[str, Any], list[MetricSample]], CollectorStatus]:
        data, samples, status = collect_gpu(sample_seconds=sample_seconds)
        return (data, samples), status

    report.os = _collect(report, "os", collect_os, {})
    report.firmware = _collect(report, "firmware", lambda: collect_firmware(use_sudo=use_sudo), {})
    report.gpu, report.gpu_samples = _collect(report, "gpu", gpu_snapshot, ({}, []))
    report.memory = _collect(report, "memory", collect_memory, None)
    report.cuda_env = _collect(report, "cuda_env", lambda: collect_cuda_env(python_executable=python_executable), {})
    report.docker = _collect(report, "docker", collect_docker, {})
    report.processes = _collect(report, "processes", collect_processes, [])
    report.network = _collect(report, "network", collect_network, {})

    if read_logs:
        report.logs = _collect(report, "logs", collect_logs, {})
        if not include_logs:
            report.collector_statuses[-1].optional = True

    report.findings = run_rules(report)
    # Logs are read so log-based rules can fire, but raw log text leaves the machine
    # only on explicit opt-in; findings carry just the redacted matching lines.
    if not include_logs:
        report.logs = {}

    if anonymize:
        report = redact_report(report, include_network_identifiers=include_network_identifiers)

    return report


@app.command()
def version() -> None:
    """Print Spark Doctor version."""
    console.print(f"spark-doctor {__version__}")


@app.command()
def scan(
    sample_seconds: int = typer.Option(5, "--sample-seconds", min=1, help="GPU sampling duration."),
    python_executable: str = typer.Option("python3", "--python", help="Python interpreter for the workload's CUDA/package checks (default: python3 on PATH)."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write JSON report to this path."),
    markdown_out: Optional[Path] = typer.Option(None, "--markdown", help="Write markdown report."),
    no_logs: bool = typer.Option(False, "--no-logs", help="Do not read dmesg/journalctl at all."),
    include_logs: bool = typer.Option(False, "--include-logs", help="Include dmesg/journalctl snippets in saved reports."),
    use_sudo: bool = typer.Option(False, "--sudo", help="Allow sudo for firmware collection."),
    include_sensitive_data: bool = typer.Option(False, "--include-sensitive-data", help="Keep raw identifiers and credentials in outputs."),
    include_network_identifiers: bool = typer.Option(
        False, "--include-network-identifiers", help="Keep private IPv4, IPv6, and MAC addresses in report; hardware identifiers stay redacted."
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Save scan under .spark-doctor/reports/."),
) -> None:
    """Run collectors, evaluate rules, print findings."""
    report = _build_report(
        sample_seconds=sample_seconds,
        read_logs=not no_logs,
        include_logs=include_logs and not no_logs,
        use_sudo=use_sudo,
        anonymize=not include_sensitive_data,
        include_network_identifiers=include_network_identifiers,
        python_executable=python_executable,
    )

    render_console(report, console=console)

    if save:
        stamp = report.created_at.strftime("%Y-%m-%dT%H%M%S")
        default = Path(".spark-doctor/reports") / f"{stamp}.json"
        _save_json(report, default)
        console.print(redact_text(f"\nReport saved: {default}"), markup=False)

    if json_out is not None:
        _save_json(report, json_out)
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(report))

    raise typer.Exit(code=_exit_code_for(report))


@app.command()
def doctor(
    from_file: Path = typer.Option(..., "--from", help="Load a scan JSON fixture and re-run rules."),
    include_sensitive_data: bool = typer.Option(False, "--include-sensitive-data", help="Keep raw identifiers and credentials in output."),
    include_network_identifiers: bool = typer.Option(False, "--include-network-identifiers"),
) -> None:
    """Re-run diagnosis rules against an existing scan JSON."""
    report = _load_report(from_file)
    report.findings = run_rules(report)
    if not include_sensitive_data:
        report = redact_report(report, include_network_identifiers=include_network_identifiers)
    render_console(report, console=console)
    raise typer.Exit(code=_exit_code_for(report))


@app.command()
def report(
    from_file: Path = typer.Option(..., "--from", help="Path to scan JSON."),
    format: str = typer.Option(
        "markdown",
        "--format",
        help=" | ".join(REPORT_FORMATS),
        autocompletion=lambda incomplete: [f for f in REPORT_FORMATS if f.startswith(incomplete)],
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Write to path instead of stdout."),
    include_sensitive_data: bool = typer.Option(False, "--include-sensitive-data", help="Keep raw identifiers and credentials in output."),
    include_network_identifiers: bool = typer.Option(False, "--include-network-identifiers"),
) -> None:
    """Render a report from a saved scan JSON."""
    rep = _load_report(from_file)
    if not rep.findings:
        rep.findings = run_rules(rep)
    if not include_sensitive_data:
        rep = redact_report(rep, include_network_identifiers=include_network_identifiers)
    fmt = format.lower()
    if fmt == "markdown":
        text = render_markdown(rep)
    elif fmt == "forum":
        text = render_forum(rep)
    elif fmt == "github":
        text = render_github(rep)
    else:
        console.print(redact_text(f"Unknown format: {format}"), markup=False)
        raise typer.Exit(code=3)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        console.print(redact_text(f"Wrote {out}"), markup=False)
    else:
        typer.echo(text)


@app.command()
def anonymize(
    scan_file: Path = typer.Argument(..., help="Path to scan JSON to anonymize."),
    out: Path = typer.Option(..., "--out", help="Path to write redacted JSON."),
    include_network_identifiers: bool = typer.Option(False, "--include-network-identifiers"),
) -> None:
    """Produce a redacted copy of an existing scan JSON."""
    rep = _load_report(scan_file)
    redacted = redact_report(rep, include_network_identifiers=include_network_identifiers)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(redacted.model_dump_json(indent=2))
    console.print(redact_text(f"Wrote redacted report: {out}"), markup=False)


@app.command("self-test")
def self_test() -> None:
    """Run a minimal self-test that does not require GPU hardware."""
    from .rules.engine import ALL_RULES

    report = ScanReport(spark_doctor_version=__version__)
    report.findings = run_rules(report)
    if report.incomplete:
        render_console(redact_report(report), console=console)
        raise typer.Exit(code=3)
    console.print("[green]self-test ok[/]")
    console.print(f"rules registered: {len(ALL_RULES)}")


@recipe_app.command("check")
def recipe_check(
    recipe_file: Path = typer.Argument(..., help="Path to a YAML recipe file."),
    gpus: int = typer.Option(1, "--gpus", min=1, help="Detected GPU count to validate against."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Architecture (e.g. aarch64)."),
    mem_available_gb: Optional[float] = typer.Option(None, "--mem-available-gb", min=0),
) -> None:
    """Validate a recipe YAML for DGX Spark compatibility."""
    if mem_available_gb is not None and not isfinite(mem_available_gb):
        raise typer.BadParameter("must be a finite number", param_hint="--mem-available-gb")
    try:
        recipe = load_recipe(recipe_file)
        result = validate_recipe(
            recipe,
            detected_gpu_count=gpus,
            detected_arch=arch,
            mem_available_gb=mem_available_gb,
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        _input_error("recipe", error, 2)
    status_color = {"pass": "green", "warn": "yellow", "fail": "red"}[result.status]
    console.print(redact_text(f"\nRecipe Check: {recipe_file.name}\n"), markup=False)
    console.print(f"Status: [{status_color}]{result.status.upper()}[/]\n")
    for i, issue in enumerate(result.issues, start=1):
        color = {"info": "cyan", "warning": "yellow", "critical": "red"}[issue.severity]
        console.print(redact_text(f"{i}. {issue.severity} {issue.title}  ({issue.id})"), style=color, markup=False)
        console.print(redact_text(f"   {issue.detail}"), markup=False)
        if issue.suggested_fix:
            console.print(redact_text(f"   Suggested fix: {issue.suggested_fix}"), markup=False)
        console.print()
    exit_code = {"pass": 0, "warn": 1, "fail": 2}[result.status]
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
