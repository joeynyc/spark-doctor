from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Literal

from .models import Finding, LogEvidence, ScanReport

Signal = Literal["oom_kill", "xid", "thermal", "kv_cache_oom"]
Source = Literal["kernel", "journal"]

OOM_KILL = re.compile(
    r"(?P<cgroup>Memory cgroup )?[Oo]ut of memory: Killed process (?P<pid>\d+) \((?P<name>[^)]*)\)"
)
XID = re.compile(r"NVRM: Xid \((?P<device>[^)]*)\): (?P<code>\d+)(?P<rest>[^\n]*)")
THERMAL_KEYWORDS = ("thermal shutdown", "over temperature", "overheat", "thermal event", "critical temperature")
# vLLM raises this when nothing is left for the KV cache after weights, activations and
# CUDA graphs. It is an application message, so it can only come from the general journal.
KV_CACHE_OOM = re.compile(r"no available memory for (?:the )?cache blocks", re.IGNORECASE)
_VLLM = re.compile(r"vllm", re.IGNORECASE)

_SYSLOG_KERNEL = re.compile(r"^[A-Z][a-z]{2} [ \d]?\d \d{2}:\d{2}:\d{2} \S+ kernel: ")
_DMESG_PREFIX = re.compile(r"^\[[^\]]*\]\s*")

# A flapping fault can log thousands of identical-shape lines; the rules only quote a few.
MAX_LINES_PER_SIGNAL = 100
_MAX_LINE_CHARS = 1024

CARRIED_OVER_NOTE = (
    "Carried over from the original scan: its raw logs were not saved, so this was not re-checked."
)
_LEGACY_LOG_KEYS = ("dmesg_tail", "journal_tail")


def _kernel_signals(line: str) -> list[Signal]:
    signals: list[Signal] = []
    if OOM_KILL.search(line):
        signals.append("oom_kill")
    if XID.search(line):
        signals.append("xid")
    low = line.lower()
    if any(kw in low for kw in THERMAL_KEYWORDS):
        signals.append("thermal")
    return signals


def _journal_signals(line: str) -> list[Signal]:
    return ["kv_cache_oom"] if KV_CACHE_OOM.search(line) and _VLLM.search(line) else []


def _signals(line: str, source: Source) -> list[Signal]:
    return _kernel_signals(line) if source == "kernel" else _journal_signals(line)


def _kernel_message(line: str) -> str:
    """Strip the dmesg or journal prefix so one event read from both sources dedups."""
    syslog = _SYSLOG_KERNEL.match(line)
    if syslog:
        return line[syslog.end():]
    return _DMESG_PREFIX.sub("", line, count=1)


def _extract(lines: Iterable[tuple[Source, str]]) -> list[LogEvidence]:
    out: list[LogEvidence] = []
    seen: set[tuple[Signal, str]] = set()
    counts: dict[Signal, int] = {}
    for source, raw in lines:
        line = raw.strip()[:_MAX_LINE_CHARS]
        if not line:
            continue
        key_text = _kernel_message(line) if source == "kernel" else line
        for signal in _signals(line, source):
            key = (signal, key_text)
            if key in seen or counts.get(signal, 0) >= MAX_LINES_PER_SIGNAL:
                continue
            seen.add(key)
            counts[signal] = counts.get(signal, 0) + 1
            out.append(LogEvidence(source=source, signal=signal, line=line))
    return out


def kernel_evidence(texts: Iterable[str]) -> list[LogEvidence]:
    """Evidence from kernel-provenance text only: `dmesg` or `journalctl -k` output."""
    return _extract(("kernel", line) for text in texts for line in text.splitlines())


def journal_evidence(text: str) -> list[LogEvidence]:
    """Evidence from the general journal. Any process can write there, so only the
    application-level kv-cache signature is accepted from it."""
    return _extract(("journal", line) for line in text.splitlines())


def _legacy_evidence(logs: dict[str, str]) -> list[LogEvidence]:
    lines: list[tuple[Source, str]] = []
    dmesg = logs.get("dmesg_tail")
    if isinstance(dmesg, str):
        lines.extend(("kernel", line) for line in dmesg.splitlines())
    journal = logs.get("journal_tail")
    if isinstance(journal, str):
        for line in journal.splitlines():
            lines.append(("kernel" if _SYSLOG_KERNEL.match(line.strip()) else "journal", line))
    return _extract(lines)


def _is_legacy_without_logs(report: ScanReport) -> bool:
    return report.log_evidence is None and not any(report.logs.get(k) for k in _LEGACY_LOG_KEYS)


def report_evidence(report: ScanReport) -> list[LogEvidence]:
    if report.log_evidence is not None:
        return report.log_evidence
    return _legacy_evidence(report.logs)


def lines_for(report: ScanReport, signal: Signal) -> list[LogEvidence]:
    """Evidence for one signal, re-checked against its source so a hand-edited or
    mislabelled entry (e.g. an Xid tagged as journal) cannot produce a finding."""
    return [
        e for e in report_evidence(report)
        if e.signal == signal and signal in _signals(e.line, e.source)
    ]


def carried_over(
    report: ScanReport, rule_id: str, keep: Callable[[Finding], bool] | None = None,
) -> list[Finding]:
    """Stored findings for a log rule from a scan saved before evidence was persisted
    and without raw logs: re-running the rule would otherwise silently drop them."""
    if not _is_legacy_without_logs(report):
        return []
    kept: list[Finding] = []
    for finding in report.findings:
        if finding.rule_id != rule_id or (keep is not None and not keep(finding)):
            continue
        copy = finding.model_copy(deep=True)
        if CARRIED_OVER_NOTE not in copy.evidence:
            copy.evidence.append(CARRIED_OVER_NOTE)
        kept.append(copy)
    return kept
