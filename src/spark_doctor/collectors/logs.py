from __future__ import annotations

from typing import Any

from ..log_evidence import journal_evidence, kernel_evidence
from ..models import CollectorStatus
from ..shell import ShellResult, run

DMESG_TAIL_LINES = 200
JOURNAL_TAIL_LINES = 300


def _tail(text: str, lines: int) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _failure(name: str, result: ShellResult) -> str:
    return f"{name}: {result.error}: {result.stderr.strip()[:200]}"


def _has_entries(text: str) -> bool:
    # journalctl prints "-- No entries --" (exit 0) when the user may not read the kernel log.
    return any(line.strip() and not line.startswith("-- ") for line in text.splitlines())


def collect_logs() -> tuple[dict[str, Any], CollectorStatus]:
    """Raw tails go under dmesg_tail / journal_tail; matched lines from the whole boot go
    under "evidence" (a list of LogEvidence), present only when something matched."""
    status = CollectorStatus(name="logs", ok=True)
    out: dict[str, Any] = {}
    kernel_texts: list[str] = []
    kernel_errors: list[str] = []

    dmesg = run(["dmesg", "-T"], timeout=8)
    if dmesg.stdout:
        out["dmesg_tail"] = _tail(dmesg.stdout, DMESG_TAIL_LINES)
        kernel_texts.append(dmesg.stdout)
    if not dmesg.ok:
        kernel_errors.append(_failure("dmesg", dmesg))

    # The kernel transport only: unlike the general journal, no user process can write here,
    # and it covers the whole boot even after the dmesg ring buffer wraps.
    kernel_journal = run(["journalctl", "-k", "-b", "--no-pager"], timeout=15)
    if _has_entries(kernel_journal.stdout):
        kernel_texts.append(kernel_journal.stdout)
    if not kernel_journal.ok:
        kernel_errors.append(_failure("journalctl -k", kernel_journal))
    elif not _has_entries(kernel_journal.stdout):
        kernel_errors.append("journalctl -k: no readable kernel entries for this boot")

    kernel_read = (dmesg.ok and bool(dmesg.stdout.strip())) or (
        kernel_journal.ok and _has_entries(kernel_journal.stdout)
    )
    if not kernel_read:
        status.errors.append(
            "kernel log unreadable; OOM, Xid and thermal log checks did not run ("
            + "; ".join(kernel_errors) + ")"
        )

    journal = run(["journalctl", "-b", "--no-pager", "-n", str(JOURNAL_TAIL_LINES)], timeout=10)
    if journal.stdout:
        out["journal_tail"] = _tail(journal.stdout, JOURNAL_TAIL_LINES)
    if not journal.ok:
        status.errors.append(_failure("journalctl", journal))

    evidence = kernel_evidence(kernel_texts) + journal_evidence(journal.stdout)
    if evidence:
        out["evidence"] = evidence

    status.ok = bool(out) or not status.errors
    return out, status
