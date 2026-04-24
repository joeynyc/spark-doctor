import json
from pathlib import Path

from spark_doctor.models import ScanReport
from spark_doctor.reports import render_forum, render_github, render_markdown
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> ScanReport:
    data = json.loads((FIXTURES / name).read_text())
    r = ScanReport.model_validate(data)
    r.findings = run_rules(r)
    return r


def test_markdown_contains_finding_title():
    r = _load("power_limited_14w.json")
    md = render_markdown(r)
    assert "Spark Doctor Report" in md
    assert "Possible GPU low-power state" in md
    assert "Next steps" in md


def test_forum_contains_title_and_tldr():
    r = _load("power_limited_14w.json")
    txt = render_forum(r)
    assert "Title suggestion" in txt
    assert "TL;DR" in txt
    assert "CRITICAL" in txt


def test_github_has_env_section():
    r = _load("power_limited_14w.json")
    txt = render_github(r)
    assert "Environment" in txt
    assert "Steps to reproduce" in txt


def test_shell_runner_timeout_and_missing():
    from spark_doctor.shell import run

    # missing command
    r = run(["this_command_does_not_exist_12345"])
    assert not r.ok
    assert r.error == "command_not_found"

    # successful command
    r2 = run(["true"])
    assert r2.ok
    assert r2.returncode == 0

    # timeout
    r3 = run(["sleep", "5"], timeout=0.2)
    assert not r3.ok
    assert r3.error == "timeout"
