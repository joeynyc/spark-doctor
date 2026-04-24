from pathlib import Path

from typer.testing import CliRunner

from spark_doctor.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "spark-doctor" in result.stdout


def test_doctor_from_fixture_exits_critical():
    result = runner.invoke(app, ["doctor", "--from", str(FIXTURES / "power_limited_14w.json")])
    assert result.exit_code == 2
    assert "Possible GPU low-power state" in result.stdout


def test_report_forum():
    result = runner.invoke(
        app,
        ["report", "--from", str(FIXTURES / "power_limited_14w.json"), "--format", "forum"],
    )
    assert result.exit_code == 0
    assert "Title suggestion" in result.stdout


def test_recipe_check_fails_for_bad_tp():
    result = runner.invoke(app, ["recipe", "check", str(FIXTURES / "recipe_tp_too_high.yaml")])
    assert result.exit_code == 2
    assert "FAIL" in result.stdout
