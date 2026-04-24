from __future__ import annotations

from typing import Callable

from ..models import Finding, ScanReport

RuleFn = Callable[[ScanReport], list[Finding]]


class Rule:
    def __init__(self, id: str, title: str, fn: RuleFn) -> None:
        self.id = id
        self.title = title
        self.fn = fn

    def run(self, report: ScanReport) -> list[Finding]:
        return self.fn(report) or []


def run_rules(report: ScanReport, rules: list[Rule] | None = None) -> list[Finding]:
    rules = rules if rules is not None else ALL_RULES
    findings: list[Finding] = []
    for r in rules:
        try:
            findings.extend(r.run(report))
        except Exception as e:  # noqa: BLE001
            findings.append(
                Finding(
                    rule_id=f"rule.error.{r.id}",
                    title=f"Rule {r.id} errored",
                    severity="info",
                    confidence="low",
                    evidence=[f"{type(e).__name__}: {e}"],
                    explanation="Rule raised an exception during evaluation.",
                )
            )
    return findings


def _lazy_rules() -> list[Rule]:
    from .power import rule_power_low_draw_under_load
    from .thermal import rule_thermal_shutdown_risk
    from .memory import rule_memory_uma_pressure
    from .runtime import rule_runtime_docker_unhealthy
    from .backend import rule_backend_multiple_heavy_models

    return [
        rule_power_low_draw_under_load,
        rule_thermal_shutdown_risk,
        rule_memory_uma_pressure,
        rule_runtime_docker_unhealthy,
        rule_backend_multiple_heavy_models,
    ]


ALL_RULES: list[Rule] = _lazy_rules()
