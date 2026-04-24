# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

Local, read-only diagnostic CLI for NVIDIA DGX Spark. Collect → diagnose → report. No dashboards, no auto-fixes, no telemetry.

## Hard rules

- **No system mutation.** No installs, kills, reboots, clock/power changes. Fixes are instructions only.
- **Collectors are best-effort.** Never raise on missing command, permission denied, or timeout — record in `CollectorStatus` and continue.
- **Redact by default.** Anything user-facing goes through `privacy/redact.py`. Only bypass with explicit `--include-*` flags.
- **No network calls** from CLI, collectors, or rules.
- **Python 3.11+.** Type hints required. `from __future__ import annotations` in new modules.

## Layout

```
src/spark_doctor/
  cli.py              # Typer entrypoint
  models.py           # Pydantic: ScanReport, Finding, MetricSample, etc.
  shell.py            # run() — safe subprocess wrapper
  collectors/         # os_info, firmware, gpu, memory, docker_runtime, processes, network, logs
  rules/              # engine + power, thermal, memory, runtime, backend
  recipes/            # schema, validator, known_registry.yaml
  reports/            # console, markdown, forum, github
  privacy/redact.py
tests/
  fixtures/           # *.json scan fixtures + *.yaml recipe fixtures
```

## Adding a rule

1. New function `(report: ScanReport) -> list[Finding]` in `rules/<area>.py`.
2. Wrap in `Rule(id, title, fn)`; register in `rules/engine.py::_lazy_rules`.
3. Add a fixture in `tests/fixtures/`.
4. Add a test in `tests/test_rules_*.py`.

Every `Finding` must have: `rule_id`, `title`, `severity`, `evidence`, `explanation`, `recommended_actions`. Plain-English first, technical second.

## Adding a recipe check

Extend `recipes/validator.py` with a new `RecipeIssue` branch. Add fixture + test.

## Severity → exit codes

`info` → 0 · `warning` → 1 · `critical` → 2 · collector failure → 3.

## Dev loop

```bash
source .venv/bin/activate
pytest -q
spark-doctor doctor --from tests/fixtures/power_limited_14w.json
```

## Don't

- Don't add a web UI, daemon, or background service.
- Don't add an auto-fix mode without an explicit per-action confirmation design.
- Don't pull in heavy deps (no pandas, no numpy, no torch). Current stack: typer, rich, pydantic, pyyaml, psutil.
- Don't rely on NVML totals for unified memory pressure — use `MemAvailable` and PSI.
- Don't write comments explaining *what* code does; only *why* when non-obvious.
