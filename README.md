# Spark Doctor

Local diagnostic CLI for NVIDIA DGX Spark. Collects system, GPU, memory, Docker, runtime, network, and recipe data, applies DGX Spark-specific rules, and prints a short answer: **what is wrong, why, and what to try next.**

Read-only. No dashboard. No auto-fixes. No telemetry.

## Why

DGX Spark is a new platform. When something goes wrong, the signal is scattered across `nvidia-smi`, `/proc/pressure/*`, `dmesg`, `docker info`, backend logs, forum threads, and Field Diagnostics. Owners hit the same issues repeatedly — GPU stuck in a 14 W low-power state, unified memory pressure stalling inference, thermal shutdowns, Docker runtime not registered, recipes with `tensor_parallel_size` set for a multi-GPU box.

Spark Doctor collects those signals in one command, applies DGX Spark-specific rules, and tells you the likely cause and the next step — in plain English, with the evidence attached.

## Install

```bash
git clone <repo-url> && cd spark-doctor
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python 3.11+.

## Commands

```bash
spark-doctor scan                              # full scan + diagnosis
spark-doctor scan --json scan.json --markdown report.md
spark-doctor doctor --from scan.json           # re-run rules on saved scan
spark-doctor report --from scan.json --format {markdown,forum,github}
spark-doctor recipe check recipe.yaml
spark-doctor anonymize scan.json --out redacted.json
spark-doctor self-test
spark-doctor version
```

Exit codes: `0` clean · `1` warning · `2` critical · `3` collector failure.

## What it detects

| ID | Detects |
|---|---|
| `power.low_draw_under_load` | High GPU utilization with suspiciously low power draw (e.g. 14 W cap). |
| `thermal.shutdown_risk` | GPU temp ≥ 85/90 C or thermal events in logs. |
| `memory.uma_pressure` | Low `MemAvailable`, high memory PSI, or heavy swap use. |
| `runtime.docker_unhealthy` | Docker/NVIDIA container runtime missing or misconfigured. |
| `backend.multiple_heavy_models` | Two or more heavy model backends running concurrently. |

Recipe validator checks tensor-parallel vs GPU count, container image registry, arm64 compatibility, memory budget, and aggressive `gpu_memory_utilization` / context lengths.

## Privacy

Reports are anonymized by default:

- Hostname, username, and home paths replaced.
- Private IPv4 and MAC addresses redacted unless `--include-network-identifiers`.
- HF, NGC, OpenAI, bearer, JWT, and SSH-key patterns redacted.
- Logs (`dmesg`, `journalctl`) only included with `--include-logs`.

## Safety

No package installs, driver updates, process kills, reboots, clock locking, or power changes. All fixes are instructions.

## Development

```bash
pip install -e '.[dev]'
pytest
```

New rules go in `src/spark_doctor/rules/`, register in `rules/engine.py`, add a fixture in `tests/fixtures/`, add a test.

## License

Apache-2.0.
