from spark_doctor.models import ProcessInfo, ScanReport
from spark_doctor.rules.backend import rule_backend_multiple_heavy_models


def _proc(pid: int, backend: str | None, rss_gb: float, ppid: int | None = None) -> ProcessInfo:
    return ProcessInfo(
        pid=pid, ppid=ppid, command="python3", rss_kb=int(rss_gb * 1024 * 1024), detected_backend=backend
    )


def test_two_heavy_backends_warn() -> None:
    report = ScanReport(processes=[_proc(1, "vllm", 40), _proc(2, "ollama", 20)])
    finding = rule_backend_multiple_heavy_models.fn(report)[0]
    assert finding.severity == "warning"
    assert any("Total heavy backend RSS: 60.0 GB" in line for line in finding.evidence)


def test_legacy_scan_without_ppid_does_not_count_same_backend_processes_twice() -> None:
    report = ScanReport(processes=[_proc(1, "vllm", 40), _proc(2, "vllm", 30)])
    assert rule_backend_multiple_heavy_models.fn(report) == []


def test_small_or_non_heavy_processes_are_ignored() -> None:
    report = ScanReport(
        processes=[_proc(1, "vllm", 40), _proc(2, "ollama", 0.5), _proc(3, "open-webui", 4)]
    )
    assert rule_backend_multiple_heavy_models.fn(report) == []


def test_two_vllm_servers_with_engine_children_warn() -> None:
    report = ScanReport(processes=[
        _proc(100, "vllm", 4, ppid=1), _proc(101, "vllm", 36, ppid=100),
        _proc(200, "vllm", 4, ppid=1), _proc(201, "vllm", 36, ppid=200),
    ])
    finding = rule_backend_multiple_heavy_models.fn(report)[0]
    assert any(line.startswith("vllm: 2 instance(s), 4 process(es)") for line in finding.evidence)


def test_one_vllm_server_with_engine_children_is_one_instance() -> None:
    report = ScanReport(processes=[
        _proc(100, "vllm", 4, ppid=1), _proc(101, "vllm", 36, ppid=100), _proc(102, "vllm", 36, ppid=100),
    ])
    assert rule_backend_multiple_heavy_models.fn(report) == []


def test_light_vllm_parent_still_groups_heavy_workers() -> None:
    report = ScanReport(processes=[
        _proc(100, "vllm", 0.5, ppid=1),
        _proc(150, None, 0.1, ppid=100),
        _proc(101, "vllm", 36, ppid=150), _proc(102, "vllm", 36, ppid=150),
    ])
    assert rule_backend_multiple_heavy_models.fn(report) == []


def test_vllm_and_ollama_with_ppid_warn() -> None:
    report = ScanReport(processes=[_proc(100, "vllm", 40, ppid=1), _proc(200, "ollama", 20, ppid=1)])
    finding = rule_backend_multiple_heavy_models.fn(report)[0]
    assert "vllm: 1 instance(s), 1 process(es), 40.0 GB RSS" in finding.evidence
    assert "ollama: 1 instance(s), 1 process(es), 20.0 GB RSS" in finding.evidence
