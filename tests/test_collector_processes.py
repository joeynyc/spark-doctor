from __future__ import annotations

from types import SimpleNamespace

import pytest

from spark_doctor.collectors import processes
from spark_doctor.collectors.processes import _detect_backend


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["code", "/home/u/notes/ollama.md"], None),
        (["vim", "vllm_notes.txt"], None),
        (["python3", "-m", "jupyter_server", "--port", "8888"], None),
        (["python3", "/home/u/scripts/plot.py", "--model", "vllm"], None),
        (["docker", "run", "vllm/vllm-openai", "--model", "x"], None),
        (["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "x"], "vllm"),
        (["/venv/bin/vllm", "serve", "Qwen/Qwen3"], "vllm"),
        (["/venv/bin/python3", "/venv/bin/vllm", "serve", "Qwen/Qwen3"], "vllm"),
        (["VLLM::EngineCore"], "vllm"),
        (["ollama", "serve"], "ollama"),
        (["/usr/local/bin/ollama", "runner", "--model", "/x"], "ollama"),
        (["llama-server", "-m", "x.gguf"], "llama.cpp"),
        (["python3", "-m", "sglang.launch_server"], "sglang"),
        (["open-webui", "serve"], "open-webui"),
        (["uvicorn", "--host", "0.0.0.0", "open_webui.main:app"], "open-webui"),
        (["litellm", "--config", "c.yaml"], "litellm"),
        (["python", "/opt/ComfyUI/main.py", "--listen"], "comfyui"),
        (["python3", "-u", "my_server.py", "--model", "meta/llama"], "python-serve"),
        (["python3", "-m", "transformers.commands.serving"], "python-serve"),
    ],
)
def test_detect_backend(argv: list[str], expected: str | None) -> None:
    name = argv[0].rsplit("/", 1)[-1]
    assert _detect_backend(name, argv) == expected


def test_proc_title_name_detected_without_cmdline() -> None:
    assert _detect_backend("VLLM::EngineCore", []) == "vllm"


def test_collector_records_ppid(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    info = {
        "pid": 42, "ppid": 7, "name": "vllm", "cmdline": ["vllm", "serve", "x"],
        "memory_info": SimpleNamespace(rss=2048), "memory_percent": 1.0,
    }
    monkeypatch.setattr(psutil, "process_iter", lambda _, **kw: iter([SimpleNamespace(info=info)]))
    data, status = processes.collect_processes()
    assert data[0].ppid == 7
    assert data[0].detected_backend == "vllm"
    assert status.ok
