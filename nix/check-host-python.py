from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Any


def scan(cli: str, root: Path, environment: dict[str, str], python: Path | None = None) -> dict[str, Any]:
    output = root / "scan.json"
    output.unlink(missing_ok=True)
    arguments = [
        cli, "scan", "--no-save", "--no-logs", "--sample-seconds", "1",
        "--json", str(output),
    ]
    if python is not None:
        arguments.extend(["--python", str(python)])
    result = subprocess.run(
        arguments, env=environment, cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, timeout=60, check=False,
    )
    assert result.returncode in (0, 1, 2, 3), result.stderr
    return json.loads(output.read_text())["cuda_env"]["python"]


def add_workload_metadata(site: Path, version: str) -> None:
    distribution = site / f"vllm-{version}.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: vllm\nVersion: {version}\n"
    )


def check_host_python(cli: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        python = root / "python3"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os\n"
            "print(json.dumps({\n"
            "    'executable': 'workload-python',\n"
            "    'python_version': '3.14',\n"
            "    'torch_import_ok': False,\n"
            "    'optional_gpu_packages': {},\n"
            "    'path_preserved': os.environ.get('PATH') == os.environ.get('SPARK_DOCTOR_TEST_PATH'),\n"
            "    'user_site_setting': os.environ.get('PYTHONNOUSERSITE'),\n"
            "}))\n"
        )
        python.chmod(0o755)
        for explicit in (False, True):
            for setting in (None, "1"):
                environment = {**os.environ, "PATH": f"{root}:{os.environ['PATH']}"}
                environment["SPARK_DOCTOR_TEST_PATH"] = environment["PATH"]
                environment.pop("PYTHONNOUSERSITE", None)
                if setting is not None:
                    environment["PYTHONNOUSERSITE"] = setting
                probe = scan(cli, root, environment, python if explicit else None)
                assert probe["executable"] == "workload-python", probe["executable"]
                assert probe["path_preserved"], "CLI changed PATH"
                assert probe["user_site_setting"] == setting, "CLI changed PYTHONNOUSERSITE"

        environment = dict(os.environ)
        environment.pop("PYTHONNOUSERSITE", None)
        workload = root / "workload"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(workload)
        python = workload / "bin/python3"
        site = Path(subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            env=environment, text=True,
        ).strip())
        add_workload_metadata(site, "0.0.0+workloadprobe")
        environment["PATH"] = f"{workload / 'bin'}:{os.environ['PATH']}"
        for explicit in (False, True):
            probe = scan(cli, root, environment, python if explicit else None)
            assert probe["executable"] == str(python), probe["executable"]
            assert probe["packages"]["vllm"] == "0.0.0+workloadprobe"

        environment = dict(os.environ)
        environment.pop("PYTHONNOUSERSITE", None)
        environment["PYTHONUSERBASE"] = str(root / "user-base")
        site = Path(subprocess.check_output(
            [sys.executable, "-c", "import site; print(site.getusersitepackages())"],
            env=environment, text=True,
        ).strip())
        add_workload_metadata(site, "0.0.0+usersiteprobe")
        # The CLI must remain isolated even while its probes use the caller's user site.
        (site / "typer.py").write_text("raise RuntimeError('CLI loaded workload user site')\n")
        result = subprocess.run(
            [cli, "self-test"], env=environment, cwd=root,
            capture_output=True, text=True, timeout=30, check=False,
        )
        assert result.returncode == 0, result.stderr
        for setting in (None, "1"):
            if setting is not None:
                environment["PYTHONNOUSERSITE"] = setting
            probe = scan(cli, root, environment, Path(sys.executable))
            expected = "0.0.0+usersiteprobe" if setting is None else None
            assert probe["packages"].get("vllm") == expected
        print("installed CLI checks passed: environment (4), virtualenv (2), user site (2), CLI isolation (1)")


if __name__ == "__main__":
    check_host_python(sys.argv[1])
