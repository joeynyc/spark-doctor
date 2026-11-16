from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class ShellResult:
    command: str
    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def found(self) -> bool:
        return self.error != "command_not_found"


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _decode_partial(data: bytes | str | None) -> str:
    # TimeoutExpired carries raw bytes even when the call used text=True.
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def run(
    args: list[str],
    *,
    timeout: float = 10.0,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> ShellResult:
    """Run a command, returning a ShellResult. Never raises for normal failures."""
    display = " ".join(args)
    if args and which(args[0]) is None:
        return ShellResult(
            command=display,
            ok=False,
            returncode=None,
            stdout="",
            stderr="",
            error="command_not_found",
        )

    try:
        proc = subprocess.run(
            args,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=cwd,
            check=False,
        )
        return ShellResult(
            command=display,
            ok=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            error=None if proc.returncode == 0 else "nonzero_exit",
        )
    except subprocess.TimeoutExpired as e:
        return ShellResult(
            command=display,
            ok=False,
            returncode=None,
            stdout=_decode_partial(e.stdout),
            stderr=_decode_partial(e.stderr),
            error="timeout",
        )
    except FileNotFoundError:
        return ShellResult(
            command=display,
            ok=False,
            returncode=None,
            stdout="",
            stderr="",
            error="command_not_found",
        )
    except Exception as e:  # noqa: BLE001
        return ShellResult(
            command=display,
            ok=False,
            returncode=None,
            stdout="",
            stderr="",
            error=f"exception:{type(e).__name__}:{e}",
        )


def read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None
