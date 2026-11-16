from __future__ import annotations

import sys

from spark_doctor.shell import run


def test_timeout_preserves_partial_output() -> None:
    script = (
        "import sys, time\n"
        "print('partial out é', flush=True)\n"
        "print('partial err', file=sys.stderr, flush=True)\n"
        "time.sleep(10)\n"
    )
    result = run([sys.executable, "-c", script], timeout=1.0)
    assert result.error == "timeout"
    assert not result.ok
    assert "partial out é" in result.stdout
    assert "partial err" in result.stderr
