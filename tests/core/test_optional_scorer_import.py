import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_bench_modules_import_without_bfcl_eval():
    code = (
        "import sys; sys.modules['bfcl_eval'] = None\n"
        "import agent_loops.bench.bfcl.env, agent_loops.bench.core.runner, agent_loops.bench.bfcl.single_runner\n"
        "print('ok')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-600:]
