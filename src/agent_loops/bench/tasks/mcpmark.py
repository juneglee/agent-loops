from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

VERIFY_TIMEOUT = 120.0


def is_mcpmark_dir(path: Path | str) -> bool:
    path = Path(path)
    return path.is_dir() and any(path.glob("*/*/*/meta.json"))


def load_mcpmark(path: Path | str) -> list[dict[str, Any]]:
    root = Path(path).resolve()
    cases: list[dict[str, Any]] = []
    for meta_path in sorted(root.glob("*/*/*/meta.json")):
        task_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        suite = task_dir.parent.parent.name
        category = meta.get("category_id", task_dir.parent.name)
        description = (task_dir / "description.md").read_text(encoding="utf-8").strip()
        cases.append(
            {
                "id": f"{suite}/{category}/{meta.get('task_id', task_dir.name)}",
                "fixture": f"fixtures/{category}",
                "cell": "single_turn_multi_step",
                "turns": [description],
                "gt_calls": [[]],
                "expect": {"verify": str((task_dir / "verify.py").relative_to(root))},
                "tags": [
                    *meta.get("tags", []),
                    f"difficulty:{meta.get('difficulty', '?')}",
                    "mcpmark",
                ],
                "source": "mcpmark",
            }
        )
    return cases


def run_verifier(
    verify_path: Path | str, workspace: Path | str
) -> tuple[bool, str | None]:
    verify_path = Path(verify_path).resolve()
    env = {
        **os.environ,
        "FILESYSTEM_TEST_DIR": str(Path(workspace).resolve()),
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        done = subprocess.run(
            [sys.executable, str(verify_path)],
            cwd=str(verify_path.parent),
            env=env,
            timeout=VERIFY_TIMEOUT,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "verify: timed out"
    if done.returncode == 0:
        return True, None
    tail = "; ".join(
        line.strip()
        for line in (done.stdout + done.stderr).splitlines()
        if line.strip()
    )[-300:]
    return False, f"verify: exit {done.returncode}: {tail}"
