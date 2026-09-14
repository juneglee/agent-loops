from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOTS = {"fs_1": "testbed", "fs_2": "system", "fs_3": "workspace"}
BUILD_TIMEOUT = 60.0
_ABSOLUTE = re.compile(r"(^|\s)/(?!bin/|dev/null|usr/bin/)[A-Za-z]")


def is_intercode_dir(path: Path | str) -> bool:
    path = Path(path)
    return (path / "tasks.json").is_file() and (path / "setup").is_dir()


def _portable(script: str, root_name: str) -> str:
    lines = []
    for line in script.splitlines():
        if line.startswith("rm -rf"):
            continue
        line = line.replace(f"/{root_name}", root_name)
        if _ABSOLUTE.search(line):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def build_fixtures(path: Path | str, out: Path | str | None = None) -> list[str]:
    path = Path(path).resolve()
    out = Path(out) if out is not None else path / "fixtures"
    built: list[str] = []
    for script in sorted((path / "setup").glob("setup_nl2b_fs_*.sh")):
        name = f"fs_{script.stem.rsplit('_', 1)[1]}"
        root_name = ROOTS.get(name)
        if root_name is None:
            continue
        target = out / name
        if target.is_dir():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        text = _portable(script.read_text(encoding="utf-8"), root_name)
        subprocess.run(
            ["/bin/bash", "-c", text],
            cwd=target,
            timeout=BUILD_TIMEOUT,
            capture_output=True,
            check=False,
        )
        built.append(name)
    return built


def load_intercode(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    build_fixtures(path)
    return json.loads((path / "tasks.json").read_text(encoding="utf-8"))
