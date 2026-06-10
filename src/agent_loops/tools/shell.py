from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_loops.tools.fs import ToolError
from agent_loops.tools.guard import DEFAULT_GUARD, Blocked, Guard

MAX_OUTPUT_CHARS = 20_000


def make(
    root: Path | str, timeout: float = 10.0, guard: Guard | None = None
) -> dict[str, Callable[..., str]]:
    root = Path(root).resolve()
    guard = guard or DEFAULT_GUARD

    def bash(command: str) -> str:
        try:
            guard.check("bash", {"command": command}, root)
        except Blocked as exc:
            raise ToolError(f"blocked before execution: {exc}") from exc
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        try:
            done = subprocess.run(
                ["/bin/bash", "-c", command],
                cwd=root,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"timeout: killed after not finishing within {timeout:g}s: {command}"
            ) from exc
        out = done.stdout
        if done.stderr:
            out = f"{out}{'' if out.endswith(chr(10)) or not out else chr(10)}[stderr] {done.stderr}"
        if len(out) > MAX_OUTPUT_CHARS:
            out = (
                out[:MAX_OUTPUT_CHARS]
                + f"\n... ({len(out) - MAX_OUTPUT_CHARS} chars omitted)"
            )
        if done.returncode != 0:
            raise ToolError(f"exit={done.returncode}: {out.strip() or command}")
        return out

    return {"bash": bash}


BASH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command inside the workspace and return its output. Use it to move, copy, delete files, create folders, and zip/unzip "
            "(basic commands only: mv, cp, rm, mkdir, zip, unzip, ls, cat, find, wc, etc.). "
            "Network access, other programs, and paths outside the workspace are blocked before execution. On failure, the exit code and error message are returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "A single command line to run",
                }
            },
            "required": ["command"],
        },
    },
}
