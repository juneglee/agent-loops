from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_loops.tools import fs, shell
from agent_loops.tools.toolset import TOOLS_VERSION, Toolset


def schemas() -> list[dict[str, Any]]:
    return [
        dict(s, function=dict(s["function"]))
        for s in (*fs.FS_SCHEMAS, shell.BASH_SCHEMA)
    ]


def implementations(
    root: Path | str, bash_timeout: float = 10.0
) -> dict[str, Callable[..., str]]:
    return {**fs.make(root), **shell.make(root, timeout=bash_timeout)}


__all__ = ["TOOLS_VERSION", "Toolset", "implementations", "schemas"]
