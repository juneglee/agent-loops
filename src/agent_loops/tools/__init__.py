from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_loops.tools.toolset import TOOL_VERSIONS, TOOLS_VERSION, Toolset, schemas_for


def schemas(version: str = TOOLS_VERSION) -> list[dict[str, Any]]:
    return schemas_for(version)


def implementations(
    root: Path | str, bash_timeout: float = 10.0, version: str = TOOLS_VERSION
) -> dict[str, Callable[..., str]]:
    return Toolset(root, bash_timeout=bash_timeout, version=version)._impl


__all__ = ["TOOLS_VERSION", "TOOL_VERSIONS", "Toolset", "implementations", "schemas"]
