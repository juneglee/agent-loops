from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent_loops.tools import fs, fs_tool, shell
from agent_loops.tools.guard import DEFAULT_GUARD, PATH_GUARD, Blocked, Guard

TOOLS_VERSION = "t1"
TOOL_VERSIONS = ("t1", fs_tool.TOOLS_VERSION)


def _build(
    version: str,
    root: Path,
    bash_timeout: float,
    guard: Guard,
    seen_paths: Iterable[str] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if version == "t1":
        impl = {
            **fs.make(root),
            **shell.make(root, timeout=bash_timeout, guard=guard),
        }
        return impl, [*fs.FS_SCHEMAS, shell.BASH_SCHEMA]
    if version == fs_tool.TOOLS_VERSION:
        return fs_tool.make(
            root, bash_timeout=bash_timeout, guard=guard, seen_paths=seen_paths
        ), fs_tool.SCHEMAS
    raise ValueError(f"unknown tools version: {version}")


def schemas_for(version: str = TOOLS_VERSION) -> list[dict[str, Any]]:
    if version == "t1":
        source = [*fs.FS_SCHEMAS, shell.BASH_SCHEMA]
    elif version == fs_tool.TOOLS_VERSION:
        source = fs_tool.SCHEMAS
    else:
        raise ValueError(f"unknown tools version: {version}")
    return [dict(s, function=dict(s["function"])) for s in source]


class Toolset:
    def __init__(
        self,
        root: Path | str,
        guard: Guard | None = None,
        bash_timeout: float = 10.0,
        version: str = TOOLS_VERSION,
        seen_paths: Iterable[str] = (),
    ) -> None:
        self.root = Path(root).resolve()
        self.guard = guard or (
            PATH_GUARD if version == fs_tool.TOOLS_VERSION else DEFAULT_GUARD
        )
        self.version = version
        self._impl, source = _build(
            version, self.root, bash_timeout, self.guard, seen_paths
        )
        self._schemas = [dict(s, function=dict(s["function"])) for s in source]

    def schemas(self) -> list[dict[str, Any]]:
        return [dict(s, function=dict(s["function"])) for s in self._schemas]

    def names(self) -> list[str]:
        return [s["function"]["name"] for s in self._schemas]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        arguments = dict(arguments or {})
        fn = self._impl.get(name)
        if fn is None:
            raise fs.ToolError(f"unknown tool: {name}")
        try:
            self.guard.check(name, arguments, self.root)
        except Blocked as exc:
            raise fs.ToolError(f"blocked before execution: {exc}") from exc
        try:
            return fn(**arguments)
        except TypeError as exc:
            raise fs.ToolError(f"argument error: {exc}") from exc
