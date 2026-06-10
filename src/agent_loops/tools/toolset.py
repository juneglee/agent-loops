from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_loops.tools import fs, shell
from agent_loops.tools.guard import DEFAULT_GUARD, Blocked, Guard

TOOLS_VERSION = "t1"


class Toolset:
    version = TOOLS_VERSION

    def __init__(
        self, root: Path | str, guard: Guard | None = None, bash_timeout: float = 10.0
    ) -> None:
        self.root = Path(root).resolve()
        self.guard = guard or DEFAULT_GUARD
        self._impl = {
            **fs.make(self.root),
            **shell.make(self.root, timeout=bash_timeout, guard=self.guard),
        }
        self._schemas = [
            dict(s, function=dict(s["function"]))
            for s in (*fs.FS_SCHEMAS, shell.BASH_SCHEMA)
        ]

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
