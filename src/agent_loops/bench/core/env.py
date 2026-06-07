from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Env(Protocol):
    calls: list[dict[str, Any]]

    def execute(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        pass

    def tool_names(self) -> list[str]: ...

    def enable_code_execution(self) -> None:
        pass
