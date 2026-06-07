from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_loops.bench.core.env import Env


@runtime_checkable
class Track(Protocol):
    name: str

    def revision(self) -> str:
        pass

    def cases(self, cell: str) -> list[dict[str, Any]]:
        pass

    def turns_of(self, case: dict[str, Any]) -> list[str]: ...

    def tools_for(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        pass

    def make_env(self, case: dict[str, Any], budgets: Any) -> Env: ...

    def score(
        self, case: dict[str, Any], env: Env, turn_traces: list[Any]
    ) -> tuple[bool, str | None]:
        pass
