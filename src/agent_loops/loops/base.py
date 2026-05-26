from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def calls_of(response: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (call["name"], dict(call.get("arguments") or {}))
        for call in (response.get("tool_calls") or [])
    ]


@dataclass
class Step:
    llm_response: dict[str, Any] | None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] | None = None
    stage: str | None = None


@dataclass
class Trace:
    task: str
    loop: str
    steps: list[Step] = field(default_factory=list)
    parse_ok: bool = True
    terminated_by: str = "success"

    @property
    def n_llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.llm_response is not None)

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool_name is not None)
