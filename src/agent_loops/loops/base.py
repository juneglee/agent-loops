from __future__ import annotations

import re
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

    def stop(self, reason: str) -> Trace:
        self.terminated_by = reason
        self.parse_ok = reason not in ("parse_fail", "no_action")
        return self

    @property
    def n_llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.llm_response is not None)

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool_name is not None)


_DECORATION = "*_`#>\"'“”「」[]() \t"

_COMPLETE = re.compile(
    r"^(?:(?:[^:：\n]{0,12}[,，]\s*)?(?:final|response)|최종|완료)\s*(?:답|answer)?\s*[:：]"
)
_GIVE_UP = re.compile(
    r"^(?:i\s+)?(?:give up|giving up|task failed|포기)\s*(?:[:：—–\-]|[.!]?\s*$)"
)


def _declaration_lines(text: Any) -> list[str]:
    lines = [
        line.strip().strip(_DECORATION).lower()
        for line in str(text or "").splitlines()
        if line.strip()
    ]
    return lines[:1] + (lines[-1:] if len(lines) > 1 else [])


def response_is_complete(response: dict[str, Any]) -> bool:
    if response.get("completed") is True or response.get("done") is True:
        return True
    if response.get("final") is not None:
        return True
    return any(
        line.startswith("task completed") or _COMPLETE.match(line)
        for line in _declaration_lines(response.get("text", ""))
    )


def response_gives_up(response: dict[str, Any]) -> bool:
    return any(
        _GIVE_UP.match(line) for line in _declaration_lines(response.get("text", ""))
    )


def classify_stop(text: str | None, did_work: bool) -> str:
    if (text or "").strip():
        if response_is_complete({"text": text}):
            return "success" if did_work else "no_action"
        return "no_plan" if did_work else "parse_fail"
    return "success" if did_work else "no_action"
