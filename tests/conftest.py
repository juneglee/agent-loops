from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls_made = 0
        self.prompts: list[Any] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        self.prompts.append(messages)
        self.kwargs.append(dict(kwargs))
        if self.calls_made >= len(self._responses):
            raise AssertionError(
                f"unscripted LLM call #{self.calls_made + 1}: the loop calls more than expected"
            )
        resp = self._responses[self.calls_made]
        self.calls_made += 1
        return resp


class RecordingEnv:
    def __init__(self, tools: dict[str, Callable[..., Any]]) -> None:
        self._tools = dict(tools)
        self.executed: list[tuple[str, dict[str, Any]]] = []

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.executed.append((name, dict(arguments)))
        fn = self._tools.get(name)
        if fn is None:
            return {"ok": False, "error": f"unknown tool: {name}", "output": ""}
        try:
            return {"ok": True, "error": None, "output": fn(**arguments)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "output": ""}


@pytest.fixture
def scripted_llm():
    return ScriptedLLM


@pytest.fixture
def recording_env():
    return RecordingEnv
