from __future__ import annotations

import json
from typing import Any

WARN_AFTER = 3
STOP_AFTER = 4

_WARNING = (
    "[WARNING] Repetitive pattern detected: this exact tool call was made "
    "{n} times in a row. Change the approach or answer with what you know."
)
_STOP = "Doom loop: agent stuck after {n} consecutive repetitions"


def _key(name: str, arguments: dict[str, Any] | None) -> str:
    return name + json.dumps(arguments or {}, sort_keys=True, default=str)


class _DoomLoopEnv:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._last: str | None = None
        self._streak = 0
        self.stopped = False

    def execute(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        key = _key(name, arguments)
        self._streak = self._streak + 1 if key == self._last else 1
        self._last = key
        if self._streak >= STOP_AFTER:
            self.stopped = True
            return {"ok": False, "error": _STOP.format(n=self._streak), "output": ""}
        observation = self._inner.execute(name, arguments)
        if self._streak >= WARN_AFTER:
            output = str(observation.get("output") or "")
            warning = _WARNING.format(n=self._streak)
            observation = {**observation, "output": f"{output}\n{warning}".strip()}
        return observation

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class _DoomLoopLLM:
    def __init__(self, inner: Any, env: _DoomLoopEnv) -> None:
        self._inner = inner
        self._env = env

    def __call__(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        if self._env.stopped:
            return {
                "tool_calls": None,
                "text": _STOP.format(n=STOP_AFTER),
                "done": True,
            }
        return self._inner(messages=messages, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class _DoomLoopLayer:
    def wrap_env(self, env: Any) -> Any:
        self._env = _DoomLoopEnv(env)
        return self._env

    def wrap_llm(self, llm: Any) -> Any:
        return _DoomLoopLLM(llm, self._env)

    def finish(self, trace: Any) -> None:
        if self._env.stopped:
            trace.terminated_by = "doom_loop"


def doom_loop() -> _DoomLoopLayer:
    return _DoomLoopLayer()


doom_loop.NAME = "doom_loop"
