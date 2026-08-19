from __future__ import annotations

from typing import Any

TODO_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "update_todo",
        "description": (
            "Update the task list (todo) by **full replacement**. Pass the complete list reflecting "
            "progress so far. Set status to completed for finished items."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["task", "status"],
                    },
                }
            },
            "required": ["tasks"],
        },
    },
}

_STATE_ATTR = "_harness_todo"


def _render(tasks: list[dict[str, Any]]) -> str:
    lines = [f"- [{t.get('status', '?')}] {t.get('task', '')}" for t in tasks]
    return (
        "Current TODO (when progress changes, replace the whole list with update_todo):\n"
        + "\n".join(lines)
    )


class _TodoEnv:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def execute(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if name == "update_todo":
            tasks = list((arguments or {}).get("tasks") or [])
            setattr(self._inner, _STATE_ATTR, tasks)
            return {
                "ok": True,
                "error": None,
                "output": f"todo updated ({len(tasks)} items)",
            }
        return self._inner.execute(name, arguments)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class _TodoLLM:
    def __init__(self, inner: Any, state_holder: Any) -> None:
        self._inner = inner
        self._state_holder = state_holder

    def __call__(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        tasks = getattr(self._state_holder, _STATE_ATTR, [])
        if tasks:
            messages = [*messages, {"role": "system", "content": _render(tasks)}]
        return self._inner(messages=messages, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class _TodoLayer:
    def wrap_env(self, env: Any) -> Any:
        self._state_holder = env
        if not hasattr(env, _STATE_ATTR):
            setattr(env, _STATE_ATTR, [])
        return _TodoEnv(env)

    def wrap_llm(self, llm: Any) -> Any:
        return _TodoLLM(llm, self._state_holder)


def todo() -> _TodoLayer:
    return _TodoLayer()


todo.NAME = "todo"
todo.TOOL_SCHEMA = TODO_TOOL_SCHEMA
