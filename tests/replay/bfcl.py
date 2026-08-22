from __future__ import annotations

import ast
from typing import Any


def parse_call(source: str) -> dict[str, Any]:
    node = ast.parse(source.strip(), mode="eval").body
    args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    if node.args:
        args["__positional__"] = [ast.literal_eval(a) for a in node.args]
    return {"name": node.func.id, "arguments": args}


def render_plan_args(arguments: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in arguments.items())


class _TurnReplayer:
    def __init__(self, turns: list[list[str]]) -> None:
        self._turns = list(turns)
        self._turn_index = 0
        self.calls_made = 0
        self.parse_failures = 0
        self.quiet_failures = 0
        self.truncations = 0

    def _current(self) -> list[str]:
        if self._turn_index < len(self._turns):
            return self._turns[self._turn_index]
        return []

    def _advance(self) -> None:
        self._turn_index += 1


class ReActReplay(_TurnReplayer):
    def __init__(self, turns: list[list[str]]) -> None:
        super().__init__(turns)
        self._queue: list[dict] | None = None

    def __call__(self, messages: Any, **_: Any) -> dict[str, Any]:
        self.calls_made += 1
        if self._queue is None:
            self._queue = [parse_call(c) for c in self._current()]
        if self._queue:
            return {"tool_calls": [self._queue.pop(0)], "text": "", "parse_ok": True}
        self._queue = None
        self._advance()
        return {"tool_calls": None, "text": "Final: done", "parse_ok": True}


class SingleCallReplay(_TurnReplayer):
    def __call__(self, messages: Any, **_: Any) -> dict[str, Any]:
        self.calls_made += 1
        calls = self._current()
        self._advance()
        if not calls:
            return {"tool_calls": None, "text": "nothing to do", "parse_ok": True}
        return {"tool_calls": [parse_call(calls[0])], "text": "", "parse_ok": True}


class PlanReplay(_TurnReplayer):
    def __init__(
        self,
        turns: list[list[str]],
        style: str = "numbered",
        follow_up: bool = True,
        first_only: bool = False,
    ) -> None:
        super().__init__(turns)
        self._style = style
        self._follow_up = follow_up
        self._first_only = first_only
        self._planned = False

    def _render(self, calls: list[str]) -> str:
        lines = []
        for index, source in enumerate(calls, start=1):
            call = parse_call(source)
            args = render_plan_args(call["arguments"])
            if self._style == "variable":
                lines.append(f"#E{index} = {call['name']}[{args}]")
            else:
                lines.append(f"{index}. {call['name']}[{args}]")
        return "\n".join(lines)

    def __call__(self, messages: Any, **_: Any) -> dict[str, Any]:
        self.calls_made += 1
        if self._planned:
            self._planned = False
            self._advance()
            return {"tool_calls": None, "text": "Final: done", "parse_ok": True}

        calls = self._current()
        if self._first_only:
            calls = calls[:1]
        if not calls:
            self._advance()
            return {"tool_calls": None, "text": "Final: done", "parse_ok": True}

        if self._follow_up:
            self._planned = True
        else:
            self._advance()
        return {"tool_calls": None, "text": self._render(calls), "parse_ok": True}


class HierarchicalReplay(_TurnReplayer):
    def __init__(self, turns: list[list[str]]) -> None:
        super().__init__(turns)
        self._queue: list[dict] | None = None
        self._planning = True

    def __call__(
        self, messages: Any, want: str | None = None, **_: Any
    ) -> dict[str, Any]:
        self.calls_made += 1
        if want == "text":
            if self._planning and self._current():
                self._planning = False
                self._queue = [parse_call(c) for c in self._current()]
                return {
                    "tool_calls": None,
                    "text": "- carry out this turn's request as given",
                    "parse_ok": True,
                }
            self._planning = True
            self._queue = None
            self._advance()
            return {"tool_calls": None, "text": "Final: done", "parse_ok": True}
        if self._queue:
            return {"tool_calls": [self._queue.pop(0)], "text": "", "parse_ok": True}
        return {"tool_calls": None, "text": "Task completed", "parse_ok": True}


class GateReplay:
    def __init__(self, turns: list[list[str]]) -> None:
        self._inner = ReActReplay(turns)
        self.calls_made = 0
        self.parse_failures = 0
        self.quiet_failures = 0
        self.truncations = 0

    def __call__(
        self, messages: Any, want: str | None = None, **kw: Any
    ) -> dict[str, Any]:
        self.calls_made += 1
        if want == "text":
            return {"tool_calls": None, "text": "simple", "parse_ok": True}
        return self._inner(messages, **kw)
