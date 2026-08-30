from __future__ import annotations

import re
from typing import Any

from tests.replay.bfcl import render_plan_args


def call_string(call: dict[str, Any]) -> str:
    return f"{call['name']}({render_plan_args(call['arguments'])})"


class UniversalReplay:
    def __init__(self, case: dict[str, Any], mode: str = "react") -> None:
        self.case = case
        self.mode = mode
        self.calls_made = 0
        self.parse_failures = 0
        self.quiet_failures = 0
        self.truncations = 0
        self._consumed: list[int] = [0] * len(case["turns"])
        tokens = (case.get("expect") or {}).get("answer_contains") or []
        self.final_text = "Final: " + (" ".join(tokens) if tokens else "done")

    def _turn_index(self, messages: Any) -> int:
        content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = str(m.get("content", ""))
                break
        marker = re.search(r"\[turn (\d+)\]", content)
        if marker:
            return int(marker.group(1))
        for i, text in enumerate(self.case["turns"]):
            if text in content:
                return i
        for i in range(len(self._consumed) - 1, -1, -1):
            if self._consumed[i]:
                return i
        return 0

    def _remaining(self, turn: int) -> list[dict[str, Any]]:
        return self.case["gt_calls"][turn][self._consumed[turn] :]

    def _take(self, turn: int, n: int | None = None) -> list[dict[str, Any]]:
        calls = self._remaining(turn)
        chosen = calls if n is None else calls[:n]
        self._consumed[turn] += len(chosen)
        return chosen

    def _native(self, turn: int) -> dict[str, Any]:
        remaining = self._remaining(turn)
        if not remaining:
            return {"tool_calls": None, "text": self.final_text, "parse_ok": True}
        if self.mode == "batch":
            calls = self._take(turn)
        else:
            calls = self._take(turn, 1)
        if self.mode == "code":
            code = "\n".join(
                f"print({c['name']}({render_plan_args(c['arguments'])}))" for c in calls
            )
            return {
                "tool_calls": [{"name": "execute_code", "arguments": {"code": code}}],
                "text": "",
                "parse_ok": True,
            }
        return {"tool_calls": [dict(c) for c in calls], "text": "", "parse_ok": True}

    def _text(self, turn: int, instruction: str) -> dict[str, Any]:
        remaining = self._remaining(turn)
        if "simple" in instruction and "complex" in instruction:
            return {"tool_calls": None, "text": "simple", "parse_ok": True}
        if "- subtask" in instruction:
            if remaining:
                return {
                    "tool_calls": None,
                    "text": f"- [turn {turn}] carry out the request as given",
                    "parse_ok": True,
                }
            return {"tool_calls": None, "text": self.final_text, "parse_ok": True}
        if "1. step description" in instruction:
            if remaining:
                return {
                    "tool_calls": None,
                    "text": f"1. [turn {turn}] perform the next call",
                    "parse_ok": True,
                }
            return {
                "tool_calls": None,
                "text": "1. report the result",
                "parse_ok": True,
            }
        if "Replan:" in instruction:
            if remaining:
                return {
                    "tool_calls": None,
                    "text": "Replan: more actions are needed",
                    "parse_ok": True,
                }
            return {"tool_calls": None, "text": self.final_text, "parse_ok": True}
        if "#E1" in instruction:
            if remaining:
                calls = self._take(turn)
                lines = [
                    f"#E{i} = {c['name']}[{render_plan_args(c['arguments'])}]"
                    for i, c in enumerate(calls, 1)
                ]
                return {"tool_calls": None, "text": "\n".join(lines), "parse_ok": True}
            return {"tool_calls": None, "text": self.final_text, "parse_ok": True}
        if "1. tool_name[arg=value]" in instruction:
            if remaining:
                calls = self._take(turn)
                lines = [
                    f"{i}. {c['name']}[{render_plan_args(c['arguments'])}]"
                    for i, c in enumerate(calls, 1)
                ]
                if "every step is independent" in instruction:
                    lines.append(self.final_text)
                return {"tool_calls": None, "text": "\n".join(lines), "parse_ok": True}
            return {"tool_calls": None, "text": self.final_text, "parse_ok": True}
        return {"tool_calls": None, "text": self.final_text, "parse_ok": True}

    def __call__(
        self, messages: Any, want: str | None = None, **_: Any
    ) -> dict[str, Any]:
        self.calls_made += 1
        turn = self._turn_index(messages)
        if want == "text":
            instruction = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    instruction = str(m.get("content", ""))
                    break
            return self._text(turn, instruction)
        return self._native(turn)


def replay_mode(loop_name: str) -> str:
    pieces = loop_name.split("+")
    if "codeact" in pieces:
        return "code"
    if "single_call" in pieces or "fixed_pipeline" in pieces:
        return "batch"
    return "react"
