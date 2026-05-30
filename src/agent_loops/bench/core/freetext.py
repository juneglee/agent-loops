from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_NAME_KEYS = ("name", "tool", "function", "tool_name", "function_name")
_ARG_KEYS = ("arguments", "parameters", "args", "params", "input")


def render_tool_catalog(tools: list[dict[str, Any]]) -> str:
    lines = ["Available tools:"]
    for t in tools:
        fn = t.get("function", t)
        params = (fn.get("parameters") or {}).get("properties", {}) or {}
        required = set((fn.get("parameters") or {}).get("required", []) or [])
        arg_desc = ", ".join(
            f"{k}{'' if k in required else '?'}: {v.get('type', 'string')}"
            for k, v in params.items()
        )
        lines.append(f"  {fn.get('name')}({arg_desc}) — {fn.get('description', '')}")
    return "\n".join(lines)


def render_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            render_tool_catalog(tools),
            "",
            'To use a tool, output exactly one JSON object: {"name": "tool", "arguments": {...}}',
            "If nothing is left to do, answer without JSON.",
        ]
    )


def _normalize(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    name = next((obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str)), None)
    if not name:
        return None
    args = next((obj[k] for k in _ARG_KEYS if isinstance(obj.get(k), dict)), None)
    return {"name": name, "arguments": args if args is not None else {}}


def _candidates(text: str) -> list[str]:
    found = [m.group(1).strip() for m in _FENCE.finditer(text)]
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and start >= 0:
                found.append(text[start : i + 1])
                start = -1
    return found


def parse_freetext_calls(text: str) -> list[dict[str, Any]] | None:
    for chunk in _candidates(text or ""):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            calls = [c for c in (_normalize(o) for o in obj) if c]
            if calls:
                return calls
        else:
            call = _normalize(obj)
            if call:
                return [call]
    return None
