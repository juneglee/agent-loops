from __future__ import annotations

import json
from typing import Any

from agent_loops.bench.core.freetext import render_tool_catalog
from agent_loops.bench.core.llm import call


class LocalLLM:
    def __init__(
        self,
        tools: list[dict[str, Any]],
        *,
        base_url: str | None = None,
        model: str = "local",
        timeout: float = 120.0,
        temperature: float = 0.0,
        seed: int | None = 0,
        extra_system: str | None = None,
    ) -> None:
        self.tools = tools
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.extra_system = extra_system
        self.calls_made = 0
        self.parse_failures = 0
        self.quiet_failures = 0
        self.truncations = 0
        self.errors = 0

    def __call__(
        self, messages: list[dict[str, Any]], want: str = "tool_calls", **_: Any
    ) -> dict[str, Any]:
        self.calls_made += 1
        text_mode = want == "text"
        prepared = _with_tool_docs(messages, self.tools) if text_mode else messages
        if self.extra_system:
            prepared = _append_system(prepared, self.extra_system)
        out = call(
            base_url=self.base_url,
            model=self.model,
            messages=_serialize(prepared),
            tools=None if text_mode else self.tools,
            timeout=self.timeout,
            temperature=self.temperature,
            **({"seed": self.seed} if self.seed is not None else {}),
        )
        if out.get("error"):
            self.errors += 1
        if not out.get("parse_ok", True):
            self.parse_failures += 1
        if out.get("quiet_failure"):
            self.quiet_failures += 1
        if out.get("truncated"):
            self.truncations += 1
        return out


def _append_system(messages: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            out[i] = {**m, "content": f"{m.get('content', '')}\n\n{text}"}
            return out
    return [{"role": "system", "content": text}, *out]


def _with_tool_docs(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not tools:
        return messages

    docs = render_tool_catalog(tools)
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            out[i] = {**m, "content": f"{m.get('content', '')}\n\n{docs}"}
            return out
    return [{"role": "system", "content": docs}, *out]


_RUNTIME_KEYS = frozenset({"raw", "parse_ok", "quiet_failure", "truncated"})


def _strip_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip_runtime(v) for k, v in value.items() if k not in _RUNTIME_KEYS
        }
    if isinstance(value, list):
        return [_strip_runtime(v) for v in value]
    return value


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _serialize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    counter = 0
    for m in messages:
        content = m.get("content")
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        if isinstance(content, dict):
            content = _strip_runtime(content)

        if (
            role == "assistant"
            and isinstance(content, dict)
            and content.get("tool_calls")
        ):
            pending_ids = []
            calls = []
            for call in content["tool_calls"]:
                counter += 1
                cid = f"call_{counter}"
                pending_ids.append(cid)
                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": json.dumps(
                                call.get("arguments") or {},
                                ensure_ascii=False,
                                default=str,
                            ),
                        },
                    }
                )
            out.append(
                {
                    "role": "assistant",
                    "content": str(content.get("text") or ""),
                    "tool_calls": calls,
                }
            )
            continue

        if role == "tool" and pending_ids:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_ids.pop(0),
                    "content": _as_text(content),
                }
            )
            continue

        if (
            isinstance(content, dict)
            and "text" in content
            and not content.get("tool_calls")
        ):
            content = str(content.get("text") or "")
        text = _as_text(content)
        if role == "tool":
            role, text = "user", f"[observation] {text}"
        out.append({"role": role, "content": text})
    return out
