from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Runtime:
    name: str
    default_base_url: str
    supports_grammar: bool
    grammar_field: str | None = None


RUNTIMES: dict[str, Runtime] = {
    "llamacpp": Runtime("llamacpp", "http://127.0.0.1:8080/v1", True, "grammar"),
    "ollama": Runtime("ollama", "http://127.0.0.1:11434/v1", False),
    "lmstudio": Runtime("lmstudio", "http://127.0.0.1:1234/v1", False),
    "vllm": Runtime("vllm", "http://127.0.0.1:8000/v1", False),
}

_LOOKS_LIKE_CALL = re.compile(r'"(name|function|tool_name)"\s*:', re.IGNORECASE)


def build_payload(
    *,
    runtime: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float = 0.0,
    grammar: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    rt = RUNTIMES[runtime]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
    if grammar is not None:
        if not rt.supports_grammar:
            raise ValueError(
                f"runtime {rt.name!r} does not support grammar-constrained decoding"
            )
        payload[rt.grammar_field] = grammar
    payload.update(extra)
    return payload


def parse_response(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tool_calls": None,
        "text": "",
        "parse_ok": True,
        "quiet_failure": False,
        "truncated": False,
        "raw": raw,
    }
    try:
        choice = raw["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        out["parse_ok"] = False
        return out

    out["truncated"] = choice.get("finish_reason") == "length"

    content = message.get("content") or ""
    out["text"] = content

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        if _LOOKS_LIKE_CALL.search(content):
            out["quiet_failure"] = True
        return out

    parsed = []
    for call in tool_calls:
        fn = call.get("function", call)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                out["parse_ok"] = False
                out["tool_calls"] = None
                return out
        if (
            not isinstance(fn.get("name"), str)
            or not fn.get("name")
            or not isinstance(args, dict)
        ):
            out["parse_ok"] = False
            out["tool_calls"] = None
            return out
        parsed.append({"name": fn.get("name"), "arguments": args})

    out["tool_calls"] = parsed
    return out


def call(
    *,
    runtime: str = "llamacpp",
    base_url: str | None = None,
    model: str = "local",
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    grammar: str | None = None,
    timeout: float = 120.0,
    **extra: Any,
) -> dict[str, Any]:
    import requests

    rt = RUNTIMES[runtime]
    url = (base_url or rt.default_base_url).rstrip("/") + "/chat/completions"
    payload = build_payload(
        runtime=runtime,
        model=model,
        messages=messages,
        tools=tools or [],
        grammar=grammar,
        **extra,
    )
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return parse_response(resp.json())
    except Exception as exc:  # noqa: BLE001
        return {
            "tool_calls": None,
            "text": "",
            "parse_ok": False,
            "quiet_failure": False,
            "truncated": False,
            "raw": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
