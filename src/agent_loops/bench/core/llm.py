from __future__ import annotations

import json
import re
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"

_LOOKS_LIKE_CALL = re.compile(r'"(name|function|tool_name)"\s*:', re.IGNORECASE)


def build_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float = 0.0,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
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
    base_url: str | None = None,
    model: str = "local",
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    timeout: float = 120.0,
    **extra: Any,
) -> dict[str, Any]:
    import requests

    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    payload = build_payload(
        model=model,
        messages=messages,
        tools=tools or [],
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
