from __future__ import annotations

import ast
from typing import Any

_VAR_PREFIX = "#E"

_OPENERS = "([{"
_CLOSERS = ")]}"


def _at_value_start(before: str | list[str]) -> bool:
    prev = "".join(before).rstrip()
    return not prev or prev[-1] in "=,[({"


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False

    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'" and _at_value_start(buf):
            quote = ch
            buf.append(ch)
            continue
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
        elif ch == separator and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)

    if buf:
        parts.append("".join(buf))
    return parts


def extract_bracketed(text: str, open_index: int) -> tuple[str, int] | None:
    if open_index >= len(text) or text[open_index] not in "[(":
        return None

    depth = 0
    quote: str | None = None
    escaped = False

    for i in range(open_index, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'" and _at_value_start(text[open_index + 1 : i]):
            quote = ch
            continue
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i
    return None


def parse_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""

    if text.startswith(_VAR_PREFIX):
        return text

    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass

    if len(text) >= 2 and text[0] == "[" and text[-1] == "]":
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [parse_value(item) for item in _split_top_level(inner)]

    return text.strip("\"'")


def parse_args(argstr: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for part in _split_top_level(argstr):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        args[key.strip()] = parse_value(value)
    return args


def substitute(value: Any, results: dict[str, Any]) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(_VAR_PREFIX):
            if stripped in results:
                return results[stripped]
            if _is_bare_var(stripped):
                return value
        return _substitute_in_text(value, results)
    if isinstance(value, list):
        return [substitute(v, results) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, results) for k, v in value.items()}
    return value


def _is_bare_var(text: str) -> bool:
    return text.startswith(_VAR_PREFIX) and text[len(_VAR_PREFIX) :].isdigit()


def _substitute_in_text(text: str, results: dict[str, Any]) -> str:
    import re

    return re.sub(
        r"#E\d+",
        lambda m: str(results[m.group(0)]) if m.group(0) in results else m.group(0),
        text,
    )
