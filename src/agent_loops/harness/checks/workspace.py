from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from agent_loops.tools.paths import PathEscape, resolve


def _target(env: Any, path: str) -> Path | None:
    try:
        return resolve(env.root, path)
    except PathEscape:
        return None


def _check_write_file(env: Any, args: dict, obs: dict) -> str | None:
    target = _target(env, str(args.get("path", "")))
    if target is None or not target.is_file():
        return f"file missing after write_file: {args.get('path')}"
    if target.read_text(encoding="utf-8", errors="replace") != str(
        args.get("content", "")
    ):
        return f"content differs after write_file: {args.get('path')}"
    return None


def _check_edit_file(env: Any, args: dict, obs: dict) -> str | None:
    target = _target(env, str(args.get("path", "")))
    new = str(args.get("new", ""))
    if target is None or not target.is_file():
        return f"file missing after edit_file: {args.get('path')}"
    if new and new not in target.read_text(encoding="utf-8", errors="replace"):
        return f"new string missing after edit_file: {args.get('path')}"
    return None


def _segments(command: str) -> list[list[str]]:
    if "|" in command or "$(" in command or "`" in command:
        return []
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return []
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in ("&&", ";"):
            segments.append([])
        else:
            segments[-1].append(token)
    return [seg for seg in segments if seg]


def _operands(segment: list[str]) -> list[str]:
    return [tok for tok in segment[1:] if not tok.startswith("-")]


def _check_bash(env: Any, args: dict, obs: dict) -> str | None:
    for segment in _segments(str(args.get("command", ""))):
        program, operands = segment[0], _operands(segment)
        if program == "mv" and len(operands) >= 2:
            sources, destination = operands[:-1], _target(env, operands[-1])
            for src in sources:
                if any(ch in src for ch in "*?["):
                    continue
                source = _target(env, src)
                if source is not None and source.exists():
                    return f"source still exists after mv: {src}"
            if destination is not None and not destination.exists():
                return f"destination missing after mv: {operands[-1]}"
        elif program == "cp" and len(operands) >= 2:
            destination = _target(env, operands[-1])
            if destination is not None and not destination.exists():
                return f"destination missing after cp: {operands[-1]}"
        elif program == "mkdir":
            for op in operands:
                target = _target(env, op)
                if target is not None and not target.is_dir():
                    return f"directory missing after mkdir: {op}"
        elif program in ("rm", "rmdir"):
            for op in operands:
                if any(ch in op for ch in "*?["):
                    continue
                target = _target(env, op)
                if target is not None and target.exists():
                    return f"target still exists after {program}: {op}"
    return None


WORKSPACE_CHECKS: dict[str, Any] = {
    "write_file": _check_write_file,
    "edit_file": _check_edit_file,
    "bash": _check_bash,
}
