from __future__ import annotations

import re
import shlex
from pathlib import Path

from agent_loops.tools.paths import PathEscape, resolve


class Blocked(Exception):
    pass


ALLOWED_PROGRAMS = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "find",
        "mv",
        "cp",
        "mkdir",
        "rm",
        "rmdir",
        "touch",
        "echo",
        "zip",
        "unzip",
        "pwd",
        "du",
        "grep",
        "tree",
        "basename",
        "dirname",
        "cd",
        "true",
        "false",
        "test",
        "[",
        "diff",
        "cut",
        "tr",
        "printf",
        "stat",
        "date",
    }
)
_SEPARATORS = {"|", "||", "&&", ";", "(", ")", "{", "}"}
_DENY_PATTERNS = [
    re.compile(p)
    for p in (
        r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(\s|$)",
        r"(^|[;&|(]\s*)cd\s+\.\.",
        r">\s*/dev/",
        r"\$\(|`",
    )
]
_PATH_KEYS = ("path", "source", "destination", "file_path", "dir_path")


def _programs(cmd: str) -> list[str]:
    try:
        tokens = shlex.split(cmd, posix=True) if cmd else []
    except ValueError as exc:
        raise Blocked(f"cannot parse command: {exc}") from exc
    heads: list[str] = []
    expect_head = True
    for token in tokens:
        if token in _SEPARATORS:
            expect_head = True
            continue
        if expect_head:
            heads.append(token)
            expect_head = False
    return heads


def _tokens(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd, posix=True) if cmd else []
    except ValueError as exc:
        raise Blocked(f"cannot parse command: {exc}") from exc


class Guard:
    def __init__(self, allowed_programs: frozenset[str] = ALLOWED_PROGRAMS) -> None:
        self.allowed_programs = frozenset(allowed_programs)

    def check(self, name: str, arguments: dict | None, root: Path | str) -> None:
        root = Path(root).resolve()
        if name == "bash":
            cmd = str((arguments or {}).get("command", ""))
            for pattern in _DENY_PATTERNS:
                if pattern.search(cmd):
                    raise Blocked(f"blocked command pattern: {cmd}")
            for token in _tokens(cmd):
                if token.startswith("~") or (
                    token.startswith("/") and not token.startswith(str(root))
                ):
                    raise Blocked(f"path outside the workspace: {token}")
            for program in _programs(cmd):
                if program not in self.allowed_programs:
                    raise Blocked(f"program not in the allow list: {program}")
            return
        for key in _PATH_KEYS:
            if key in (arguments or {}):
                try:
                    resolve(root, str(arguments[key]))
                except PathEscape as exc:
                    raise Blocked(str(exc)) from exc
        return


DEFAULT_GUARD = Guard()


def check(name: str, arguments: dict | None, root: Path | str) -> None:
    return DEFAULT_GUARD.check(name, arguments, root)
