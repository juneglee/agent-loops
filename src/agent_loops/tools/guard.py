from __future__ import annotations

import os
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
        "sed",
        "awk",
        "xargs",
        "tee",
        "paste",
        "seq",
        "bc",
        "rev",
        "fold",
        "nl",
        "expr",
        "readlink",
        "realpath",
        "file",
        "md5",
        "shasum",
        "cmp",
        "comm",
        "for",
        "do",
        "done",
        "while",
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "in",
        "exit",
        "export",
    }
)
_ROOTS = (
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/var",
    "/tmp",
    "/private",
    "/Users",
    "/home",
    "/opt",
    "/dev",
    "/root",
    "/proc",
    "/sys",
    "/Library",
    "/System",
    "/Applications",
)
_SEPARATORS = {"|", "||", "&&", ";", "(", ")", "{", "}"}
_DENY_PATTERNS = [
    re.compile(p)
    for p in (
        r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(\s|$)",
        r"(^|[;&|(]\s*)cd\s+\.\.",
        r">\s*/dev/",
    )
]
_DENY_SUBSTITUTION = re.compile(r"\$\(|`")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
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
        if token.endswith(";"):
            token = token[:-1]
            if expect_head and token and not _ASSIGNMENT.match(token):
                heads.append(token)
            expect_head = True
            continue
        if expect_head:
            if _ASSIGNMENT.match(token):
                continue
            heads.append(token)
            expect_head = False
    return heads


_ABSOLUTE_OK = ("/dev/null", "/bin/", "/usr/bin/", "/usr/local/bin/")


_PATH_LIKE = re.compile(r"^/(?:[A-Za-z0-9._-]|$)")


def _escapes(token: str, root: Path) -> bool:
    for part in token.split("="):
        if not _PATH_LIKE.match(part):
            continue
        if part.startswith(_ABSOLUTE_OK):
            continue
        real = os.path.realpath(part)
        if real == str(root) or real.startswith(str(root) + "/"):
            continue
        return True
    return False


def _tokens(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd, posix=True) if cmd else []
    except ValueError as exc:
        raise Blocked(f"cannot parse command: {exc}") from exc


class Guard:
    def __init__(
        self, allowed_programs: frozenset[str] | None = ALLOWED_PROGRAMS
    ) -> None:
        self.allowed_programs = (
            None if allowed_programs is None else frozenset(allowed_programs)
        )

    def check(self, name: str, arguments: dict | None, root: Path | str) -> None:
        root = Path(root).resolve()
        if name == "bash":
            cmd = str((arguments or {}).get("command", ""))
            for pattern in _DENY_PATTERNS:
                if pattern.search(cmd):
                    raise Blocked(f"blocked command pattern: {cmd}")
            if self.allowed_programs is not None and _DENY_SUBSTITUTION.search(cmd):
                raise Blocked(f"blocked command pattern: {cmd}")
            for token in _tokens(cmd):
                if token.startswith("~") or _escapes(token, root):
                    raise Blocked(f"path outside the workspace: {token}")
            if self.allowed_programs is None:
                return
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
PATH_GUARD = Guard(allowed_programs=None)


def check(name: str, arguments: dict | None, root: Path | str) -> None:
    return DEFAULT_GUARD.check(name, arguments, root)
