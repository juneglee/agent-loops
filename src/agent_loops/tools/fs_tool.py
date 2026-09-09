"""Tools of the fs_tool set (arguments in call order):
Read(file_path, offset, limit)      raw file content; line numbers only with offset or limit
Write(file_path, content)           create or overwrite a file, parent directories are created
Edit(file_path, old_string, new_string, replace_all)
                                    replace one unique string; the file must have been Read first
Bash(command, timeout)              run an allow-listed shell command in the workspace
Glob(pattern, path)                 matching paths sorted by modification time
Grep(pattern, path, glob, output_mode)
                                    regex search; output_mode is content, files_with_matches or count
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from agent_loops.tools.fs import ToolError, _guarded, _is_text, _read_pdf
from agent_loops.tools.guard import DEFAULT_GUARD, Blocked, Guard

TOOLS_VERSION = "fs_tool"
DEFAULT_READ_LINES = 2000
MAX_LINE_CHARS = 2000
MAX_GLOB_RESULTS = 200
MAX_GREP_MATCHES = 200
MAX_OUTPUT_CHARS = 30_000
DEFAULT_BASH_TIMEOUT = 120.0
MAX_BASH_TIMEOUT = 600.0


def make(
    root: Path | str,
    bash_timeout: float = DEFAULT_BASH_TIMEOUT,
    guard: Guard | None = None,
    seen_paths: Iterable[str] = (),
) -> dict[str, Callable[..., str]]:
    root = Path(root).resolve()
    guard = guard or DEFAULT_GUARD
    seen: set[Path] = {(root / p).resolve() for p in seen_paths}

    def _rel(target: Path) -> str:
        return target.relative_to(root).as_posix()

    def read(
        file_path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        target = _guarded(root, file_path)
        if not target.exists():
            raise ToolError(f"File does not exist: {file_path}")
        if target.is_dir():
            raise ToolError(f"Path is a directory, not a file: {file_path}")
        seen.add(target)
        if target.suffix.lower() == ".pdf":
            return _read_pdf(target, None)
        data = target.read_bytes()
        if not _is_text(data):
            raise ToolError(f"Cannot read binary file as text: {file_path}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"File is not valid UTF-8: {file_path}") from exc
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if not lines:
            return "(empty file)"
        paged = offset is not None or limit is not None
        start = max(int(offset), 1) if offset else 1
        count = int(limit) if limit else DEFAULT_READ_LINES
        chosen = lines[start - 1 : start - 1 + count]
        if not chosen:
            raise ToolError(
                f"offset {start} is beyond the end of the file ({len(lines)} lines)"
            )
        out = []
        for number, line in enumerate(chosen, start=start):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + "... (line truncated)"
            out.append(f"{number:6d}\t{line}" if paged else line)
        remaining = len(lines) - (start - 1 + len(chosen))
        if remaining > 0:
            out.append(
                f"... ({remaining} more lines; use offset={start + len(chosen)} to continue)"
            )
        return "\n".join(out)

    def write(file_path: str, content: str) -> str:
        target = _guarded(root, file_path)
        if target.is_dir():
            raise ToolError(f"Path is a directory, not a file: {file_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        seen.add(target)
        return f"File written successfully: {file_path}"

    def edit(
        file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        target = _guarded(root, file_path)
        if not target.is_file():
            raise ToolError(f"File does not exist: {file_path}")
        if target not in seen:
            raise ToolError(
                f"File has not been read yet. Read it first before editing: {file_path}"
            )
        if old_string == new_string:
            raise ToolError(
                "old_string and new_string are identical; nothing to change"
            )
        text = target.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            raise ToolError(f"old_string not found in {file_path}")
        if count > 1 and not replace_all:
            raise ToolError(
                f"Found {count} matches of old_string in {file_path}. "
                "Provide more surrounding lines to make it unique, or set replace_all to true"
            )
        updated = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        target.write_text(updated, encoding="utf-8")
        return (
            f"Edited {file_path}: replaced {count if replace_all else 1} occurrence(s)"
        )

    def bash(command: str, timeout: float | None = None) -> str:
        try:
            guard.check("bash", {"command": command}, root)
        except Blocked as exc:
            raise ToolError(f"Command blocked: {exc}") from exc
        seconds = min(float(timeout) if timeout else bash_timeout, MAX_BASH_TIMEOUT)
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        try:
            done = subprocess.run(
                ["/bin/bash", "-c", command],
                cwd=root,
                env=env,
                timeout=seconds,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"Command timed out after {seconds:g}s: {command}") from exc
        out = done.stdout
        if done.stderr:
            out = f"{out}{'' if not out or out.endswith(chr(10)) else chr(10)}{done.stderr}"
        if len(out) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            out = f"{out[:half]}\n... ({len(out) - MAX_OUTPUT_CHARS} chars omitted) ...\n{out[-half:]}"
        if done.returncode != 0:
            raise ToolError(f"Exit code {done.returncode}\n{out.strip() or command}")
        return out if out.strip() else "(no output)"

    def glob(pattern: str, path: str | None = None) -> str:
        base = _guarded(root, path or ".")
        if not base.is_dir():
            raise ToolError(f"Path is not a directory: {path}")
        matches = [p for p in base.glob(pattern) if p.is_file()]
        if not matches:
            return "No files found"
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        shown = matches[:MAX_GLOB_RESULTS]
        out = [_rel(p) for p in shown]
        if len(matches) > len(shown):
            out.append(
                f"... ({len(matches) - len(shown)} more files; narrow the pattern)"
            )
        return "\n".join(out)

    def grep(
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "content",
    ) -> str:
        if output_mode not in ("content", "files_with_matches", "count"):
            raise ToolError(f"Unknown output_mode: {output_mode}")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}") from exc
        base = _guarded(root, path or ".")
        if not base.exists():
            raise ToolError(f"Path does not exist: {path}")
        files = (
            [base]
            if base.is_file()
            else sorted(p for p in base.rglob("*") if p.is_file())
        )
        lines: list[str] = []
        per_file: dict[str, int] = {}
        total = 0
        for file in files:
            if glob and not fnmatch.fnmatch(file.name, glob):
                continue
            data = file.read_bytes()
            if not _is_text(data):
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            rel = _rel(file)
            for number, line in enumerate(text.splitlines(), start=1):
                if not regex.search(line):
                    continue
                total += 1
                per_file[rel] = per_file.get(rel, 0) + 1
                if output_mode == "content":
                    lines.append(f"{rel}:{number}:{line}")
                if total >= MAX_GREP_MATCHES:
                    break
            if total >= MAX_GREP_MATCHES:
                break
        if output_mode == "files_with_matches":
            lines = list(per_file)
        elif output_mode == "count":
            lines = [f"{rel}:{n}" for rel, n in per_file.items()]
        if not lines:
            return "No matches found"
        if total >= MAX_GREP_MATCHES:
            lines.append(
                f"... (stopped at {MAX_GREP_MATCHES} matches; narrow the search)"
            )
        return "\n".join(lines)

    return {
        "Read": read,
        "Write": write,
        "Edit": edit,
        "Bash": bash,
        "Glob": glob,
        "Grep": grep,
    }


def _schema(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


SCHEMAS: list[dict[str, Any]] = [
    _schema(
        "Read",
        "Reads a file from the local filesystem. The file_path must be a path inside the workspace. "
        "By default it reads up to 2000 lines starting from the beginning of the file and returns the raw content. "
        "When offset or limit is given, each line is prefixed with its line number and a tab; never copy that prefix into a file. "
        "Any lines longer than 2000 characters are truncated. PDFs are returned as extracted text. "
        "When you already know which part of the file you need, only read that part with offset and limit. "
        "You must read a file before you edit or overwrite it.",
        {
            "file_path": {
                "type": "string",
                "description": "The path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "The line number to start reading from. Only provide if the file is too large to read at once",
            },
            "limit": {
                "type": "integer",
                "description": "The number of lines to read. Only provide if the file is too large to read at once",
            },
        },
        ["file_path"],
    ),
    _schema(
        "Write",
        "Writes a file to the local filesystem, overwriting if one exists. "
        "Use it for creating a new file or fully replacing an existing one. "
        "Missing parent folders are created. For partial changes, use Edit instead. "
        "Always prefer editing existing files; never create new files unless the task explicitly requires them.",
        {
            "file_path": {
                "type": "string",
                "description": "The path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        ["file_path", "content"],
    ),
    _schema(
        "Edit",
        "Performs exact string replacement in a file. You must Read the file before editing, or the call will fail. "
        "old_string must match the file exactly, including indentation, and be unique; the edit fails otherwise. "
        "Never include the line-number prefix from Read output in old_string or new_string. "
        "Set replace_all to true to replace every occurrence instead.",
        {
            "file_path": {
                "type": "string",
                "description": "The path to the file to modify",
            },
            "old_string": {"type": "string", "description": "The text to replace"},
            "new_string": {
                "type": "string",
                "description": "The text to replace it with (must be different from old_string)",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences of old_string (default false)",
            },
        },
        ["file_path", "old_string", "new_string"],
    ),
    _schema(
        "Bash",
        "Executes a bash command inside the workspace and returns its output. "
        "Use it to move, copy, delete files, create folders, and zip or unzip. "
        "Prefer the dedicated tools over shell commands when one fits: use Read instead of cat, Glob instead of find, Grep instead of grep. "
        "Network access, other programs, and paths outside the workspace are blocked. "
        "On failure, the exit code and error output are returned.",
        {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout": {
                "type": "number",
                "description": "Optional timeout in seconds (max 600)",
            },
        },
        ["command"],
    ),
    _schema(
        "Glob",
        'Fast file pattern matching tool. Supports glob patterns like "*.md" or "**/*.py" and '
        "returns matching file paths sorted by modification time. "
        "Use this tool when you need to find files by name patterns.",
        {
            "pattern": {
                "type": "string",
                "description": "The glob pattern to match files against",
            },
            "path": {
                "type": "string",
                "description": "The directory to search in. If not specified, the workspace root is used",
            },
        },
        ["pattern"],
    ),
    _schema(
        "Grep",
        "A search tool for file contents. Supports full regex syntax. "
        'Filter files with the glob parameter (e.g. "*.txt"). '
        'output_mode: "content" shows matching lines as path:line:content (default), '
        '"files_with_matches" shows only file paths, "count" shows match counts per file. '
        "Use this tool for searching text inside files; to find files by name use Glob.",
        {
            "pattern": {
                "type": "string",
                "description": "The regular expression pattern to search for in file contents",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in. Defaults to the workspace root",
            },
            "glob": {
                "type": "string",
                "description": 'Glob pattern to filter files (e.g. "*.md")',
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "Output mode (default: content)",
            },
        },
        ["pattern"],
    ),
]
