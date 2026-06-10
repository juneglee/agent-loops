from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_loops.tools.paths import PathEscape, resolve

MAX_READ_LINES = 2000
MAX_READ_CHARS = 200_000
MAX_GREP_MATCHES = 200
TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".log",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".py",
    ".html",
    ".xml",
}


class ToolError(Exception):
    pass


def _guarded(root: Path, path: str) -> Path:
    try:
        return resolve(root, path)
    except PathEscape as exc:
        raise ToolError(str(exc)) from exc


def _is_text(data: bytes) -> bool:
    return b"\x00" not in data[:8192]


def _parse_pages(pages: str | None, n: int) -> list[int]:
    if not pages:
        return list(range(n))
    out: list[int] = []
    for part in str(pages).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a) - 1, min(int(b), n)))
        elif part:
            out.append(int(part) - 1)
    bad = [i for i in out if i < 0 or i >= n]
    if bad:
        raise ToolError(f"page range out of bounds: {pages} (total {n} pages)")
    return out


def _read_pdf(target: Path, pages: str | None) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ToolError("reading PDF requires pypdf (pip install pypdf)") from exc
    reader = PdfReader(str(target))
    chunks = []
    for index in _parse_pages(pages, len(reader.pages)):
        chunks.append(f"[page {index + 1}]\n{reader.pages[index].extract_text() or ''}")
    return "\n".join(chunks)


def make(root: Path | str) -> dict[str, Callable[..., str]]:
    root = Path(root).resolve()

    def read_file(
        path: str,
        offset: int | None = None,
        limit: int | None = None,
        pages: str | None = None,
    ) -> str:
        target = _guarded(root, path)
        if not target.exists():
            raise ToolError(f"file not found: {path}")
        if target.is_dir():
            raise ToolError(f"this is a directory — use list_dir: {path}")
        if target.suffix.lower() == ".pdf":
            return _read_pdf(target, pages)
        data = target.read_bytes()
        if not _is_text(data):
            raise ToolError(f"file cannot be read as text: {path}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"file cannot be read as UTF-8: {path}") from exc
        lines = text.splitlines(keepends=True)
        start = max(int(offset) - 1, 0) if offset else 0
        end = start + int(limit) if limit else len(lines)
        chosen = lines[start:end]
        omitted = len(chosen) - MAX_READ_LINES
        if omitted > 0:
            chosen = chosen[:MAX_READ_LINES] + [f"... ({omitted} lines omitted)\n"]
        out = "".join(chosen)
        if len(out) > MAX_READ_CHARS:
            out = (
                out[:MAX_READ_CHARS]
                + f"\n... ({len(out) - MAX_READ_CHARS} chars omitted)\n"
            )
        return out

    def write_file(path: str, content: str) -> str:
        target = _guarded(root, path)
        if target.is_dir():
            raise ToolError(f"cannot write to a directory: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return f"wrote {len(str(content).encode('utf-8'))} bytes to {path}"

    def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
        target = _guarded(root, path)
        if not target.is_file():
            raise ToolError(f"file not found: {path}")
        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ToolError(f"not found: {old!r} (file {path})")
        if count > 1 and not replace_all:
            raise ToolError(
                f"matches {count} times — use a longer old or replace_all=true"
            )
        target.write_text(
            text.replace(old, new) if replace_all else text.replace(old, new, 1),
            encoding="utf-8",
        )
        return f"{path}: replaced {count if replace_all else 1} occurrence(s)"

    def list_dir(
        path: str = ".", pattern: str | None = None, recursive: bool = False
    ) -> str:
        target = _guarded(root, path)
        if not target.is_dir():
            raise ToolError(f"directory not found: {path}")
        entries: list[str] = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(target):
                dirnames.sort()
                rel = Path(dirpath).relative_to(target)
                for d in dirnames:
                    entries.append(str(rel / d) + "/")
                for f in sorted(filenames):
                    entries.append(str(rel / f))
        else:
            for child in sorted(target.iterdir(), key=lambda p: p.name):
                entries.append(child.name + "/" if child.is_dir() else child.name)
        if pattern:
            entries = [
                e for e in entries if fnmatch.fnmatch(Path(e.rstrip("/")).name, pattern)
            ]
        entries = [e.replace(os.sep, "/") for e in entries]
        return "\n".join(entries) if entries else "(empty)"

    def grep(pattern: str, path: str = ".", glob: str | None = None) -> str:
        target = _guarded(root, path)
        if not target.exists():
            raise ToolError(f"path not found: {path}")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regex: {exc}") from exc
        files = (
            [target]
            if target.is_file()
            else sorted(p for p in target.rglob("*") if p.is_file())
        )
        matches: list[str] = []
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
            rel = file.relative_to(root).as_posix()
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{number}:{line}")
                    if len(matches) >= MAX_GREP_MATCHES:
                        matches.append(f"... (stopped at {MAX_GREP_MATCHES} matches)")
                        return "\n".join(matches)
        return "\n".join(matches)

    return {
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "list_dir": list_dir,
        "grep": grep,
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


FS_SCHEMAS: list[dict[str, Any]] = [
    _schema(
        "read_file",
        "Read the contents of a file. Text files are returned as-is; .pdf files have their text extracted (use pages to select a range). "
        "Binary files such as images cannot be read. Read long files in line chunks with offset/limit.",
        {
            "path": {"type": "string", "description": "Path relative to the workspace"},
            "offset": {
                "type": "integer",
                "description": "Starting line (1-based). Omit to start from the beginning",
            },
            "limit": {
                "type": "integer",
                "description": "Number of lines to read. Omit to read to the end",
            },
            "pages": {
                "type": "string",
                "description": 'PDF page range, e.g. "1-3" or "2"',
            },
        },
        ["path"],
    ),
    _schema(
        "write_file",
        "Write content to a file. If it already exists, it is **overwritten**. Missing parent folders are created. "
        "To change only part of an existing file, use edit_file.",
        {
            "path": {"type": "string", "description": "Path relative to the workspace"},
            "content": {"type": "string", "description": "Full contents of the file"},
        },
        ["path", "content"],
    ),
    _schema(
        "edit_file",
        "Replace the string old with new inside a file. old must appear **exactly once** in the file — "
        "if it appears multiple times, use a longer old or set replace_all to true. Fails if it is not found.",
        {
            "path": {"type": "string", "description": "Path relative to the workspace"},
            "old": {"type": "string", "description": "String to replace (exact match)"},
            "new": {"type": "string", "description": "New string"},
            "replace_all": {
                "type": "boolean",
                "description": "Replace all matches (default false)",
            },
        },
        ["path", "old", "new"],
    ),
    _schema(
        "list_dir",
        "List the direct entries of a folder (folders end with /). Use recursive=true to include subfolders, "
        'and pattern (e.g. "*.md") to filter by name.',
        {
            "path": {
                "type": "string",
                "description": "Folder path. Omit for the workspace root",
            },
            "pattern": {"type": "string", "description": 'glob pattern, e.g. "*.md"'},
            "recursive": {
                "type": "boolean",
                "description": "Include subfolders (default false)",
            },
        },
        [],
    ),
    _schema(
        "grep",
        "Search file contents with a regex. Results are one line each as `path:line:content`. If path is a folder, all subfolders are searched. "
        "This searches **contents**, not file names — for names use the pattern of list_dir.",
        {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {
                "type": "string",
                "description": "File or folder. Omit for the root",
            },
            "glob": {
                "type": "string",
                "description": 'Target file name filter, e.g. "*.txt"',
            },
        },
        ["pattern"],
    ),
]
