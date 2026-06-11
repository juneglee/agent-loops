from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent_loops.tools.fs import TEXT_SUFFIXES, ToolError
from agent_loops.tools.toolset import Toolset

_REPO = Path(__file__).resolve().parents[3]

_CODE_RUNNER = r"""
import json, sys, traceback
from pathlib import Path
root, code_path, calls_path, repo = sys.argv[1:5]
sys.path.insert(0, repo)
from agent_loops.tools import implementations
calls = []
namespace = {"__name__": "__codeact__"}
for _name, _fn in implementations(root).items():
    def _make(name, fn):
        def wrapper(*args, **kwargs):
            if args:
                raise TypeError(f"{name}: keyword arguments only (e.g. {name}(path='a.txt'))")
            calls.append({"name": name, "arguments": kwargs})
            return fn(**kwargs)
        wrapper.__name__ = name
        return wrapper
    namespace[_name] = _make(_name, _fn)
code = Path(code_path).read_text(encoding="utf-8")
rc = 0
try:
    exec(compile(code, "<codeact>", "exec"), namespace)
except BaseException:
    traceback.print_exc()
    rc = 1
finally:
    Path(calls_path).write_text(json.dumps(calls, ensure_ascii=False, default=str), encoding="utf-8")
sys.exit(rc)
"""


def _zip_digest(data: bytes) -> bytes | None:
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            entries = sorted(
                (info.filename, hashlib.sha256(zf.read(info)).hexdigest())
                for info in zf.infolist()
            )
    except (zipfile.BadZipFile, RuntimeError):
        return None
    return json.dumps(entries, ensure_ascii=False).encode("utf-8")


def _normalised_text(data: bytes) -> bytes | None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return (
        "\n".join(line.rstrip() for line in text.splitlines())
        .rstrip("\n")
        .encode("utf-8")
    )


class WorkspaceEnv:
    def __init__(
        self,
        fixture_dir: Path | str | None,
        code_timeout: float = 5.0,
        bash_timeout: float = 10.0,
        toolset_factory: Any = None,
    ) -> None:
        self.fixture_dir = Path(fixture_dir).resolve() if fixture_dir else None
        self._tmp = Path(tempfile.mkdtemp(prefix="ws_"))
        self.root = self._tmp / "root"
        if self.fixture_dir is not None:
            shutil.copytree(self.fixture_dir, self.root)
        else:
            self.root.mkdir()
        self.calls: list[dict[str, Any]] = []
        self.code_timeout = code_timeout
        self.bash_timeout = bash_timeout
        self._toolset = (
            toolset_factory(self.root)
            if toolset_factory is not None
            else Toolset(self.root, bash_timeout=bash_timeout)
        )
        self._snapshots: dict[str, Path] = {}
        self._code_enabled = False
        self._initial = self.snapshot()

    def tool_names(self) -> list[str]:
        return self._toolset.names() + (["execute_code"] if self._code_enabled else [])

    def enable_code_execution(self) -> None:
        self._code_enabled = True

    def execute(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        arguments = dict(arguments or {})
        if name == "execute_code":
            if not self._code_enabled:
                return {
                    "ok": False,
                    "error": "unknown tool: execute_code",
                    "output": "",
                }
            return self._execute_code(str(arguments.get("code", "")))

        self.calls.append({"name": name, "arguments": arguments})
        try:
            output = self._toolset.call(name, arguments)
        except ToolError as exc:
            return {"ok": False, "error": str(exc), "output": ""}
        return {"ok": True, "error": None, "output": output}

    def _execute_code(self, code: str) -> dict[str, Any]:
        try:
            compile(code, "<codeact>", "exec")
        except SyntaxError as exc:
            return {
                "ok": False,
                "error": f"SyntaxError: {exc}",
                "output": "",
                "calls": [],
            }
        work = Path(tempfile.mkdtemp(prefix="code_", dir=self._tmp))
        runner, code_path, calls_path = (
            work / "runner.py",
            work / "code.py",
            work / "calls.json",
        )
        runner.write_text(_CODE_RUNNER, encoding="utf-8")
        code_path.write_text(code, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(_REPO), "PYTHONIOENCODING": "utf-8"}
        try:
            done = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(runner),
                    str(self.root),
                    str(code_path),
                    str(calls_path),
                    str(_REPO),
                ],
                cwd=self.root,
                env=env,
                timeout=self.code_timeout,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"TimeoutError: code did not finish within {self.code_timeout:g}s (killed)",
                "output": "",
                "calls": [],
            }
        inner: list[dict[str, Any]] = []
        if calls_path.exists():
            try:
                inner = json.loads(calls_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                inner = []
        self.calls.extend(inner)
        if done.returncode != 0:
            return {
                "ok": False,
                "error": done.stderr.strip()[-2000:] or f"exit={done.returncode}",
                "output": done.stdout,
                "calls": inner,
            }
        return {"ok": True, "error": None, "output": done.stdout, "calls": inner}

    def state(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames.sort()
            rel_dir = Path(dirpath).relative_to(self.root)
            for d in dirnames:
                out[(rel_dir / d).as_posix()] = "<dir>"
            for f in sorted(filenames):
                path = Path(dirpath) / f
                data = path.read_bytes()
                if path.suffix.lower() in TEXT_SUFFIXES:
                    normalised = _normalised_text(data)
                    data = normalised if normalised is not None else data
                elif path.suffix.lower() == ".zip":
                    digest = _zip_digest(data)
                    data = digest if digest is not None else data
                out[(rel_dir / f).as_posix()] = hashlib.sha256(data).hexdigest()
        return out

    def snapshot(self) -> str:
        snap_id = f"snap{len(self._snapshots)}"
        target = self._tmp / snap_id
        shutil.copytree(self.root, target)
        self._snapshots[snap_id] = target
        return snap_id

    def restore(self, snap_id: str) -> None:
        source = self._snapshots[snap_id]
        shutil.rmtree(self.root)
        shutil.copytree(source, self.root)

    def reset(self) -> None:
        self.restore(self._initial)

    def close(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)
