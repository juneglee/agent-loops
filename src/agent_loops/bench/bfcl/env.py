from __future__ import annotations

import builtins
import contextlib
import copy
import importlib
import inspect
import io
import math
import multiprocessing
import os
import resource
import traceback
from typing import Any


def _backend_config():
    from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING

    try:
        from bfcl_eval.constants.executable_backend_config import STATELESS_CLASSES
    except ImportError:
        STATELESS_CLASSES = ()
    return CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES


_SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)
SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES
}


class BFCLEnv:
    def __init__(
        self,
        case: dict[str, Any],
        long_context: bool = False,
        code_timeout: float = 5.0,
        code_memory_mb: int = 512,
    ) -> None:
        self.case = case
        self.calls: list[dict[str, Any]] = []
        self.code_actions: list[dict[str, Any]] = []
        self._methods: dict[str, Any] = {}
        self._instances: dict[str, Any] = {}
        self._code_enabled = False
        self.code_timeout = code_timeout
        self.code_memory_mb = code_memory_mb

        initial_config = case.get("initial_config", {})
        CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES = _backend_config()
        for class_name in case["involved_classes"]:
            module = importlib.import_module(CLASS_FILE_PATH_MAPPING[class_name])
            instance = getattr(module, class_name)()
            if class_name not in STATELESS_CLASSES:
                instance._load_scenario(
                    copy.deepcopy(initial_config.get(class_name, {})),
                    long_context=long_context,
                )
            self._instances[class_name] = instance
        self._bind_methods()

    def _bind_methods(self) -> None:
        self._methods = {}
        for instance in self._instances.values():
            for name, method in inspect.getmembers(
                instance, predicate=inspect.ismethod
            ):
                if not name.startswith("_"):
                    self._methods[name] = method

    def enable_code_execution(self) -> None:
        self._code_enabled = True

    @property
    def tool_names(self) -> list[str]:
        names = sorted(self._methods)
        return [*names, "execute_code"] if self._code_enabled else names

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

        method = self._methods.get(name)
        if method is None:
            return {"ok": False, "error": f"unknown tool: {name}", "output": ""}

        try:
            output = method(**arguments)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "output": ""}

        if isinstance(output, dict) and "error" in output:
            return {"ok": False, "error": str(output["error"]), "output": output}
        return {"ok": True, "error": None, "output": output}

    def _tool_namespace(self) -> dict[str, Any]:
        namespace: dict[str, Any] = {}

        for tool_name, method in self._methods.items():

            def wrapper(
                *args: Any, _name: str = tool_name, _fn: Any = method, **kwargs: Any
            ):
                bound = dict(kwargs)
                if args:
                    try:
                        params = [
                            p
                            for p in inspect.signature(_fn).parameters
                            if p not in ("self",)
                        ]
                        bound.update(dict(zip(params, args, strict=False)))
                    except (TypeError, ValueError):
                        bound["__positional__"] = list(args)
                result = self.execute(_name, bound)
                return result["output"]

            namespace[tool_name] = wrapper

        return namespace

    def _execute_code(self, code: str) -> dict[str, Any]:
        before = len(self.calls)
        self.code_actions.append({"code": code})
        try:
            compile(code, "<codeact>", "exec")
        except SyntaxError as exc:
            return {
                "ok": False,
                "error": f"SyntaxError: {exc}",
                "output": "",
                "calls": [],
            }

        ctx = multiprocessing.get_context("fork")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_run_code_in_child,
            args=(
                self,
                code,
                before,
                child_conn,
                self.code_timeout,
                self.code_memory_mb,
            ),
        )
        proc.start()
        child_conn.close()
        try:
            if not parent_conn.poll(self.code_timeout):
                proc.kill()
                proc.join(1)
                return {
                    "ok": False,
                    "error": f"TimeoutError: code did not finish within {self.code_timeout:g}s (killed)",
                    "output": "",
                    "calls": [],
                }
            try:
                result = parent_conn.recv()
            except EOFError:
                proc.join(1)
                return {
                    "ok": False,
                    "error": "RuntimeError: code process died (resource limit?)",
                    "output": "",
                    "calls": [],
                }
        finally:
            parent_conn.close()
            if proc.is_alive():
                proc.kill()
            proc.join(1)

        self._instances = result["instances"]
        self._bind_methods()
        self.calls.extend(result["calls"])
        return result["observation"]


def _run_code_in_child(
    env: BFCLEnv,
    code: str,
    before: int,
    conn: Any,
    timeout: float,
    memory_mb: int,
) -> None:
    cpu = max(1, math.ceil(timeout) + 1)
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    with contextlib.suppress(ValueError, OSError):
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    namespace = env._tool_namespace()
    namespace["__builtins__"] = dict(SAFE_BUILTINS)
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(  # noqa: S102
                compile(code, "<codeact>", "exec"), namespace
            )
        observation = {
            "ok": True,
            "error": None,
            "output": stdout.getvalue(),
            "calls": list(env.calls[before:]),
        }
    except Exception as exc:  # noqa: BLE001
        observation = {
            "ok": False,
            "error": _model_facing_traceback(exc, sorted(env._methods)),
            "output": stdout.getvalue(),
            "calls": list(env.calls[before:]),
        }
    try:
        conn.send(
            {
                "observation": observation,
                "instances": env._instances,
                "calls": list(env.calls[before:]),
            }
        )
    finally:
        conn.close()
        os._exit(0)


def _model_facing_traceback(exc: BaseException, functions: list[str]) -> str:
    frames = [
        f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<codeact>"
    ]
    text = (
        "Traceback (most recent call last):\n"
        + "".join(traceback.format_list(frames))
        + "".join(traceback.format_exception_only(type(exc), exc))
    )
    available = ", ".join(functions)
    if isinstance(exc, ImportError) or "__import__" in str(exc):
        text += (
            "\nHint: there is no import in this environment - it is a virtual environment, not a real OS. "
            f"The only functions callable from code are: {available}"
        )
    elif isinstance(exc, NameError):
        text += f"\nHint: undefined name. Functions callable from code: {available}"
    return text
