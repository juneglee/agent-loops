from __future__ import annotations

from typing import Any

from agent_loops.harness.checks.gfs import GFS_CHECKS, Check


class _VerifierEnv:
    def __init__(self, inner: Any, checks: dict[str, Check]) -> None:
        self._inner = inner
        self._checks = checks

    def execute(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        observation = self._inner.execute(name, arguments)
        if not observation.get("ok"):
            return observation
        check = self._checks.get(name)
        if check is None:
            return observation
        problem = check(self._inner, dict(arguments or {}), observation)
        if problem is None:
            return observation
        return {**observation, "ok": False, "error": f"postcondition: {problem}"}

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class _VerifierLayer:
    def __init__(self, checks: dict[str, Check]) -> None:
        self._checks = checks

    def wrap_env(self, env: Any) -> Any:
        return _VerifierEnv(env, self._checks)


def verifier(checks: dict[str, Check] | None = None):
    table = GFS_CHECKS if checks is None else checks

    def factory() -> _VerifierLayer:
        return _VerifierLayer(table)

    factory.NAME = "verifier"
    return factory
