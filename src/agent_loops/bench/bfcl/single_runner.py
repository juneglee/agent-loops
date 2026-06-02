from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent_loops.bench.bfcl.single import load_single_answers, single_turn_task

CHECKER_MODEL_NAME = "gpt-4.1-2025-04-14-FC"


@dataclass
class SingleResult:
    case_id: str
    loop: str
    run_index: int = 0
    valid: bool = False
    error: str | None = None
    n_llm_calls: int = 0
    n_calls_emitted: int = 0
    parse_failures: int = 0
    quiet_failures: int = 0
    truncations: int = 0
    made_any_call: bool = False
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class NullEnv:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append({"name": name, "arguments": dict(arguments or {})})
        return {"ok": True, "error": None, "output": ""}


def calls_to_ast_output(calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [{c["name"]: dict(c.get("arguments") or {})} for c in (calls or [])]


def run_single_case(
    case: dict[str, Any],
    llm: Any,
    category: str,
    loop_module: Any = None,
    loop_kwargs: dict[str, Any] | None = None,
) -> SingleResult:
    name = getattr(loop_module, "NAME", "direct")
    result = SingleResult(case_id=case["id"], loop=name)
    started = time.time()

    env = NullEnv()
    task = single_turn_task(case)

    if loop_module is None:
        response = llm(messages=[{"role": "user", "content": task}])
        calls = response.get("tool_calls") or []
        result.n_llm_calls = 1
    else:
        trace = loop_module.run(task=task, env=env, llm=llm, **(loop_kwargs or {}))
        result.n_llm_calls = trace.n_llm_calls
        calls = env.calls

    result.n_calls_emitted = len(calls)
    result.made_any_call = bool(calls)
    result.parse_failures = getattr(llm, "parse_failures", 0)
    result.quiet_failures = getattr(llm, "quiet_failures", 0)
    result.truncations = getattr(llm, "truncations", 0)

    try:
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

        scored = ast_checker(
            func_description=case["function"],
            model_output=calls_to_ast_output(calls),
            possible_answer=load_single_answers(category)[case["id"]],
            language=Language.PYTHON,
            test_category=category,
            model_name=CHECKER_MODEL_NAME,
        )
        result.valid = bool(scored.get("valid"))
        if not result.valid:
            result.error = str(scored.get("error_type") or scored.get("error"))[:160]
    except Exception as exc:  # noqa: BLE001
        result.valid = False
        result.error = f"scorer:{type(exc).__name__}: {exc}"[:160]

    result.seconds = time.time() - started
    return result


def summarize_single(results: list[SingleResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}
    llm_calls = sum(r.n_llm_calls for r in results)
    return {
        "n_cases": n,
        "accuracy": sum(r.valid for r in results) / n,
        "call_rate": sum(r.made_any_call for r in results) / n,
        "parse_rate": (
            1.0 - sum(r.parse_failures for r in results) / llm_calls
            if llm_calls
            else 0.0
        ),
        "quiet_failure_rate": (
            sum(r.quiet_failures for r in results) / llm_calls if llm_calls else 0.0
        ),
        "truncation_rate": (
            sum(r.truncations for r in results) / llm_calls if llm_calls else 0.0
        ),
        "mean_llm_calls": llm_calls / n,
        "mean_seconds": sum(r.seconds for r in results) / n,
    }
