from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agent_loops.bench.core.codeact_setup import prepare
from agent_loops.bench.core.history import turn_messages

TERMINATION_REASONS = frozenset(
    {
        "success",
        "no_plan",
        "parse_fail",
        "no_action",
        "max_steps",
        "max_depth",
        "max_trials",
        "give_up",
    }
)


@dataclass
class CaseResult:
    case_id: str
    loop: str
    run_index: int = 0
    valid: bool = False
    error: str | None = None
    n_turns: int = 0
    n_llm_calls: int = 0
    n_steps: int = 0
    n_tool_calls: int = 0
    n_recorded_tool_calls: int = 0
    n_duplicate_tool_calls: int = 0
    n_success_turns_without_tools: int = 0
    parse_failures: int = 0
    loop_parse_flags: int = 0
    quiet_failures: int = 0
    truncations: int = 0
    terminated_by: list[str] = field(default_factory=list)
    seconds: float = 0.0
    tps: float = 0.0

    @property
    def parse_rate(self) -> float:
        if not self.n_llm_calls:
            return 0.0
        return 1.0 - self.parse_failures / self.n_llm_calls


def silent_success(trace: Any) -> bool:
    return trace.terminated_by == "success" and trace.n_tool_calls == 0


def count_duplicate_calls(calls: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for call in calls:
        key = json.dumps(
            {"name": call.get("name"), "arguments": call.get("arguments")},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def run_turns(
    env: Any,
    llm: Any,
    loop_module: Any,
    tasks: list[str],
    loop_kwargs: dict[str, Any] | None,
    result: CaseResult,
    trace_sink: Any = None,
) -> list[Any]:
    turn_traces: list[Any] = []
    prior: list[dict[str, Any]] = []
    for turn_index, task in enumerate(tasks):
        trace = loop_module.run(
            task=task, env=env, llm=llm, history=list(prior), **(loop_kwargs or {})
        )
        turn_traces.append(trace)
        prior.extend(turn_messages(task, trace))
        result.n_llm_calls += trace.n_llm_calls
        result.n_steps += len(trace.steps)
        result.n_recorded_tool_calls += trace.n_tool_calls
        result.terminated_by.append(trace.terminated_by)
        if not trace.parse_ok:
            result.loop_parse_flags += 1
        if silent_success(trace):
            result.n_success_turns_without_tools += 1
        if trace_sink is not None:
            trace_sink(turn_index, task, trace)
    return turn_traces


def _tps_of(turn_traces: list[Any]) -> float:
    values = []
    for trace in turn_traces:
        for step in trace.steps:
            raw = (step.llm_response or {}).get("raw") if step.llm_response else None
            tps = (
                ((raw or {}).get("timings") or {}).get("predicted_per_second")
                if isinstance(raw, dict)
                else None
            )
            if isinstance(tps, (int, float)):
                values.append(float(tps))
    return sum(values) / len(values) if values else 0.0


def account(result: CaseResult, env: Any, llm: Any, turn_traces: list[Any]) -> None:
    result.n_tool_calls = len(env.calls)
    result.n_duplicate_tool_calls = count_duplicate_calls(env.calls)
    result.parse_failures = getattr(llm, "parse_failures", 0)
    result.quiet_failures = getattr(llm, "quiet_failures", 0)
    result.truncations = getattr(llm, "truncations", 0)
    result.tps = _tps_of(turn_traces)


@dataclass(frozen=True)
class Budgets:
    code_timeout: float = 5.0
    bash_timeout: float = 10.0


class Runner:
    def __init__(
        self,
        track: Any,
        llm_factory: Any,
        budgets: Budgets | None = None,
        trace_sink: Any = None,
    ) -> None:
        self.track = track
        self.llm_factory = llm_factory
        self.budgets = budgets or Budgets()
        self.trace_sink = trace_sink

    def run_case(
        self, case: dict[str, Any], stack: Any, n_turns: int | None = None
    ) -> CaseResult:

        result = CaseResult(case_id=case["id"], loop=stack.name)
        started = time.time()
        env = self.track.make_env(case, self.budgets)
        try:
            tools, extra_system = prepare(env, self.track.tools_for(case), stack.name)
            llm = self.llm_factory(tools)
            if extra_system is not None and hasattr(llm, "extra_system"):
                llm.extra_system = extra_system
            tasks = self.track.turns_of(case)
            if n_turns:
                tasks = tasks[:n_turns]
            result.n_turns = len(tasks)
            turn_traces = run_turns(
                env, llm, stack, tasks, stack.kwargs, result, self.trace_sink
            )
            account(result, env, llm, turn_traces)
            try:
                result.valid, result.error = self.track.score(case, env, turn_traces)
            except Exception as exc:  # noqa: BLE001
                result.valid = False
                result.error = f"scorer:{type(exc).__name__}: {exc}"[:200]
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        result.seconds = time.time() - started
        return result


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}
    llm_calls = sum(r.n_llm_calls for r in results)
    return {
        "n_cases": n,
        "accuracy": sum(r.valid for r in results) / n,
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
        "loop_no_call_rate": (
            sum(r.loop_parse_flags for r in results) / llm_calls if llm_calls else 0.0
        ),
        "mean_llm_calls": llm_calls / n,
        "mean_steps": sum(r.n_steps for r in results) / n,
        "mean_tool_calls": sum(r.n_tool_calls for r in results) / n,
        "mean_duplicate_tool_calls": sum(r.n_duplicate_tool_calls for r in results) / n,
        "mean_success_turns_without_tools": sum(
            r.n_success_turns_without_tools for r in results
        )
        / n,
        "mean_seconds": sum(r.seconds for r in results) / n,
        "mean_tps": sum(r.tps for r in results) / n,
    }


def pass_at_k(rows: list[CaseResult]) -> dict[str, Any]:
    by_case: dict[str, list[bool]] = {}
    for r in rows:
        by_case.setdefault(r.case_id, []).append(bool(r.valid))
    k = max((len(v) for v in by_case.values()), default=1)
    n = len(by_case)
    return {
        "k": k,
        "pass_k": (sum(1 for v in by_case.values() if v and all(v)) / n) if n else 0.0,
        "accuracy": (sum(r.valid for r in rows) / len(rows)) if rows else 0.0,
    }
