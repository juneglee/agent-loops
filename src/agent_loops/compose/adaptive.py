from __future__ import annotations

import re
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace

_SUBTASK = re.compile(r"^\s*[-*]\s*(.+?)\s*$", re.MULTILINE)


def _attempt(
    worker: Any,
    task: str,
    env: Any,
    llm: Any,
    trace: Trace,
    prior: list,
    worker_kwargs: dict | None,
) -> tuple[bool, dict | None, bool]:
    sub = worker.run(
        task=task, env=env, llm=llm, history=list(prior), **(worker_kwargs or {})
    )
    trace.steps.extend(sub.steps)
    failed = next(
        (
            s.observation
            for s in sub.steps
            if s.observation is not None and not s.observation.get("ok")
        ),
        None,
    )
    return sub.terminated_by == "success" and failed is None, failed, sub.parse_ok


def _decompose(
    task: str, llm: Any, trace: Trace, observation: dict | None, prior: list
) -> list[str]:
    history: list[dict[str, Any]] = []
    if observation is not None:
        history.append({"role": "tool", "content": observation})
    response = llm(
        messages=build_messages("adaptive", task, history, prior=prior), want="text"
    )
    trace.steps.append(Step(llm_response=response))
    return _SUBTASK.findall(response.get("text", ""))


def _solve(
    worker: Any,
    task: str,
    env: Any,
    llm: Any,
    trace: Trace,
    depth: int,
    max_depth: int,
    prior: list,
    worker_kwargs: dict | None,
    flags: list[str],
) -> str:
    ok, failed, sub_parse_ok = _attempt(
        worker, task, env, llm, trace, prior, worker_kwargs
    )
    if not sub_parse_ok:
        flags.append(task)
    if ok:
        return "success"
    if depth >= max_depth:
        return "max_depth"

    subtasks = _decompose(task, llm, trace, failed, prior)
    if not subtasks:
        return "parse_fail"

    for subtask in subtasks:
        reason = _solve(
            worker,
            subtask,
            env,
            llm,
            trace,
            depth + 1,
            max_depth,
            prior,
            worker_kwargs,
            flags,
        )
        if reason != "success":
            return reason
    return "success"


def adaptive(worker: Any, max_depth: int = 3, worker_kwargs: dict | None = None):

    def run(task: str, env: Any, llm: Any, history: list | None = None) -> Trace:
        trace = Trace(task=task, loop=run.NAME)
        flags: list[str] = []
        trace.stop(
            _solve(
                worker,
                task,
                env,
                llm,
                trace,
                0,
                max_depth,
                list(history or []),
                worker_kwargs,
                flags,
            )
        )
        trace.parse_ok = trace.parse_ok and not flags
        return trace

    run.NAME = f"adaptive+{worker.NAME}"
    return run
