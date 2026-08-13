from __future__ import annotations

import re
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.compose.hierarchical import hierarchical
from agent_loops.loops.base import Step, Trace

_WORD = re.compile(r"[A-Za-z]+")


def _verdict(text: str) -> str | None:
    m = _WORD.search(text or "")
    word = m.group(0).lower() if m else ""
    return word if word in ("simple", "complex") else None


def routed(worker: Any, max_rounds: int = 5, worker_kwargs: dict | None = None):

    def run(task: str, env: Any, llm: Any, history: list | None = None) -> Trace:
        trace = Trace(task=task, loop=run.NAME)
        prior = list(history or [])

        response = llm(
            messages=build_messages("routed", task, prior=prior), want="text"
        )
        trace.steps.append(Step(llm_response=response))
        verdict = _verdict(response.get("text", ""))
        if verdict is None:
            return trace.stop("parse_fail")

        if verdict == "simple":
            sub = worker.run(
                task=task, env=env, llm=llm, history=prior, **(worker_kwargs or {})
            )
        else:
            sub = hierarchical(
                worker, max_rounds=max_rounds, worker_kwargs=worker_kwargs
            )(task=task, env=env, llm=llm, history=prior)
        trace.steps.extend(sub.steps)
        trace.stop(sub.terminated_by)
        trace.parse_ok = trace.parse_ok and sub.parse_ok
        return trace

    run.NAME = f"routed+{worker.NAME}"
    return run
