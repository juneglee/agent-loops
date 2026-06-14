from __future__ import annotations

from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, calls_of

NAME = "single_call"


def run(task: str, env: Any, llm: Any, history: list | None = None) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])

    response = llm(messages=build_messages(NAME, task, prior=prior))
    tool_calls = response.get("tool_calls")

    if not tool_calls:
        trace.parse_ok = False
        trace.terminated_by = "parse_fail"
        trace.steps.append(Step(llm_response=response))
        return trace

    for i, (name, arguments) in enumerate(calls_of(response)):
        observation = env.execute(name, arguments)
        trace.steps.append(
            Step(
                llm_response=response if i == 0 else None,
                tool_name=name,
                tool_arguments=arguments,
                observation=observation,
            )
        )
    return trace
