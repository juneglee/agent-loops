"""ReAct (arXiv:2210.03629, https://github.com/ysymyth/ReAct).

Repeats Thought -> Action -> Observation. In the authors' ALFWorld loop a
`think:` action gets the observation "OK." and changes nothing in the
environment; observations accumulate in the transcript; the episode ends on an
explicit finish (`Finish[answer]` in HotpotQA) or the step budget (7 / 49).
The paper's text actions (`Action: tool[...]`) are replaced by native tool
calls; those have no finish action, so a `Final:` marker plays that role, and
text without the marker is a thought that keeps the loop going. `max_steps`
bounds the loop like the authors' range. The instruction is zero-shot and
thoughts are optional, unlike the authors' few-shot Thought/Act/Obs exemplars.
"""

from __future__ import annotations

from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, calls_of, response_is_complete

NAME = "react"


def run(
    task: str, env: Any, llm: Any, max_steps: int = 10, history: list | None = None
) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])
    history: list[dict[str, Any]] = []

    for _ in range(max_steps):
        response = llm(messages=build_messages(NAME, task, history, prior=prior))
        tool_calls = response.get("tool_calls")

        if not tool_calls:
            trace.steps.append(Step(llm_response=response))
            if response_is_complete(response):
                trace.terminated_by = "success"
                return trace
            history.append({"role": "assistant", "content": response})
            continue

        history.append({"role": "assistant", "content": response})
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
            history.append({"role": "tool", "content": observation})

    trace.terminated_by = "max_steps"
    return trace
