"""Fixed pipeline (Agentless, arXiv:2407.01489, https://github.com/OpenAutoCoder/Agentless).

Three fixed stages and no loop. The authors' localize -> repair -> validate is
specific to code repair, so here the stages are locate -> act -> verify for
file tasks, one LLM call each, never more than three. This is a weaker
abridgement of Agentless: the authors sample dozens of candidates per stage
(4 localizations, 40 patches, 40 reproduction tests), pick by execution-based
majority vote and retry inside a stage at a higher temperature. Only the
"three stages, no loop" structure is kept, so this baseline stands for a
sequential three-call floor rather than for the Agentless result itself.
"""

from __future__ import annotations

from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, calls_of

NAME = "fixed_pipeline"
STAGES = ("locate", "act", "verify")


def run(task: str, env: Any, llm: Any, history: list | None = None) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])
    history: list[Any] = []

    for stage in STAGES:
        response = llm(
            messages=build_messages(
                NAME,
                task,
                [{"role": "system", "content": f"stage: {stage}"}, *history],
                prior=prior,
            )
        )
        tool_calls = response.get("tool_calls")

        if not tool_calls:
            trace.steps.append(Step(llm_response=response, stage=stage))
            history.append({"role": "assistant", "content": response})
            continue

        for i, (name, arguments) in enumerate(calls_of(response)):
            observation = env.execute(name, arguments)
            trace.steps.append(
                Step(
                    llm_response=response if i == 0 else None,
                    tool_name=name,
                    tool_arguments=arguments,
                    observation=observation,
                    stage=stage,
                )
            )
            history.append({"role": "tool", "content": observation})

    return trace.stop("success" if trace.n_tool_calls else "no_action")
