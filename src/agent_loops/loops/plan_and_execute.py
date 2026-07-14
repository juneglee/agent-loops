"""Plan-and-Execute, after the LangGraph tutorial (commit 23961cf,
https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb).

The replanner format is taken from the tutorial: it sees the request, the
original plan and the steps done so far, and answers with either a final
response or a new plan; an empty plan ends the loop. The tutorial executes only
the first plan step through an inner ReAct agent and replans after each one,
which is the same shape as plan_and_act. This loop is a deliberate variant:
it executes the whole plan, then replans on the observations, filling the
middle of the replanning-frequency axis (none: plan_and_solve, rewoo; after the
plan: here; on failure: adapt; every action: plan_and_act). Plans use the tool
syntax `1. tool[arg=value]` instead of natural-language steps, and the budget is
`max_rounds` (rounds) rather than the tutorial's recursion limit.
"""

from __future__ import annotations

from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, classify_stop
from agent_loops.loops.plan_and_solve import parse_plan

NAME = "plan_and_execute"


def run(
    task: str, env: Any, llm: Any, max_rounds: int = 5, history: list | None = None
) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])
    history: list[dict[str, Any]] = []

    for _round in range(max_rounds):
        response = llm(
            messages=build_messages(NAME, task, history, prior=prior), want="text"
        )
        trace.steps.append(Step(llm_response=response))
        plan = parse_plan(response.get("text", ""))

        if not plan:
            return trace.stop(
                classify_stop(response.get("text"), trace.n_tool_calls > 0)
            )

        history.append(
            {"role": "assistant", "content": {"plan": response.get("text", "")}}
        )
        for tool, args in plan:
            observation = env.execute(tool, args)
            trace.steps.append(
                Step(
                    llm_response=None,
                    tool_name=tool,
                    tool_arguments=args,
                    observation=observation,
                )
            )
            history.append(
                {
                    "role": "tool",
                    "content": {
                        "step": {"tool": tool, "arguments": args},
                        "result": observation,
                    },
                }
            )

    return trace.stop("max_steps")
