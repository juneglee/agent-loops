"""Plan-and-Solve (arXiv:2305.04091, https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting).

The original is a zero-shot prompting trigger for reasoning tasks: "Let's first
understand the problem and devise a plan to solve the problem. Then, let's carry
out the plan to solve the problem step by step." Only that core is kept here:
one call writes the whole plan, then the plan is executed as written. No
variable references, no solver step, no replanning, so the loop costs exactly
one LLM call. Tool execution and the plan parser are this repository's
adaptation; the paper has no tools.
"""

from __future__ import annotations

import re
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, classify_stop
from agent_loops.loops.planargs import extract_bracketed, parse_args

NAME = "plan_and_solve"

_PLAN_HEAD = re.compile(r"^\s*\d+[.)]\s*(\w+)\s*[\[(]", re.MULTILINE)


def parse_plan(text: str) -> list[tuple[str, dict]]:
    plan: list[tuple[str, dict]] = []
    pos = 0
    text = text or ""

    while (match := _PLAN_HEAD.search(text, pos)) is not None:
        found = extract_bracketed(text, match.end() - 1)
        if found is None:
            break
        argstr, end = found
        plan.append((match.group(1), parse_args(argstr)))
        pos = end + 1

    return plan


def run(task: str, env: Any, llm: Any, history: list | None = None) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])

    response = llm(messages=build_messages(NAME, task, prior=prior), want="text")
    trace.steps.append(Step(llm_response=response))
    plan = parse_plan(response.get("text", ""))

    if not plan:
        return trace.stop(classify_stop(response.get("text"), did_work=False))

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

    return trace.stop("success")
