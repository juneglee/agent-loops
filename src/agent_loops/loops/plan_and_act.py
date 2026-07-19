"""Plan-and-Act (arXiv:2503.09572, https://github.com/SqueezeAILab/plan-and-act).

Source: arXiv:2503.09572 (ICML 2025), authors' code
https://github.com/SqueezeAILab/plan-and-act (run_plan_and_act_with_replanning.py,
plan_and_act/cot/inference/dynamic_plan.py, dynamic_act.py).
Claim: WebArena-Lite 57.58%, WebVoyager 81.36%.

Loop (as in the authors' code):
  1. Planner writes a natural-language high-level plan (dynamic_plan.py always returns
     a plan; it has no termination authority).
  2. Executor receives the task, the WHOLE current plan and every previous round
     (action + observation), decides for itself which step it is on, and emits ONE
     action per round (dynamic_act.py:43-51, 228-246).
  3. After every executor action the Planner replans from the task, the LATEST plan only
     and the list of actions taken so far (dynamic_plan.py:146-204,
     run_plan_and_act_with_replanning.py:697-712). Earlier plans are not accumulated.
  4. The episode ends only when the Executor exits (`exit(message=...)`) or on budget:
     `max_steps=30`, early stop after 3 consecutive unparseable executor responses and
     after 5 consecutive identical actions (run_plan_and_act_with_replanning.py:152-170,
     317-360). The two early stops are labelled `parse_fail` and `max_steps`.

Literature verdict: on-device (triangle) - LLM calls scale with the number of actions
(one plan + one act per action). That property is kept so that measurements can be
interpreted against it.

Adaptations (deliberate, keep them in mind when reading results):
  * The Executor acts through native tool calls instead of the `do(action=...)` text
    grammar; the Planner still answers on the text channel (`want="text"`).
  * The Executor's `exit(...)` is read as a `Final:` completion marker
    (`response_is_complete`). A `Final:` from the Planner is NOT an exit: the Planner
    returned no readable plan, which is `parse_fail`.
  * The Planner is prompted, not the trained one from the paper; only the loop structure
    is reproduced. Prompts are condensed versions of the authors' prompts.
  * The environment has no "current HTML state"; the observation of each action is
    passed instead, to both the Executor and the Replanner.
  * Only the first tool call of an executor response is executed (one action per
    round); any extra calls are recorded as `ignored` in the round and never run.
"""

from __future__ import annotations

import re
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, calls_of, response_is_complete

NAME = "plan_and_act"
EXECUTOR = "plan_and_act_executor"
PARSING_FAILURE_TH = 3
REPEATING_ACTION_TH = 5

_STEP = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$", re.MULTILINE)


def parse_steps(text: str) -> list[str]:
    return [step for step in _STEP.findall(text or "") if step]


def _early_stop(actions: list[str]) -> str | None:
    if len(actions) >= PARSING_FAILURE_TH and not any(actions[-PARSING_FAILURE_TH:]):
        return "parse_fail"
    last = actions[-1] if actions else ""
    if (
        last
        and len(actions) >= REPEATING_ACTION_TH
        and all(a == last for a in actions[-REPEATING_ACTION_TH:])
    ):
        return "max_steps"
    return None


def run(
    task: str, env: Any, llm: Any, max_steps: int = 30, history: list | None = None
) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])
    rounds: list[dict[str, Any]] = []
    actions: list[str] = []
    plan_text = ""

    for _ in range(max_steps):
        planner_history = (
            [{"role": "assistant", "content": {"plan": plan_text}}] if plan_text else []
        ) + rounds
        plan_response = llm(
            messages=build_messages(NAME, task, planner_history, prior=prior),
            want="text",
        )
        trace.steps.append(Step(llm_response=plan_response))
        plan_text = plan_response.get("text", "")
        if not parse_steps(plan_text):
            return trace.stop("parse_fail")

        executor_history = [
            {"role": "assistant", "content": {"plan": plan_text}}
        ] + rounds
        exec_response = llm(
            messages=build_messages(EXECUTOR, task, executor_history, prior=prior)
        )
        calls = calls_of(exec_response)
        if not calls:
            trace.steps.append(Step(llm_response=exec_response))
            if response_is_complete(exec_response):
                return trace.stop("success")
            rounds.append(
                {"role": "tool", "content": {"executor": exec_response.get("text", "")}}
            )
            actions.append("")
        else:
            name, arguments = calls[0]
            observation = env.execute(name, arguments)
            trace.steps.append(
                Step(
                    llm_response=exec_response,
                    tool_name=name,
                    tool_arguments=arguments,
                    observation=observation,
                )
            )
            result: dict[str, Any] = {
                "tool": name,
                "arguments": arguments,
                "result": observation,
            }
            if len(calls) > 1:
                result["ignored"] = [{"tool": n, "arguments": a} for n, a in calls[1:]]
            rounds.append({"role": "tool", "content": result})
            actions.append(f"{name}({sorted(arguments.items())})")

        stop = _early_stop(actions)
        if stop:
            return trace.stop(stop)

    return trace.stop("max_steps")
