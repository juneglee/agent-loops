"""ADaPT (arXiv:2311.05772, https://github.com/archiki/ADaPT).

Source: arXiv:2311.05772 (NAACL 2024 findings). Reference implementation:
https://github.com/archiki/ADaPT (``run_textcraft.py``: ``plan_and_run``,
``textcraft_run``, ``plan_to_args``).

Loop: the executor tries the task as given first; only when it declares failure is
the task decomposed by a planner into subtasks joined by And/Or, and each subtask is
solved recursively (Alg. 1). Easy inputs spend no planning tokens.

Not included on purpose: self-reflection, retries, backtracking. Decomposition is not
a retry; it re-poses the failed task in a different shape.

[paper] The executor is a ReAct-style iterative loop. Success is self-declared:
        "output 'task completed' if it determines it has succeeded, otherwise output
        'task failed'". Planner emits 3-5 subtasks with a logical operator; d_max = 3.
[authors] ``textcraft_run`` success is hybrid: the executor's own ``task completed``
        / ``task failed`` declaration, OR environment reward > 0. Our environment has
        no reward channel, so only the self-declaration is used here.
[authors] State propagation: ``plan_and_run`` passes the previous subtasks' action
        checkpoint and propagated info into the next subtask's executor. Here the
        successful steps (tool, arguments, observation) of earlier subtasks in the
        same plan, plus what the parent received, are prepended to the next
        subtask executor's history as a single tool message ``{"carried": [...]}``.
[authors] Patience: ``textcraft_run`` stops the executor after ``max_patience = 8``
        consecutive non-progress actions ("Could not ..." observations or think-only
        "OK."). Here ``patience`` counts consecutive steps that are thought-only or
        whose observation has ``ok is False``; reaching it ends the trial as failure.
[authors] Planner input: ``plan_llm(plan_prompt + commands + Goal)`` - the planner
        sees only the (sub)task, never the failed trajectory. Same here.
[adaptation] Declarations are read as text markers via ``response_is_complete``
        (``Task completed``) and ``response_gives_up`` (``Task failed:``). Text that is
        neither is a thought and the trial continues.
[adaptation] Executor step budget ``max_steps`` (paper: 15-20 per benchmark); default
        matches react for on-device cost. Exhaustion counts as failure (-> decompose).
[adaptation] Global LLM call budget ``max_calls`` shared by the whole recursion; not in
        the paper. Without it Or x5 branches x depth 3 reached 1,591 calls. Exhaustion
        is recorded as ``max_steps``. At most 5 subtasks are used (paper: "typically
        3-5").
[adaptation] The planner uses the text channel (``want="text"``); the native tool
        channel would return a tool call instead of a plan.
"""

from __future__ import annotations

import re
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import (
    Step,
    Trace,
    calls_of,
    response_gives_up,
    response_is_complete,
)

NAME = "adapt"

_SUBTASK = re.compile(r"^\s*[-*•]\s+(.+?)\s*$", re.MULTILINE)
MAX_SUBTASKS = 5
_OPERATOR = re.compile(
    r"^\s*(?:조합|operator|logic)\s*[:：]\s*(and|or)\b", re.IGNORECASE | re.MULTILINE
)

_DECOMPOSE = (
    "This request cannot be completed as given. "
    "Split it into 3-5 smaller subtasks and answer only as a list in the form `- subtask`. "
    "Write each subtask concretely enough that a worker can carry it out from that line alone. "
    "End the list with one line `Operator: And` (all must succeed) or `Operator: Or` "
    "(one success is enough)."
)


def _execute(
    task: str,
    env: Any,
    llm: Any,
    trace: Trace,
    prior: list,
    max_steps: int,
    budget: dict,
    carried: list[dict],
    patience: int,
) -> tuple[bool, list[dict]]:
    history: list[dict[str, Any]] = []
    if carried:
        history.append({"role": "tool", "content": {"carried": list(carried)}})
    done: list[dict] = []
    stalled = 0
    for _ in range(max_steps):
        if budget["left"] <= 0:
            return False, done
        budget["left"] -= 1
        response = llm(messages=build_messages(NAME, task, history, prior=prior))
        tool_calls = response.get("tool_calls")
        if not tool_calls:
            trace.steps.append(Step(llm_response=response))
            history.append({"role": "assistant", "content": response})
            if response_is_complete(response):
                return True, done
            if response_gives_up(response):
                return False, done
            stalled += 1
            if stalled >= patience:
                return False, done
            continue

        history.append({"role": "assistant", "content": response})
        progressed = True
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
            done.append(
                {"tool": name, "arguments": arguments, "observation": observation}
            )
            if observation.get("ok") is False:
                progressed = False
        stalled = 0 if progressed else stalled + 1
        if stalled >= patience:
            return False, done
    return False, done


def _decompose(
    task: str, llm: Any, trace: Trace, prior: list, budget: dict
) -> tuple[list[str], str]:
    history: list[dict[str, Any]] = [{"role": "system", "content": _DECOMPOSE}]
    budget["left"] -= 1
    response = llm(
        messages=build_messages(NAME, task, history, prior=prior), want="text"
    )
    trace.steps.append(Step(llm_response=response))
    text = response.get("text", "")
    match = _OPERATOR.search(text)
    operator = match.group(1).lower() if match else "and"
    subtasks = [s for s in _SUBTASK.findall(text) if not _OPERATOR.match(s)]
    return subtasks[:MAX_SUBTASKS], operator


def _solve(
    task: str,
    env: Any,
    llm: Any,
    trace: Trace,
    depth: int,
    max_depth: int,
    prior: list,
    max_steps: int,
    budget: dict,
    carried: list[dict],
    patience: int,
) -> tuple[str, list[dict]]:
    ok, done = _execute(
        task, env, llm, trace, prior, max_steps, budget, carried, patience
    )
    if ok:
        return "success", done
    if budget["left"] <= 0:
        return "max_steps", []
    if depth >= max_depth:
        return "max_depth", []

    subtasks, operator = _decompose(task, llm, trace, prior, budget)
    if not subtasks:
        return "parse_fail", []

    gathered: list[dict] = []
    if operator == "or":
        reason = "max_depth"
        for subtask in subtasks:
            if budget["left"] <= 0:
                return "max_steps", []
            reason, done = _solve(
                subtask,
                env,
                llm,
                trace,
                depth + 1,
                max_depth,
                prior,
                max_steps,
                budget,
                carried,
                patience,
            )
            if reason == "success":
                return "success", done
        return reason, []
    for subtask in subtasks:
        if budget["left"] <= 0:
            return "max_steps", []
        reason, done = _solve(
            subtask,
            env,
            llm,
            trace,
            depth + 1,
            max_depth,
            prior,
            max_steps,
            budget,
            carried + gathered,
            patience,
        )
        if reason != "success":
            return reason, []
        gathered.extend(done)
    return "success", gathered


def run(
    task: str,
    env: Any,
    llm: Any,
    max_depth: int = 3,
    max_steps: int = 10,
    max_calls: int = 30,
    history: list | None = None,
    patience: int = 8,
) -> Trace:
    trace = Trace(task=task, loop=NAME)
    budget = {"left": max_calls}
    reason, _ = _solve(
        task,
        env,
        llm,
        trace,
        0,
        max_depth,
        list(history or []),
        max_steps,
        budget,
        [],
        patience,
    )
    return trace.stop(reason)
