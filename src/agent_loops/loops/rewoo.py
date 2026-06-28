"""ReWOO (arXiv:2305.18323, https://github.com/billxbf/ReWOO).

Reference implementation: https://github.com/billxbf/ReWOO (``algos/PWS.py``,
``prompts/solver.py``). The loop is Planner -> Worker -> Solver: the planner emits
the whole plan at once as ``#E<n> = Tool[input]`` lines with ``#E`` variables
expressing dependencies, the worker executes the plan without any LLM call
(substituting earlier evidence into later inputs), and the solver is called once
with the task plus the ordered worker log to produce the final answer. LLM calls
therefore stay flat at two regardless of the number of tools, which is the basis
of the paper's token-efficiency claim. No replanning, no observation feedback and
no failure recovery are added: ReWOO is open-loop by design.

Adaptations relative to the authors' code:
  - Plan syntax is ``#E1 = tool[key=value, ...]`` because our file-management
    tools take several typed arguments, while the paper assumes a single string
    input per tool. Argument parsing is shared with the other planning loops
    (``loops.planargs``) so typing is preserved; ``(...)`` is accepted as well.
  - Tool schemas are supplied natively by the LLM client; the planner instruction
    lives in ``bench.prompts`` and both calls use ``want="text"`` so the runtime
    does not intercept the plan with a native tool call.
  - Zero-shot: the authors' few-shot planner examples are not used.
  - The solver receives, as the authors do (``PWS.py`` worker_log), ordered
    ``Plan: ...`` / ``Evidence: ...`` pairs wrapped by ``SOLVER_PREFIX`` and
    ``SOLVER_SUFFIX`` (English renderings of ``prompts/solver.py``). Each Plan line
    carries the planner's own ``Plan:`` text when present and the resolved
    ``tool[args]`` call. A failed tool stores its error text as the evidence
    (the authors store ``"No evidence found"`` or the raw output), so the solver
    sees the failure instead of an empty string.
"""

from __future__ import annotations

import re
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, classify_stop
from agent_loops.loops.planargs import extract_bracketed, parse_args, substitute

NAME = "rewoo"

SOLVER_PREFIX = (
    "Solve the following task or problem. To assist you, we provide some plans "
    "and corresponding evidences that might be helpful. Notice that some of these "
    "information contain noise so you should trust them with caution.\n\n"
)
SOLVER_SUFFIX = (
    "\nNow begin to solve the task or problem. Respond with the answer directly "
    "with no extra words.\n\n"
)
NO_EVIDENCE = "No evidence found"

_PLAN_HEAD = re.compile(r"#E(\d+)\s*=\s*(\w+)\s*[\[(]")
_PLAN_LINE = re.compile(r"^\s*Plan:\s*(.*)$", re.MULTILINE)


def parse_plan_line(text: str) -> list[tuple[str, str, dict]]:
    return [(var, tool, args) for var, tool, args, _ in parse_plan(text)]


def parse_plan(text: str) -> list[tuple[str, str, dict, str]]:
    plan: list[tuple[str, str, dict, str]] = []
    pos = 0
    text = text or ""

    while (match := _PLAN_HEAD.search(text, pos)) is not None:
        found = extract_bracketed(text, match.end() - 1)
        if found is None:
            break
        argstr, end = found
        described = _PLAN_LINE.findall(text[pos : match.start()])
        description = described[-1].strip() if described else ""
        plan.append(
            (f"#E{match.group(1)}", match.group(2), parse_args(argstr), description)
        )
        pos = end + 1

    return plan


def _render_call(tool: str, args: dict[str, Any]) -> str:
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{tool}[{rendered}]"


def _evidence_of(observation: dict[str, Any]) -> Any:
    if observation.get("ok", True):
        return observation.get("output", "")
    return observation.get("error") or NO_EVIDENCE


def build_solver_message(task: str, log: list[tuple[str, str, Any]]) -> str:
    body = ""
    for description, call, evidence in log:
        plan = f"{description} {call}".strip() if description else call
        body += f"Plan: {plan}\nEvidence: {evidence}\n"
    return f"{SOLVER_PREFIX}{body}\nTask: {task}\n{SOLVER_SUFFIX}"


def run(task: str, env: Any, llm: Any, history: list | None = None) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])

    plan_response = llm(messages=build_messages(NAME, task, prior=prior), want="text")
    plan = parse_plan(plan_response.get("text", ""))
    trace.steps.append(Step(llm_response=plan_response))

    if not plan:
        return trace.stop(classify_stop(plan_response.get("text"), did_work=False))

    results: dict[str, Any] = {}
    log: list[tuple[str, str, Any]] = []
    for var, tool, args, description in plan:
        resolved = {k: substitute(v, results) for k, v in args.items()}
        observation = env.execute(tool, resolved)
        trace.steps.append(
            Step(
                llm_response=None,
                tool_name=tool,
                tool_arguments=resolved,
                observation=observation,
            )
        )
        results[var] = _evidence_of(observation)
        log.append(
            (description, f"{var} = {_render_call(tool, resolved)}", results[var])
        )

    solve_response = llm(
        messages=build_messages(
            NAME,
            task,
            [{"role": "user", "content": build_solver_message(task, log)}],
            prior=prior,
        ),
        want="text",
    )
    trace.steps.append(Step(llm_response=solve_response))
    return trace.stop("success")
