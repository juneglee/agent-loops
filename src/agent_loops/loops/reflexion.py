"""Reflexion (arXiv:2303.11366, https://github.com/noahshinn/reflexion).

Source: arXiv:2303.11366 (Shinn et al., "Reflexion: Language Agents with Verbal
Reinforcement Learning"). Reference code: https://github.com/noahshinn/reflexion
(alfworld_runs/alfworld_trial.py, alfworld_runs/env_history.py,
alfworld_runs/generate_reflections.py, hotpotqa_runs/agents.py, hotpotqa_runs/prompts.py).
Claim: HumanEval pass@1 91% (vs. GPT-4 80% at the time).

Loop: the unit is a trial, not a step. A trial runs a ReAct-style actor; when the
trial fails the model reflects verbally on the trajectory, the reflection is appended
to an episodic memory, and the last ``memory_size`` reflections are injected into the
context of the next trial. No weight updates - reinforcement is purely linguistic.

Mirrored from the authors' code:
  - Reflection instruction (hotpotqa_runs/prompts.py REFLECT_INSTRUCTION,
    alfworld_runs/generate_reflections.py): diagnose a possible reason for failure AND
    devise a new, concise, high-level plan. ``_REFLECT_INSTRUCTION`` below keeps that
    wording.
  - Memory cap: the actor sees only the last 3 reflections
    (alfworld_trial.py ``memory[-3:]``, generate_reflections.py ``memory[-3:]``).
    ``memory_size=3`` reproduces it.
  - Injection format (hotpotqa_runs/agents.py ``format_reflections`` +
    prompts.py REFLECTION_HEADER): a header sentence followed by
    ``Reflections:\\n- ...``. ``_REFLECTION_HEADER`` keeps that wording.
  - Repeated-action exhaustion (env_history.py ``EnvironmentHistory.add`` /
    ``check_is_exhausted``, used in alfworld_trial.py): the trial is marked exhausted
    as soon as an action equals the immediately preceding action, i.e. the same
    action twice in a row (``_REPEAT_LIMIT = 2``). One LLM output is one action, so we
    compare the whole tool-call list of a response with the previous response's.
  - Lazy reflection: hotpotqa_runs/agents.py reflects at the start of the next
    ``run`` rather than right after the failure, so a failure that is never
    retried produces no reflection. We skip the reflection after the last trial for
    the same reason (one LLM call saved, no effect on accuracy).

Adaptations (declared, not in the paper or the reference code):
  - Actor terminal actions: native tool calling has no ``finish`` action, so a
    ``Final:`` declaration ends a trial as a completion claim and ``Task failed:``
    ends it as a give-up. A text step with neither marker is a thought and the
    trajectory continues.
  - Evaluator: the paper's evaluator is always an external signal (ALFWorld reward,
    HotpotQA exact match, HumanEval unit tests). Our file-management environment has
    no reward, so a trial succeeds when the model declares completion AND no tool
    call in that trial returned an error. This is weaker than any evaluator in the
    paper; it exists so that give-ups, tool errors, repeated actions and step-budget
    exhaustion trigger reflection.
  - Environment reset between trials: the paper's environments reset per episode.
    File management leaves side effects, so ``env.reset`` is called when available
    and otherwise the retry runs on top of the previous trial's residue.
  - The reflection prompt receives the failed trial transcript plus the instruction;
    the authors' few-shot reflection examples are not included.

Deliberately absent: weight updates, search/backtracking (LATS, DFSDT), intra-step
retries. ``max_trials`` is extra compute that a one-trial loop does not have, so a
fair comparison must budget at the runner level.
"""

from __future__ import annotations

import json
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import (
    Step,
    Trace,
    calls_of,
    response_gives_up,
    response_is_complete,
)

NAME = "reflexion"

_REPEAT_LIMIT = 2

_REFLECT_INSTRUCTION = (
    "You are an advanced reasoning agent that can improve based on self reflection. "
    "You will be given a previous trial in which you were given access to a tool "
    "environment and a task to complete. You were unsuccessful in completing the task "
    "either because a tool call returned an error, you gave up, you repeated the same "
    "action, or you used up your set number of steps. In a few sentences, diagnose a "
    "possible reason for failure and devise a new, concise, high level plan that aims to "
    "mitigate the same failure. Use complete sentences. Do not call any tools."
)

_REFLECTION_HEADER = (
    "You have attempted this task before and failed. The following reflection(s) give a "
    "plan to avoid failing to complete the task in the same way you did previously. Use "
    "them to improve your strategy of correctly completing the given task.\n"
)


def _format_reflections(reflections: list[str]) -> str:
    return (
        _REFLECTION_HEADER
        + "Reflections:\n- "
        + "\n- ".join(r.strip() for r in reflections)
    )


def _action_key(response: dict[str, Any]) -> str:
    return json.dumps(calls_of(response), sort_keys=True, ensure_ascii=False)


def _attempt(
    task: str,
    env: Any,
    llm: Any,
    trace: Trace,
    memory: list[str],
    max_steps: int,
    prior: list,
) -> tuple[bool, list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    if memory:
        history.append({"role": "system", "content": _format_reflections(memory)})

    had_error = False
    last_action = ""
    repeats = 0
    for _ in range(max_steps):
        response = llm(messages=build_messages(NAME, task, history, prior=prior))
        tool_calls = response.get("tool_calls")

        if not tool_calls:
            trace.steps.append(Step(llm_response=response))
            history.append({"role": "assistant", "content": response})
            if response_is_complete(response):
                return not had_error, history
            if response_gives_up(response):
                return False, history
            continue

        history.append({"role": "assistant", "content": response})
        for i, (name, arguments) in enumerate(calls_of(response)):
            observation = env.execute(name, arguments)
            had_error = had_error or not observation.get("ok", True)
            trace.steps.append(
                Step(
                    llm_response=response if i == 0 else None,
                    tool_name=name,
                    tool_arguments=arguments,
                    observation=observation,
                )
            )
            history.append({"role": "tool", "content": observation})

        action = _action_key(response)
        repeats = repeats + 1 if action == last_action else 1
        last_action = action
        if repeats >= _REPEAT_LIMIT:
            return False, history

    return False, history


def run(
    task: str,
    env: Any,
    llm: Any,
    max_trials: int = 3,
    max_steps: int = 10,
    history: list | None = None,
    memory_size: int = 3,
) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])
    memory: list[str] = []

    for trial in range(max_trials):
        if trial > 0 and hasattr(env, "reset"):
            env.reset()

        ok, transcript = _attempt(
            task,
            env,
            llm,
            trace,
            memory[-memory_size:] if memory_size > 0 else [],
            max_steps,
            prior,
        )
        if ok:
            return trace.stop("success")

        if trial < max_trials - 1:
            reflection = llm(
                messages=build_messages(
                    NAME,
                    task,
                    [
                        {"role": "tool", "content": {"failed_trial": transcript}},
                        {"role": "system", "content": _REFLECT_INSTRUCTION},
                    ],
                    prior=prior,
                ),
                want="text",
            )
            trace.steps.append(Step(llm_response=reflection))
            memory.append(reflection.get("text", ""))

    return trace.stop("max_trials")
