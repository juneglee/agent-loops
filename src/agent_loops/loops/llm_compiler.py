"""LLMCompiler (arXiv:2312.04511, https://github.com/SqueezeAILab/LLMCompiler).

Reference implementation: https://github.com/SqueezeAILab/LLMCompiler
(``src/llm_compiler/llm_compiler.py``, ``planner.py``, ``task_fetching_unit.py``,
``output_parser.py`` and the ``configs/*`` joiner prompts). The loop has three
components plus a joiner: the Planner emits a DAG of function calls at once with
``$n`` placeholders as dependency edges, the Task Fetching Unit dispatches every
task whose dependencies are resolved and substitutes the placeholders with the
real outputs, the Executor runs those tasks in parallel, and after the whole
plan has run the joiner is a separate LLM call (``LLMCompiler.join``) that reads
the plan with its observations and answers either ``Finish(answer)`` or
``Replan`` together with a ``Thought``. On ``Replan`` the planner is called again
with the previous plan, its observations and the joiner's thought as context
(``_generate_context_for_replanner``); the authors' configs all set
``max_replans = 1``.

Adaptations relative to the authors' code:
  - Plan syntax is ``#E1 = tool[key=value, ...]`` and is shared with ReWOO
    (``loops.rewoo.parse_plan_line``, ``loops.planargs``) so that planning loops
    are compared on structure rather than on parser quality; ``#E`` references
    play the role of the authors' ``$n`` placeholders and define the DAG edges.
    Forward or self references are rejected as an unreadable plan.
  - No real concurrency: on-device single process. Tasks are levelled into
    dependency waves (``_dependency_waves``) and executed wave by wave in order;
    ``Step.stage`` records the wave so that the parallelisable share can be
    reported, but the latency gain of the paper is not reproduced. The streaming
    planner (``TaskFetchingUnit.aschedule``) is likewise not reproduced.
  - The joiner speaks on the text channel: ``Final: <answer>`` stands for
    ``Finish(answer)`` (read with ``response_is_complete``) and ``Replan: <reason>``
    for ``Replan`` with its ``Thought``. Anything else is an unreadable joiner
    output and stops the loop with ``parse_fail``. The joiner instruction is
    ``JOINER_INSTRUCTION`` below; the authors' few-shot joiner examples are not used.
  - ``max_replans`` counts replans, so the loop runs ``1 + max_replans`` rounds.
    The authors switch to ``joinner_prompt_final`` on the last iteration and
    force ``is_replan = False``; here a ``Replan`` verdict on the last round stops
    the loop with ``max_steps`` instead of forcing an answer.
  - Failure handling: the authors' ``Task.__call__`` does not catch tool
    exceptions. Here a failed tool stores its error as the observation, tasks
    that reference a failed variable are skipped instead of executed with the
    error as an argument, and the skip is reported to the joiner and the
    replanner. A variable re-executed successfully by a replan is un-failed.
  - Zero-shot planner: the instruction lives in ``bench.prompts`` and the
    authors' few-shot planner examples are not used.
"""

from __future__ import annotations

import re
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, classify_stop, response_is_complete
from agent_loops.loops.planargs import substitute
from agent_loops.loops.rewoo import parse_plan_line

NAME = "llm_compiler"

JOINER_INSTRUCTION = (
    "The plan above has been executed and each action is followed by its "
    "Observation. Decide what to do next and answer with exactly one of:\n"
    "  Final: <answer>   if the observations are enough to answer the request.\n"
    "  Replan: <reason>  if more actions are needed; say what is still missing "
    "so that the next plan can continue from these results.\n"
    "Do not write anything else."
)

_VAR_REF = re.compile(r"#E(\d+)")
_REPLAN = re.compile(r"^\s*replan\s*[:：]\s*(.*)$", re.IGNORECASE)


def _refs_in(args: dict[str, Any]) -> set[str]:
    return {
        f"#E{m.group(1)}"
        for value in args.values()
        for m in _VAR_REF.finditer(str(value))
    }


def _dependency_waves(
    plan: list[tuple[str, str, dict]],
    known: set[str] | frozenset[str] = frozenset(),
) -> list[list[tuple[str, str, dict]]] | None:
    level_of: dict[str, int] = {var: 0 for var in known}
    waves: list[list[tuple[str, str, dict]]] = []
    for var, tool, args in plan:
        refs = _refs_in(args)
        if any(ref not in level_of for ref in refs):
            return None
        level = 1 + max((level_of[ref] for ref in refs), default=0)
        level_of[var] = level
        while len(waves) < level:
            waves.append([])
        waves[level - 1].append((var, tool, args))
    return waves


def _render_call(var: str, tool: str, args: dict[str, Any]) -> str:
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{var} = {tool}[{rendered}]"


def _scratchpad(executed: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for record in executed:
        lines.append(
            _render_call(record["variable"], record["tool"], record["arguments"])
        )
        if "skipped" in record:
            lines.append(f"Observation: skipped ({record['skipped']})")
        else:
            observation = record["observation"]
            if observation.get("ok", True):
                lines.append(f"Observation: {observation.get('output')}")
            else:
                lines.append(f"Observation: error: {observation.get('error')}")
    return "\n".join(lines)


def replan_thought(response: dict[str, Any]) -> str | None:
    for line in str(response.get("text") or "").splitlines():
        match = _REPLAN.match(line.strip().strip("*_`> "))
        if match:
            return match.group(1).strip()
    return None


def run(
    task: str, env: Any, llm: Any, max_replans: int = 1, history: list | None = None
) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])
    results: dict[str, Any] = {}
    failed_vars: set[str] = set()
    history: list[dict[str, Any]] = []
    scratchpad: list[dict[str, Any]] = []

    for _round in range(1 + max_replans):
        response = llm(
            messages=build_messages(NAME, task, history, prior=prior), want="text"
        )
        trace.steps.append(Step(llm_response=response))
        plan = parse_plan_line(response.get("text", ""))

        if not plan:
            return trace.stop(
                classify_stop(response.get("text"), trace.n_tool_calls > 0)
            )

        waves = _dependency_waves(plan, known=set(results))
        if waves is None:
            return trace.stop("parse_fail")

        executed: list[dict[str, Any]] = []
        for wave_index, wave in enumerate(waves, start=1):
            for var, tool, args in wave:
                broken = [ref for ref in _refs_in(args) if ref in failed_vars]
                if broken:
                    failed_vars.add(var)
                    executed.append(
                        {
                            "variable": var,
                            "tool": tool,
                            "arguments": args,
                            "skipped": f"dependency failed: {', '.join(broken)}",
                        }
                    )
                    continue
                resolved = {
                    key: substitute(value, results) for key, value in args.items()
                }
                observation = env.execute(tool, resolved)
                trace.steps.append(
                    Step(
                        llm_response=None,
                        tool_name=tool,
                        tool_arguments=resolved,
                        observation=observation,
                        stage=f"wave{wave_index}",
                    )
                )
                if observation.get("ok", True):
                    results[var] = observation.get("output")
                    failed_vars.discard(var)
                else:
                    results[var] = {"error": observation.get("error")}
                    failed_vars.add(var)
                executed.append(
                    {
                        "variable": var,
                        "tool": tool,
                        "arguments": resolved,
                        "observation": observation,
                    }
                )
        scratchpad.extend(executed)

        joiner_message = {
            "role": "user",
            "content": f"{_scratchpad(scratchpad)}\n\n{JOINER_INSTRUCTION}",
        }
        verdict = llm(
            messages=build_messages(
                NAME, task, [*history, joiner_message], prior=prior
            ),
            want="text",
        )
        trace.steps.append(Step(llm_response=verdict))
        if response_is_complete(verdict):
            return trace.stop("success")
        thought = replan_thought(verdict)
        if thought is None:
            return trace.stop("parse_fail")
        history.append({"role": "assistant", "content": response})
        history.append(
            {"role": "tool", "content": {"executed": executed, "thought": thought}}
        )

    return trace.stop("max_steps")
