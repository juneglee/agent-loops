"""CodeAct (arXiv:2402.01030, https://github.com/xingyaoww/code-act).

Source: Wang et al., "Executable Code Actions Elicit Better LLM Agents",
arXiv:2402.01030 (ICML 2024). Reference implementation: https://github.com/xingyaoww/code-act
(mint/envs/general_env.py, mint/tools/python_tool.py, scripts/eval/m3tooleval/main.py).
Claim: up to +20% success over JSON/text actions across 17 LLMs.

Loop: the action is executable Python; the observation is stdout or a traceback. One code
block can compose several tools, so a multi-step task collapses into one LLM call.
Self-debug (in the paper): after a failed execution the model sees the traceback and
rewrites the code. There is no separate self-debug budget; only ``max_steps`` bounds it.

Termination (authors): an episode ends only on an explicit answer - ``<solution>`` in MINT
(general_env.py ``handle_propose_solution``) or ``Answer:`` in M3ToolEval (tasks/base.py
``parse_generation``). Text with neither code nor the answer tag is an invalid action in
MINT and an ``invalid`` generation in M3ToolEval; both feed a reminder back and continue.
Here a response without an ``execute_code`` call is success only if it carries the ``Final:``
completion marker (``response_is_complete``); otherwise it is a thought and the loop continues,
the same rule as react.

Remaining-turn notice (authors): MINT appends the number of steps left to every observation
(general_env.py ``count_down`` -> ``StepOutput.to_str``). Each observation fed back here
carries ``"N steps remaining"``.

Adaptations, all deliberate:
  * Code travels as a native ``execute_code`` tool call instead of the fenced ``<execute>``
    / ``Action:`` text of the authors. bench/core/codeact_setup.py opens that single tool
    and lists the callable functions as signatures in the prompt (action space = code only).
  * Each execution runs in a fresh child process (bench/bfcl/env.py), so REPL variables do
    not persist between turns; the authors' IPython shell keeps state across turns. This is
    a sandbox adaptation.
  * Execution timeout 5s (``Budgets.code_timeout``) versus the authors' 30s.
  * The traceback is filtered to the model's own frames (``_model_facing_traceback``);
    the authors only redact their file path.
  * A call to any tool other than ``execute_code`` is outside the action space. It is not
    executed; the error is returned as an observation so self-debug can recover, mirroring
    MINT's ``INVALID_INPUT_MESSAGE``.

Not included (deliberate): replanning, decomposition, external verifiers.
"""

from __future__ import annotations

from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, calls_of, response_is_complete

NAME = "codeact"
CODE_TOOL = "execute_code"


def run(
    task: str, env: Any, llm: Any, max_steps: int = 6, history: list | None = None
) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])
    history: list[dict[str, Any]] = []

    for step_index in range(max_steps):
        response = llm(messages=build_messages(NAME, task, history, prior=prior))
        tool_calls = response.get("tool_calls")

        if not tool_calls:
            trace.steps.append(Step(llm_response=response))
            if response_is_complete(response):
                return trace.stop("success")
            history.append({"role": "assistant", "content": response})
            continue

        remaining = max_steps - step_index - 1
        history.append({"role": "assistant", "content": response})
        for i, (name, arguments) in enumerate(calls_of(response)):
            rejected = name != CODE_TOOL
            if rejected:
                observation = {
                    "ok": False,
                    "error": f"'{name}' is not available. Call only {CODE_TOOL}.",
                    "output": "",
                }
            else:
                observation = env.execute(name, arguments)
            trace.steps.append(
                Step(
                    llm_response=response if i == 0 else None,
                    observation=observation,
                    stage="rejected" if rejected else "code",
                )
            )
            for inner in observation.get("calls") or []:
                trace.steps.append(
                    Step(
                        llm_response=None,
                        tool_name=inner["name"],
                        tool_arguments=dict(inner.get("arguments", {})),
                        observation=observation,
                    )
                )
            history.append(
                {
                    "role": "tool",
                    "content": {
                        **observation,
                        "notice": f"{remaining} steps remaining",
                    },
                }
            )

    return trace.stop("max_steps")
