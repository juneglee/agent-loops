from __future__ import annotations

import re
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import Step, Trace, classify_stop

_SUBTASK = re.compile(r"^\s*[-*]\s*(.+?)\s*$", re.MULTILINE)


def _worker_summary(sub_trace: Trace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in sub_trace.steps:
        if step.tool_name is not None:
            out.append(
                {
                    "tool": step.tool_name,
                    "arguments": step.tool_arguments,
                    "output": (step.observation or {}).get("output"),
                    "ok": (step.observation or {}).get("ok"),
                }
            )
        elif step.llm_response is not None:
            text = str(step.llm_response.get("text", "") or "")
            if text.strip():
                out.append({"answer": text})
    return out


def hierarchical(
    worker: Any, max_rounds: int = 5, worker_kwargs: dict[str, Any] | None = None
):

    def run(task: str, env: Any, llm: Any, history: list | None = None) -> Trace:
        trace = Trace(task=task, loop=run.NAME)
        prior = list(history or [])
        past: list[dict[str, Any]] = []
        workers_ok = True

        for _ in range(max_rounds):
            response = llm(
                messages=build_messages("hierarchical", task, past, prior=prior),
                want="text",
            )
            trace.steps.append(Step(llm_response=response))
            subtasks = _SUBTASK.findall(response.get("text", ""))

            if not subtasks:
                trace.stop(classify_stop(response.get("text"), trace.n_tool_calls > 0))
                trace.parse_ok = trace.parse_ok and workers_ok
                return trace

            past.append(
                {"role": "assistant", "content": {"plan": response.get("text", "")}}
            )
            for subtask in subtasks:
                sub_trace = worker.run(
                    task=subtask,
                    env=env,
                    llm=llm,
                    history=prior,
                    **(worker_kwargs or {}),
                )
                trace.steps.extend(sub_trace.steps)
                workers_ok = workers_ok and sub_trace.parse_ok
                past.append(
                    {
                        "role": "tool",
                        "content": {
                            "subtask": subtask,
                            "result": _worker_summary(sub_trace),
                        },
                    }
                )

        trace.stop("max_steps")
        trace.parse_ok = trace.parse_ok and workers_ok
        return trace

    run.NAME = f"planner+{worker.NAME}"
    return run
