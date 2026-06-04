from __future__ import annotations

from typing import Any

from agent_loops.loops.base import Trace


def turn_messages(task: str, trace: Trace) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    steps = trace.steps
    i = 0
    while i < len(steps):
        step = steps[i]
        if step.tool_name is None:
            text = str((step.llm_response or {}).get("text", "") or "")
            if text.strip():
                messages.append({"role": "assistant", "content": {"text": text}})
            i += 1
            continue
        group = [step]
        i += 1
        while (
            i < len(steps)
            and steps[i].tool_name is not None
            and steps[i].llm_response is None
        ):
            group.append(steps[i])
            i += 1
        messages.append(
            {
                "role": "assistant",
                "content": {
                    "tool_calls": [
                        {"name": g.tool_name, "arguments": g.tool_arguments}
                        for g in group
                    ]
                },
            }
        )
        for g in group:
            if g.observation is not None:
                messages.append({"role": "tool", "content": g.observation})
    return messages
