from __future__ import annotations

from typing import Any

from agent_loops.bench.bfcl.adapter import calls_to_strings
from agent_loops.bench.core.registry import Stack
from agent_loops.bench.core.runner import Budgets, CaseResult, Runner


def _decoded_from_turn_traces(turn_traces: list[Any]) -> list[list[list[str]]]:
    decoded: list[list[list[str]]] = []
    for trace in turn_traces:
        steps = [
            calls_to_strings([{"name": s.tool_name, "arguments": s.tool_arguments}])
            for s in trace.steps
            if s.tool_name is not None
        ]
        decoded.append(steps)
    return decoded


def run_case(
    case: dict[str, Any],
    loop_module: Any,
    llm_factory: Any,
    category: str,
    loop_kwargs: dict[str, Any] | None = None,
    n_turns: int | None = None,
    code_timeout: float = 5.0,
) -> CaseResult:
    from agent_loops.bench.bfcl.track import BfclTrack

    stack = Stack(
        name=loop_module.NAME, run=loop_module.run, kwargs=dict(loop_kwargs or {})
    )
    runner = Runner(
        BfclTrack(category, only_classes=None),
        llm_factory,
        Budgets(code_timeout=code_timeout),
    )
    return runner.run_case(case, stack, n_turns=n_turns)
