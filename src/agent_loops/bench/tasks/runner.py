from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_loops.bench.core.registry import Stack
from agent_loops.bench.core.runner import Budgets, CaseResult, Runner
from agent_loops.bench.tasks.score import final_answer
from agent_loops.bench.tasks.track import TaskTrack

__all__ = ["final_answer", "run_task_case"]


def run_task_case(
    case: dict[str, Any],
    loop_module: Any,
    llm_factory: Any,
    base: Path | str,
    loop_kwargs: dict[str, Any] | None = None,
    n_turns: int | None = None,
    code_timeout: float = 5.0,
    bash_timeout: float = 10.0,
    trace_sink: Any = None,
) -> CaseResult:
    stack = Stack(
        name=loop_module.NAME, run=loop_module.run, kwargs=dict(loop_kwargs or {})
    )
    runner = Runner(
        TaskTrack(Path(base)),
        llm_factory,
        Budgets(code_timeout=code_timeout, bash_timeout=bash_timeout),
        trace_sink=trace_sink,
    )
    return runner.run_case(case, stack, n_turns=n_turns)
