from pathlib import Path

import pytest

from agent_loops.bench.core.registry import LOOPS, kwargs_for, with_layers
from agent_loops.bench.tasks.format import load_tasks
from agent_loops.bench.tasks.runner import run_task_case
from tests.replay.tasks import UniversalReplay, replay_mode

SAMPLES = Path(__file__).resolve().parents[1] / "fixtures" / "samples"
CASES = load_tasks(SAMPLES / "tasks.json")
STACKS = (
    list(LOOPS)
    + [n for n in with_layers(LOOPS, ["todo"])[0]]
    + [n for n in with_layers(LOOPS, ["verifier"])[0]]
)
REGISTRY = {
    **LOOPS,
    **with_layers(LOOPS, ["todo"])[0],
    **with_layers(LOOPS, ["verifier"])[0],
}


def no_answer_channel(stack: str) -> bool:
    pieces = stack.split("+")
    return "single_call" in pieces and "planner" not in pieces


def _is_query(case) -> bool:
    return bool((case.get("expect") or {}).get("answer_contains"))


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
@pytest.mark.parametrize("stack", STACKS)
def test_ground_truth_replay_is_valid_on_every_stack(stack, case):
    module = REGISTRY[stack]
    mode = replay_mode(stack)
    result = run_task_case(
        case,
        module,
        lambda tools: UniversalReplay(case, mode),
        SAMPLES,
        loop_kwargs=kwargs_for(stack),
    )
    assert result.n_tool_calls == result.n_recorded_tool_calls, (
        f"{stack}: executed {result.n_tool_calls} != recorded {result.n_recorded_tool_calls}"
    )
    if _is_query(case) and no_answer_channel(stack):
        assert result.valid is False and str(result.error).startswith(
            "answer_missing:"
        ), f"{stack} / {case['id']}: {result.error}"
        return
    assert result.valid is True, (
        f"{stack} / {case['id']}: {result.error} | terminated_by={result.terminated_by}"
    )


GENERIC = Path("data/tasks/generic_v1")
GENERIC_CASES = (
    load_tasks(GENERIC / "tasks.json") if (GENERIC / "tasks.json").exists() else []
)


@pytest.mark.slow
@pytest.mark.skipif(
    not GENERIC_CASES, reason="generic_v1 not generated; see scripts/gen_tasks.py"
)
@pytest.mark.parametrize("stack", list(LOOPS))
def test_generic_v1_replays_valid_on_every_core_stack(stack):
    module = LOOPS[stack]
    mode = replay_mode(stack)
    failures = []
    for case in GENERIC_CASES:
        result = run_task_case(
            case,
            module,
            lambda tools, c=case: UniversalReplay(c, mode),
            GENERIC,
            loop_kwargs=kwargs_for(stack),
        )
        if result.n_tool_calls != result.n_recorded_tool_calls:
            failures.append(
                f"{case['id']}: executed {result.n_tool_calls} != recorded {result.n_recorded_tool_calls}"
            )
        elif _is_query(case) and no_answer_channel(stack):
            if result.valid or not str(result.error).startswith("answer_missing:"):
                failures.append(
                    f"{case['id']}: expected structural answer_missing, got {result.error}"
                )
        elif not result.valid:
            failures.append(f"{case['id']}: {result.error} | {result.terminated_by}")
    assert not failures, f"{stack}: {len(failures)} failures\n" + "\n".join(
        failures[:10]
    )
