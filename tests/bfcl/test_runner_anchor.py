import pytest

from agent_loops.bench.bfcl.adapter import load_cases, load_ground_truth
from agent_loops.bench.bfcl.runner import run_case
from agent_loops.loops import (
    plan_and_execute,
    plan_and_solve,
    react,
    rewoo,
    single_call,
)
from tests.replay.bfcl import PlanReplay, ReActReplay, SingleCallReplay

pytestmark = pytest.mark.integration

CATEGORY = "multi_turn_base"
FILE_SYSTEM = {"GorillaFileSystem"}

ANCHORS = [
    (react, lambda gt: ReActReplay(gt), {"max_steps": 20}, True),
    (rewoo, lambda gt: PlanReplay(gt, style="variable", follow_up=True), {}, True),
    (
        plan_and_solve,
        lambda gt: PlanReplay(gt, style="numbered", follow_up=False),
        {},
        True,
    ),
    (
        plan_and_execute,
        lambda gt: PlanReplay(gt, style="numbered", follow_up=True),
        {"max_rounds": 5},
        True,
    ),
    (single_call, lambda gt: SingleCallReplay(gt), {}, False),
]

IDS = [a[0].NAME for a in ANCHORS]


def _first_case():
    return load_cases(CATEGORY, only_classes=FILE_SYSTEM)[0]


def test_codeact_tool_accounting_matches_the_environment():
    from agent_loops.bench.bfcl.env import BFCLEnv
    from agent_loops.loops import codeact
    from tests.conftest import ScriptedLLM

    case = _first_case()
    env = BFCLEnv(case)
    env.enable_code_execution()

    trace = codeact.run(
        task="t",
        env=env,
        max_steps=3,
        llm=ScriptedLLM(
            [
                {
                    "tool_calls": [
                        {"name": "execute_code", "arguments": {"code": "pwd()\nls()"}}
                    ]
                },
                {"tool_calls": None, "text": "Final: done"},
            ]
        ),
    )

    assert len(env.calls) == trace.n_tool_calls == 2, (
        f"env {len(env.calls)} calls, trace {trace.n_tool_calls} calls; "
        f"{len(env.code_actions)} code actions may have been counted as tool calls"
    )
    assert len(env.code_actions) == 1, "code actions are counted on a separate axis"


@pytest.mark.parametrize("loop,replay,kwargs,full", ANCHORS, ids=IDS)
def test_ground_truth_reaches_the_checker_through_every_loop(
    loop, replay, kwargs, full
):
    case = _first_case()
    gt = load_ground_truth(CATEGORY)[case["id"]]

    result = run_case(
        case, loop, lambda tools: replay(gt), CATEGORY, loop_kwargs=kwargs
    )

    expected = sum(len(turn) for turn in gt)
    assert result.n_tool_calls > 0, (
        f"{loop.NAME}: replayed ground truth but no tool was called"
    )
    if full:
        assert result.n_tool_calls == expected, (
            f"{loop.NAME}: only {result.n_tool_calls} of {expected} ground-truth calls reached"
        )


@pytest.mark.parametrize("loop,replay,kwargs,full", ANCHORS, ids=IDS)
def test_executed_tools_are_all_recorded_in_the_trace(loop, replay, kwargs, full):
    case = _first_case()
    gt = load_ground_truth(CATEGORY)[case["id"]]

    result = run_case(
        case, loop, lambda tools: replay(gt), CATEGORY, loop_kwargs=kwargs
    )

    assert result.n_recorded_tool_calls == result.n_tool_calls, (
        f"{loop.NAME}: env executed {result.n_tool_calls}, trace recorded "
        f"{result.n_recorded_tool_calls}"
    )


def test_ground_truth_through_full_runner_scores_correct():
    case = _first_case()
    gt = load_ground_truth(CATEGORY)[case["id"]]

    result = run_case(
        case,
        react,
        lambda tools: ReActReplay(gt),
        CATEGORY,
        loop_kwargs={"max_steps": 20},
    )

    assert result.valid is True, f"replayed ground truth judged wrong: {result.error}"


def test_runner_anchor_holds_across_file_system_cases():
    cases = load_cases(CATEGORY, only_classes=FILE_SYSTEM)
    gts = load_ground_truth(CATEGORY)
    failures = []

    for case in cases:
        r = run_case(
            case,
            react,
            lambda tools, _id=case["id"]: ReActReplay(gts[_id]),
            CATEGORY,
            loop_kwargs={"max_steps": 20},
        )
        if not r.valid:
            failures.append((case["id"], r.error))

    assert failures == [], f"{len(failures)}/{len(cases)} failed: {failures[:3]}"


def test_rewoo_llm_calls_stay_flat_through_the_runner():
    case = _first_case()
    gt = load_ground_truth(CATEGORY)[case["id"]]

    result = run_case(
        case,
        rewoo,
        lambda tools: PlanReplay(gt, style="variable", follow_up=True),
        CATEGORY,
    )

    assert result.n_llm_calls == 2 * result.n_turns, (
        f"expected 2 per turn, got {result.n_llm_calls} (turns {result.n_turns})"
    )
    assert result.n_steps == result.n_llm_calls + result.n_tool_calls
    assert result.n_tool_calls > 0
