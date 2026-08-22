from types import SimpleNamespace

import pytest

from agent_loops.bench.bfcl.adapter import load_cases, load_ground_truth
from agent_loops.bench.bfcl.runner import run_case
from agent_loops.compose import adaptive, hierarchical, routed
from agent_loops.loops import react
from tests.replay.bfcl import GateReplay, HierarchicalReplay, ReActReplay

pytestmark = pytest.mark.integration

CATEGORY = "multi_turn_base"


def _first_case():
    return load_cases(CATEGORY, only_classes={"GorillaFileSystem"})[0]


def _as_module(run):
    return SimpleNamespace(NAME=run.NAME, run=run)


def _anchor(composed_run, replay_factory):
    case = _first_case()
    gt = load_ground_truth(CATEGORY)[case["id"]]
    result = run_case(
        case, _as_module(composed_run), lambda tools: replay_factory(gt), CATEGORY
    )
    expected = sum(len(turn) for turn in gt)
    return result, expected


def test_gt_through_planner_react_scores_correct():
    run = hierarchical(react, max_rounds=5, worker_kwargs={"max_steps": 20})
    result, expected = _anchor(run, HierarchicalReplay)

    assert result.valid is True, f"replayed ground truth judged wrong: {result.error}"
    assert result.n_tool_calls == expected
    assert result.n_recorded_tool_calls == result.n_tool_calls


def test_gt_through_adaptive_react_scores_correct():
    run = adaptive(react, worker_kwargs={"max_steps": 20})
    result, expected = _anchor(run, ReActReplay)

    assert result.valid is True, f"replayed ground truth judged wrong: {result.error}"
    assert result.n_tool_calls == expected
    assert result.n_recorded_tool_calls == result.n_tool_calls
    gt = load_ground_truth(CATEGORY)[_first_case()["id"]]
    assert result.n_llm_calls == sum(len(t) + 1 for t in gt)


def test_gt_through_routed_react_scores_correct():
    run = routed(react, worker_kwargs={"max_steps": 20})
    result, expected = _anchor(run, GateReplay)

    assert result.valid is True, f"replayed ground truth judged wrong: {result.error}"
    assert result.n_tool_calls == expected
    assert result.n_recorded_tool_calls == result.n_tool_calls
    gt = load_ground_truth(CATEGORY)[_first_case()["id"]]
    assert result.n_llm_calls == len(gt) + sum(len(t) + 1 for t in gt)
