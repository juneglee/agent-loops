import ast

import pytest

from agent_loops.bench.bfcl.adapter import load_cases, load_ground_truth
from agent_loops.bench.bfcl.partition import truncate_to_first_turn
from agent_loops.bench.bfcl.runner import run_case
from agent_loops.loops import single_call
from tests.conftest import ScriptedLLM

pytest.importorskip("bfcl_eval")

pytestmark = pytest.mark.integration


CASE = next(c for c in load_cases("multi_turn_base") if c["id"] == "multi_turn_base_6")
GT_TURN0 = load_ground_truth("multi_turn_base")[CASE["id"]][0]


def _calls_from_gt(gt_calls):
    out = []
    for text in gt_calls:
        node = ast.parse(text, mode="eval").body
        out.append(
            {
                "name": node.func.id,
                "arguments": {
                    kw.arg: ast.literal_eval(kw.value) for kw in node.keywords
                },
            }
        )
    return out


def _replay(calls):
    return lambda tools: ScriptedLLM([{"tool_calls": calls}])


def test_correct_replay_scores_valid_on_a_truncated_case():
    case = truncate_to_first_turn(CASE)
    r = run_case(
        case,
        single_call,
        _replay(_calls_from_gt(GT_TURN0)),
        "multi_turn_base",
        n_turns=1,
    )
    assert r.valid is True, r.error


def test_wrong_then_correct_is_order_independent():
    case = truncate_to_first_turn(CASE)
    wrong = run_case(
        case,
        single_call,
        _replay([{"name": "mkdir", "arguments": {"dir_name": "zzz_wrong"}}]),
        "multi_turn_base",
        n_turns=1,
    )
    assert wrong.valid is False
    right = run_case(
        case,
        single_call,
        _replay(_calls_from_gt(GT_TURN0)),
        "multi_turn_base",
        n_turns=1,
    )
    assert right.valid is True, f"state leaked from the previous run: {right.error}"


def test_repeated_correct_replay_stays_valid():
    case = truncate_to_first_turn(CASE)
    for i in range(3):
        r = run_case(
            case,
            single_call,
            _replay(_calls_from_gt(GT_TURN0)),
            "multi_turn_base",
            n_turns=1,
        )
        assert r.valid is True, f"contaminated on run {i + 1}: {r.error}"


def test_scorer_cache_is_empty_after_scoring():
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as u

    case = truncate_to_first_turn(CASE)
    run_case(
        case,
        single_call,
        _replay(_calls_from_gt(GT_TURN0)),
        "multi_turn_base",
        n_turns=1,
    )
    leftover = [k for k in vars(u) if k.endswith("_instance") and CASE["id"] in k]
    assert leftover == []
