import pytest

from agent_loops.bench.bfcl.adapter import load_cases, load_ground_truth
from agent_loops.bench.bfcl.track import (
    SCORER_MODEL_NAME,
    BfclTrack,
    _release_scorer_instances,
)

pytest.importorskip("bfcl_eval")

pytestmark = pytest.mark.integration

CASE_ID = "multi_turn_base_6"


def _case():
    return next(c for c in load_cases("multi_turn_base") if c["id"] == CASE_ID)


def _partial_ground_truth():
    gt = load_ground_truth("multi_turn_base")[CASE_ID]
    decoded = [[[call] for call in turn] for turn in gt]
    decoded[-1] = decoded[-1][:-1]
    return decoded, gt


def _args(call):
    import ast

    node = ast.parse(call, mode="eval").body
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


def _score(decoded, gt, model_name):
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )

    try:
        return multi_turn_checker(
            multi_turn_model_result_list_decoded=decoded,
            multi_turn_ground_truth_list=gt,
            test_entry=_case(),
            test_category="multi_turn_base",
            model_name=model_name,
        )
    finally:
        _release_scorer_instances(CASE_ID)


def test_the_scorer_gets_a_fixed_identifier_not_the_stack_name():
    assert SCORER_MODEL_NAME.isidentifier()
    assert "+" not in SCORER_MODEL_NAME


def test_a_plus_in_the_model_name_silently_passes_a_partial_trajectory():
    decoded, gt = _partial_ground_truth()

    assert _score(decoded, gt, "agentloops_react+todo")["valid"] is True


def test_the_sanitised_name_rejects_the_same_partial_trajectory():
    decoded, gt = _partial_ground_truth()

    scored = _score(decoded, gt, SCORER_MODEL_NAME)

    assert scored["valid"] is False
    assert scored["error_type"] == "multi_turn:instance_state_mismatch"


def test_track_score_rejects_a_partial_trajectory_for_a_layered_stack():
    from types import SimpleNamespace

    decoded, _ = _partial_ground_truth()
    steps = [
        SimpleNamespace(tool_name=call.split("(")[0], tool_arguments=_args(call))
        for turn in decoded
        for [call] in turn
    ]
    turns = []
    for turn in decoded:
        n = len(turn)
        turns.append(SimpleNamespace(loop="react+todo", steps=steps[:n]))
        steps = steps[n:]

    valid, error = BfclTrack("multi_turn_base").score(_case(), None, turns)

    assert valid is False
    assert "instance_state_mismatch" in (error or "")


def test_release_matches_keys_the_checker_sanitised():
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils

    g = vars(multi_turn_utils)
    g["agentloops_eval_multi_turn_x_1_GorillaFileSystem_instance"] = object()
    g["agentloops_multi_turn_x_1_GorillaFileSystem_instance"] = object()

    _release_scorer_instances("multi_turn.x-1")

    assert not [k for k in g if k.endswith("_instance") and "multi_turn_x_1" in k]
