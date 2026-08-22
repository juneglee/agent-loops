import pytest

from agent_loops.bench.bfcl.adapter import load_cases, load_ground_truth

pytestmark = pytest.mark.integration

CATEGORY = "multi_turn_base"


def _score(case, decoded):
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )

    gt = load_ground_truth(CATEGORY)[case["id"]]
    return multi_turn_checker(
        multi_turn_model_result_list_decoded=decoded,
        multi_turn_ground_truth_list=gt,
        test_entry=case,
        test_category=CATEGORY,
        model_name="wiring_anchor",
    )


def _ground_truth_as_decoded(case):
    gt = load_ground_truth(CATEGORY)[case["id"]]
    return [[turn_calls] for turn_calls in gt]


def test_ground_truth_scores_as_correct_through_our_format():
    case = load_cases(CATEGORY, only_classes={"GorillaFileSystem"})[0]

    result = _score(case, _ground_truth_as_decoded(case))

    assert result["valid"] is True, f"ground truth judged wrong: {result}"


def test_empty_answer_scores_as_incorrect():
    case = load_cases(CATEGORY, only_classes={"GorillaFileSystem"})[0]
    n_turns = len(case["question"])

    result = _score(case, [[] for _ in range(n_turns)])

    assert result["valid"] is False


def test_anchor_holds_across_all_file_system_cases():
    cases = load_cases(CATEGORY, only_classes={"GorillaFileSystem"})
    failures = []

    for case in cases:
        result = _score(case, _ground_truth_as_decoded(case))
        if not result["valid"]:
            failures.append((case["id"], result.get("error")))

    assert failures == [], (
        f"{len(failures)}/{len(cases)} cases: ground truth judged wrong"
    )
