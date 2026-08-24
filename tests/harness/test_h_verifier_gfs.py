from types import SimpleNamespace

import pytest

from agent_loops.bench.bfcl.adapter import load_cases, load_ground_truth
from agent_loops.bench.bfcl.env import BFCLEnv
from agent_loops.bench.bfcl.runner import run_case
from agent_loops.harness import apply
from agent_loops.harness.verifier import verifier
from agent_loops.loops import react
from tests.replay.bfcl import ReActReplay

pytestmark = pytest.mark.integration

CATEGORY = "multi_turn_base"


def _first_case():
    return load_cases(CATEGORY, only_classes={"GorillaFileSystem"})[0]


def test_correct_operations_raise_no_false_positives():
    env = verifier()().wrap_env(BFCLEnv(_first_case()))
    sequence = [
        ("mkdir", {"dir_name": "vtemp"}),
        ("cd", {"folder": "vtemp"}),
        ("touch", {"file_name": "v.txt"}),
        ("echo", {"content": "hello", "file_name": "v.txt"}),
        ("cd", {"folder": ".."}),
        ("mv", {"source": "vtemp", "destination": "vtemp2"}),
    ]

    failures = []
    for name, args in sequence:
        obs = env.execute(name, args)
        if not obs["ok"]:
            failures.append((name, obs["error"]))

    assert failures == [], (
        f"valid runs flipped to failure (false positives): {failures}"
    )


def test_gt_replay_through_react_verifier_still_scores_correct():
    case = _first_case()
    gt = load_ground_truth(CATEGORY)[case["id"]]
    run = apply(react, [verifier()])

    result = run_case(
        case,
        SimpleNamespace(NAME=run.NAME, run=run),
        lambda tools: ReActReplay(gt),
        CATEGORY,
        loop_kwargs={"max_steps": 20},
    )

    assert result.valid is True, f"replayed ground truth judged wrong: {result.error}"
    assert result.n_tool_calls == sum(len(t) for t in gt)
