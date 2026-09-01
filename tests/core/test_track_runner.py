from pathlib import Path

import pytest

from agent_loops.bench.core.env import Env
from agent_loops.bench.core.registry import Registry, Stack
from agent_loops.bench.core.runner import Budgets, Runner
from agent_loops.bench.core.track import Track
from agent_loops.bench.tasks.env import WorkspaceEnv
from agent_loops.bench.tasks.track import TaskTrack
from agent_loops.loops import react
from tests.conftest import ScriptedLLM
from tests.replay.tasks import UniversalReplay, replay_mode

SAMPLES = Path(__file__).resolve().parents[1] / "fixtures" / "samples"


def test_registry_resolves_plain_composed_and_layered_names():
    reg = Registry.default()
    plain = reg.resolve("react")
    assert (
        isinstance(plain, Stack)
        and plain.name == "react"
        and plain.kwargs == {"max_steps": 10}
        and plain.layers == ()
    )
    composed = reg.resolve("planner+react")
    assert composed.name == "planner+react" and composed.kwargs == {}
    layered = reg.resolve("react", layers=["todo"])
    assert (
        layered.name == "react+todo"
        and layered.kwargs == {"max_steps": 10}
        and layered.layers == ("todo",)
    )
    assert [s["function"]["name"] for s in reg.extra_tools(["todo"])] == ["update_todo"]
    assert "react" in reg.names() and "routed+codeact" in reg.names()


def test_task_track_exposes_cases_turns_tools_env_and_score():
    track = TaskTrack(SAMPLES)
    assert isinstance(track, Track)
    assert track.name == "tasks:samples" and track.revision().startswith(
        "tasks:samples@"
    )
    ids = [c["id"] for c in track.cases("single_turn_single_step")]
    assert ids == ["s001", "s004"]
    case = track.cases("single_turn_multi_step")[0]
    assert track.turns_of(case) == case["turns"]
    assert [t["function"]["name"] for t in track.tools_for(case)][:2] == [
        "read_file",
        "write_file",
    ]
    env = track.make_env(case, Budgets())
    assert isinstance(env, WorkspaceEnv) and isinstance(env, Env)
    env.close()


def test_runner_runs_a_task_case_end_to_end():
    track = TaskTrack(SAMPLES)
    case = next(c for c in track.cases("single_turn_multi_step") if c["id"] == "s002")
    calls = [c for turn in case["gt_calls"] for c in turn]
    llm = ScriptedLLM(
        [
            {"tool_calls": [calls[0]]},
            {"tool_calls": [calls[1]]},
            {"tool_calls": None, "text": "Final: done"},
        ]
    )
    runner = Runner(track, llm_factory=lambda tools: llm)

    result = runner.run_case(case, Registry.default().resolve("react"))

    assert result.valid is True and result.loop == "react"
    assert (
        result.n_llm_calls == 3
        and result.n_tool_calls == 2 == result.n_recorded_tool_calls
    )
    assert result.terminated_by == ["success"]


def test_runner_truncates_turns_and_scores_only_what_ran():
    track = TaskTrack(SAMPLES)
    case = next(c for c in track.cases("multi_turn_multi_step") if c["id"] == "s003")
    llm = ScriptedLLM(
        [
            {"tool_calls": [case["gt_calls"][0][0]]},
            {"tool_calls": None, "text": "Final: 1"},
        ]
    )

    result = Runner(track, lambda tools: llm).run_case(
        case, Registry.default().resolve("react"), n_turns=1
    )

    assert result.valid is True and result.n_turns == 1


def test_runner_applies_stack_kwargs_and_trace_sink():
    track = TaskTrack(SAMPLES)
    case = next(c for c in track.cases("single_turn_single_step") if c["id"] == "s001")
    seen = []
    runner = Runner(
        track,
        lambda tools: UniversalReplay(case, replay_mode("react")),
        trace_sink=lambda turn, task, trace: seen.append(turn),
    )

    result = runner.run_case(
        case, Stack(name="react", run=react.run, kwargs={"max_steps": 3})
    )

    assert result.valid is True and seen == [0]


def test_runner_reports_scorer_exceptions_as_case_errors_not_crashes():
    class Broken(TaskTrack):
        def score(self, case, env, turn_traces):
            raise RuntimeError("boom")

    track = Broken(SAMPLES)
    case = track.cases("single_turn_single_step")[0]
    result = Runner(track, lambda tools: UniversalReplay(case, "react")).run_case(
        case, Registry.default().resolve("react")
    )
    assert result.valid is False and result.error.startswith("scorer:RuntimeError")


def test_env_protocol_is_satisfied_by_both_environments(tmp_path):
    from agent_loops.bench.bfcl.env import BFCLEnv

    assert isinstance(WorkspaceEnv(None), Env)
    assert all(
        hasattr(BFCLEnv, m) for m in ("execute", "tool_names", "enable_code_execution")
    )


@pytest.mark.integration
def test_bfcl_track_through_the_same_runner_matches_the_legacy_run_case():
    from agent_loops.bench.bfcl.adapter import load_cases, load_ground_truth
    from agent_loops.bench.bfcl.partition import FILE_MANAGEMENT, partition
    from agent_loops.bench.bfcl.runner import run_case
    from agent_loops.bench.bfcl.track import BfclTrack
    from tests.replay.bfcl import ReActReplay

    category = "multi_turn_base"
    case = partition(category, only_classes=FILE_MANAGEMENT)["single_turn_multi_step"][
        0
    ]
    gt = load_ground_truth(category)[case["id"]]
    legacy = run_case(
        case,
        react,
        lambda tools: ReActReplay(gt),
        category,
        loop_kwargs={"max_steps": 20},
        n_turns=1,
    )
    track = BfclTrack(category, only_classes=FILE_MANAGEMENT)
    assert case["id"] in {c["id"] for c in track.cases("single_turn_multi_step")}
    new = Runner(track, lambda tools: ReActReplay(gt)).run_case(
        case, Stack("react", react.run, {"max_steps": 20}), n_turns=1
    )
    assert (new.valid, new.n_llm_calls, new.n_tool_calls, new.terminated_by) == (
        legacy.valid,
        legacy.n_llm_calls,
        legacy.n_tool_calls,
        legacy.terminated_by,
    )
    assert load_cases is not None
