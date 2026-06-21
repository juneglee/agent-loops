from pathlib import Path

from agent_loops.bench.tasks.format import load_tasks
from agent_loops.bench.tasks.runner import final_answer, run_task_case
from agent_loops.loops import react
from tests.conftest import ScriptedLLM

BASE = Path(__file__).resolve().parents[1] / "fixtures" / "samples"


def _case(cid):
    return next(c for c in load_tasks(BASE / "tasks.json") if c["id"] == cid)


def test_react_reproducing_the_answer_is_valid():
    case = _case("s002")
    calls = [c for turn in case["gt_calls"] for c in turn]
    llm = ScriptedLLM(
        [
            {"tool_calls": [calls[0]]},
            {"tool_calls": [calls[1]]},
            {"tool_calls": None, "text": "Final: moved and created"},
        ]
    )

    result = run_task_case(
        case, react, lambda tools: llm, BASE, loop_kwargs={"max_steps": 5}
    )

    assert result.valid is True and result.error is None
    assert (
        result.n_llm_calls == 3
        and result.n_tool_calls == 2 == result.n_recorded_tool_calls
    )
    assert result.terminated_by == ["success"] and result.n_turns == 1
    assert result.seconds >= 0


def test_wrong_action_is_reported_as_state_mismatch():
    case = _case("s002")
    llm = ScriptedLLM(
        [
            {"tool_calls": [{"name": "bash", "arguments": {"command": "mkdir wrong"}}]},
            {"tool_calls": None, "text": "Final: done"},
        ]
    )

    result = run_task_case(
        case, react, lambda tools: llm, BASE, loop_kwargs={"max_steps": 5}
    )

    assert result.valid is False and result.error.startswith("state_mismatch:")


def test_query_case_scores_the_final_turn_answer():
    case = _case("s001")
    llm = ScriptedLLM(
        [
            {"tool_calls": [case["gt_calls"][0][0]]},
            {"tool_calls": None, "text": "Final: there are a.md, b.txt, sample.pdf"},
        ]
    )

    result = run_task_case(
        case, react, lambda tools: llm, BASE, loop_kwargs={"max_steps": 5}
    )

    assert result.valid is True


def test_multi_turn_case_runs_every_turn_and_scores_the_final_state():
    case = _case("s003")
    t1, t2 = case["gt_calls"]
    llm = ScriptedLLM(
        [
            {"tool_calls": [t1[0]]},
            {"tool_calls": None, "text": "Final: replaced"},
            {"tool_calls": [t2[0]]},
            {"tool_calls": [t2[1]]},
            {"tool_calls": None, "text": "Final: copied and deleted"},
        ]
    )

    result = run_task_case(
        case, react, lambda tools: llm, BASE, loop_kwargs={"max_steps": 5}
    )

    assert (
        result.valid is True
        and result.n_turns == 2
        and result.terminated_by == ["success", "success"]
    )


def test_n_turns_truncates_both_turns_and_ground_truth():
    case = _case("s003")
    llm = ScriptedLLM(
        [
            {"tool_calls": [case["gt_calls"][0][0]]},
            {"tool_calls": None, "text": "Final: replaced"},
        ]
    )

    result = run_task_case(
        case, react, lambda tools: llm, BASE, loop_kwargs={"max_steps": 5}, n_turns=1
    )

    assert result.valid is True and result.n_turns == 1


def test_trace_sink_receives_every_turn():
    case = _case("s003")
    t1, t2 = case["gt_calls"]
    llm = ScriptedLLM(
        [
            {"tool_calls": [t1[0]]},
            {"tool_calls": None, "text": "Final: 1"},
            {"tool_calls": [t2[0]]},
            {"tool_calls": [t2[1]]},
            {"tool_calls": None, "text": "Final: 2"},
        ]
    )
    seen = []

    run_task_case(
        case,
        react,
        lambda tools: llm,
        BASE,
        loop_kwargs={"max_steps": 5},
        trace_sink=lambda turn, task, trace: seen.append(
            (turn, task, trace.terminated_by)
        ),
    )

    assert [s[0] for s in seen] == [0, 1] and seen[1][1] == case["turns"][1]


def test_final_answer_joins_text_steps_of_a_trace():
    from agent_loops.loops.base import Step, Trace

    trace = Trace(
        task="t",
        loop="react",
        steps=[
            Step(llm_response={"text": "look first"}),
            Step(
                llm_response={"tool_calls": [{"name": "ls", "arguments": {}}]},
                tool_name="ls",
            ),
            Step(llm_response={"text": "Final: a.md"}),
        ],
    )
    assert final_answer(trace) == "look first\nFinal: a.md"


def test_tps_is_read_from_server_timings_when_present():
    case = _case("s002")
    calls = [c for turn in case["gt_calls"] for c in turn]
    raw = {"timings": {"predicted_per_second": 20.0}}
    llm = ScriptedLLM(
        [
            {"tool_calls": [calls[0]], "raw": raw},
            {
                "tool_calls": [calls[1]],
                "raw": {"timings": {"predicted_per_second": 40.0}},
            },
            {"tool_calls": None, "text": "Final: done", "raw": raw},
        ]
    )

    result = run_task_case(
        case, react, lambda tools: llm, BASE, loop_kwargs={"max_steps": 5}
    )

    assert result.tps == 80.0 / 3
