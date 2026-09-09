import pytest

from agent_loops.loops import (
    adapt,
    codeact,
    dfsdt,
    fixed_pipeline,
    plan_and_act,
    react,
    reflexion,
)
from tests.conftest import RecordingEnv, ScriptedLLM

RUNAWAY = {"tool_calls": None, "text": "", "truncated": True}
EMPTY = {"tool_calls": None, "text": "", "truncated": False}
CALL = {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]}


def _env():
    return RecordingEnv({"ls": lambda path: path, "execute_code": lambda code: "ok"})


@pytest.mark.parametrize(
    "module, kwargs",
    [
        (react, {"max_steps": 5}),
        (codeact, {"max_steps": 5}),
        (fixed_pipeline, {}),
        (reflexion, {"max_trials": 2, "max_steps": 5}),
        (adapt, {"max_depth": 2}),
        (dfsdt, {"breadth": 2, "max_calls": 6}),
        (plan_and_act, {"max_steps": 5}),
    ],
)
def test_truncated_empty_response_stops_the_loop(module, kwargs):
    llm = ScriptedLLM([{"tool_calls": None, "text": "1. look at a"}, CALL, RUNAWAY])
    trace = module.run(task="t", env=_env(), llm=llm, **kwargs)

    assert trace.terminated_by == "truncated"
    assert llm.calls_made <= 3


@pytest.mark.parametrize(
    "module, kwargs",
    [
        (react, {"max_steps": 5}),
        (codeact, {"max_steps": 5}),
        (fixed_pipeline, {}),
        (reflexion, {"max_trials": 2, "max_steps": 5}),
        (adapt, {"max_depth": 2}),
        (dfsdt, {"breadth": 2, "max_calls": 6}),
        (plan_and_act, {"max_steps": 5}),
    ],
)
def test_empty_response_ends_the_turn_as_no_action(module, kwargs):
    if module in (dfsdt, plan_and_act):
        pytest.skip("keeps the authors' handling of unparseable responses")
    llm = ScriptedLLM([{"tool_calls": None, "text": "1. look at a"}, CALL, EMPTY])
    trace = module.run(task="t", env=_env(), llm=llm, **kwargs)

    assert trace.terminated_by == "no_action"
    assert llm.calls_made <= 3


def test_react_keeps_thinking_when_text_is_present_but_not_truncated():
    llm = ScriptedLLM(
        [
            {"tool_calls": None, "text": "let me think"},
            CALL,
            {"tool_calls": None, "text": "Final: done"},
        ]
    )
    trace = react.run(task="t", env=_env(), llm=llm, max_steps=5)

    assert trace.terminated_by == "success"
