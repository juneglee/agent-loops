import pytest

from agent_loops.loops import plan_and_solve, react, rewoo, single_call
from agent_loops.loops.base import Step, Trace


def _env(recording_env):
    return recording_env({"ls": lambda path: path})


def test_step_without_an_llm_response_is_not_counted_as_an_llm_call():
    trace = Trace(task="t", loop="x")
    trace.steps.append(Step(llm_response={"text": "plan"}))
    trace.steps.append(
        Step(llm_response=None, tool_name="ls", observation={"ok": True})
    )

    assert len(trace.steps) == 2
    assert trace.n_llm_calls == 1


@pytest.mark.parametrize("n_tools", [1, 3, 8])
def test_rewoo_llm_calls_stay_flat_as_tool_count_grows(
    n_tools, recording_env, scripted_llm
):
    plan = "\n".join(f"#E{i} = ls[path=p{i}]" for i in range(1, n_tools + 1))
    env = _env(recording_env)
    trace = rewoo.run(
        task="t",
        env=env,
        llm=scripted_llm(
            [{"tool_calls": None, "text": plan}, {"tool_calls": None, "text": "done"}]
        ),
    )

    assert len(env.executed) == n_tools, "tools run exactly as planned"
    assert trace.n_llm_calls == 2, "two calls regardless of tool count"


@pytest.mark.parametrize("n_tools", [1, 3, 8])
def test_plan_and_solve_makes_exactly_one_llm_call(
    n_tools, recording_env, scripted_llm
):
    plan = "\n".join(f"{i}. ls[path=p{i}]" for i in range(1, n_tools + 1))
    env = _env(recording_env)
    trace = plan_and_solve.run(
        task="t", env=env, llm=scripted_llm([{"tool_calls": None, "text": plan}])
    )

    assert len(env.executed) == n_tools
    assert trace.n_llm_calls == 1


def test_react_llm_calls_scale_with_steps(recording_env, scripted_llm):
    env = _env(recording_env)
    trace = react.run(
        task="t",
        env=env,
        max_steps=5,
        llm=scripted_llm(
            [
                {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
                {"tool_calls": [{"name": "ls", "arguments": {"path": "b"}}]},
                {"tool_calls": None, "text": "Final: done"},
            ]
        ),
    )

    assert len(env.executed) == 2
    assert trace.n_llm_calls == 3, "two actions plus one finish"


def test_single_call_is_the_floor(recording_env, scripted_llm):
    env = _env(recording_env)
    trace = single_call.run(
        task="t",
        env=env,
        llm=scripted_llm(
            [{"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]}]
        ),
    )

    assert trace.n_llm_calls == 1
