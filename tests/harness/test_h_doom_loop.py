from agent_loops.harness import apply
from agent_loops.harness.doom_loop import doom_loop
from agent_loops.loops import react


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def test_third_identical_call_runs_but_carries_a_warning(scripted_llm, recording_env):
    run = apply(react, [doom_loop])
    llm = scripted_llm([_tc("ls", path="a")] * 3 + [_txt("Task completed")])
    env = recording_env({"ls": lambda path: "a/file.txt"})

    trace = run(task="t", env=env, llm=llm, max_steps=10)

    assert trace.terminated_by == "success"
    assert len(env.executed) == 3
    assert "Repetitive pattern detected" not in trace.steps[1].observation["output"]
    assert "Repetitive pattern detected" in trace.steps[2].observation["output"]


def test_fourth_identical_call_is_not_executed_and_stops_the_loop(
    scripted_llm, recording_env
):
    run = apply(react, [doom_loop])
    llm = scripted_llm([_tc("ls", path="a")] * 4 + [_txt("never reached")])
    env = recording_env({"ls": lambda path: "a/file.txt"})

    trace = run(task="t", env=env, llm=llm, max_steps=10)

    assert trace.terminated_by == "doom_loop"
    assert len(env.executed) == 3
    assert llm.calls_made == 4
    assert trace.steps[-2].observation["ok"] is False
    assert "Doom loop" in trace.steps[-2].observation["error"]


def test_different_arguments_reset_the_streak(scripted_llm, recording_env):
    run = apply(react, [doom_loop])
    script = [_tc("ls", path="a"), _tc("ls", path="b")] * 3 + [_txt("Task completed")]
    llm = scripted_llm(script)
    env = recording_env({"ls": lambda path: f"{path}/file.txt"})

    trace = run(task="t", env=env, llm=llm, max_steps=10)

    assert trace.terminated_by == "success"
    assert len(env.executed) == 6


def test_successful_identical_calls_count_the_same_as_failed_ones(
    scripted_llm, recording_env
):
    run = apply(react, [doom_loop])
    llm = scripted_llm([_tc("bad", path="a")] * 4)
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_steps=10)

    assert trace.terminated_by == "doom_loop"
    assert len(env.executed) == 3
