from agent_loops.harness import apply
from agent_loops.harness.verifier import verifier
from agent_loops.loops import react, single_call


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def _always_violated(env, arguments, observation):
    return "src still exists"


def _always_ok(env, arguments, observation):
    return None


def test_failed_postcondition_turns_a_quiet_success_into_an_error(
    scripted_llm, recording_env
):
    run = apply(react, [verifier({"mv": _always_violated})])
    llm = scripted_llm(
        [_tc("mv", source="a.txt", destination="b"), _txt("Task completed")]
    )
    env = recording_env({"mv": lambda source, destination: "moved"})

    trace = run(task="t", env=env, llm=llm)

    obs = next(s.observation for s in trace.steps if s.tool_name == "mv")
    assert obs["ok"] is False
    assert obs["error"].startswith("postcondition:")
    assert "src still exists" in obs["error"]
    assert [n for n, _ in env.executed] == ["mv"]


def test_passing_postcondition_leaves_the_observation_untouched(
    scripted_llm, recording_env
):
    run = apply(react, [verifier({"mv": _always_ok})])
    llm = scripted_llm([_tc("mv", source="a", destination="b"), _txt("Task completed")])
    env = recording_env({"mv": lambda source, destination: "moved"})

    trace = run(task="t", env=env, llm=llm)

    obs = next(s.observation for s in trace.steps if s.tool_name == "mv")
    assert obs["ok"] is True and obs["error"] is None and obs["output"] == "moved"
    assert trace.terminated_by == "success"


def test_tools_without_a_check_entry_pass_through(scripted_llm, recording_env):
    run = apply(single_call, [verifier({"mv": _always_violated})])
    llm = scripted_llm([_tc("ls", path="a")])
    env = recording_env({"ls": lambda path: "x"})

    trace = run(task="t", env=env, llm=llm)

    obs = trace.steps[0].observation
    assert obs["ok"] is True


def test_already_failed_observations_are_not_double_flagged(
    scripted_llm, recording_env
):
    def _boom(**_kw):
        raise RuntimeError("no such file")

    run = apply(single_call, [verifier({"mv": _always_violated})])
    llm = scripted_llm([_tc("mv", source="a", destination="b")])
    env = recording_env({"mv": _boom})

    trace = run(task="t", env=env, llm=llm)

    obs = trace.steps[0].observation
    assert obs["ok"] is False
    assert "no such file" in obs["error"]


def test_checks_do_not_go_through_env_execute(scripted_llm, recording_env):
    seen = []

    def _check(env, arguments, observation):
        seen.append(arguments)

    run = apply(react, [verifier({"mv": _check})])
    llm = scripted_llm([_tc("mv", source="a", destination="b"), _txt("Task completed")])
    env = recording_env({"mv": lambda source, destination: "moved"})

    run(task="t", env=env, llm=llm)

    assert len(seen) == 1
    assert len(env.executed) == 1


def test_layer_name_and_zero_llm_overhead(scripted_llm, recording_env):
    assert apply(react, [verifier({})]).NAME == "react+verifier"

    script = [_tc("ls", path="a"), _txt("Task completed")]
    bare_llm = scripted_llm(list(script))
    react.run(task="t", env=recording_env({"ls": lambda path: "x"}), llm=bare_llm)
    layered_llm = scripted_llm(list(script))
    apply(react, [verifier({})])(
        task="t", env=recording_env({"ls": lambda path: "x"}), llm=layered_llm
    )
    assert layered_llm.calls_made == bare_llm.calls_made == 2
