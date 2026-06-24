from agent_loops.loops.plan_and_solve import run


def _plan(text):
    return {"tool_calls": None, "text": text}


def test_plan_and_solve_makes_exactly_one_llm_call(scripted_llm, recording_env):
    llm = scripted_llm([_plan("1. ls[path=a]\n2. ls[path=b]\n3. ls[path=c]")])
    env = recording_env({"ls": lambda path: path})

    trace = run(task="all three", env=env, llm=llm)

    assert llm.calls_made == 1
    assert len(env.executed) == 3
    assert trace.n_llm_calls == 1


def test_plan_and_solve_does_not_substitute_variables(scripted_llm, recording_env):
    llm = scripted_llm([_plan("1. ls[path=docs]\n2. cat[path=#E1]")])
    env = recording_env({"ls": lambda path: "found.txt", "cat": lambda path: path})

    run(task="find and read", env=env, llm=llm)

    assert env.executed[1] == ("cat", {"path": "#E1"})


def test_plan_and_solve_does_not_replan_on_failure(scripted_llm, recording_env):
    llm = scripted_llm([_plan("1. no_such_tool[x=1]\n2. ls[path=b]")])
    env = recording_env({"ls": lambda path: path})

    trace = run(task="anything", env=env, llm=llm)

    assert llm.calls_made == 1
    assert len(env.executed) == 2
    assert trace.steps[0].llm_response is not None


def test_plan_and_solve_records_parse_failure(scripted_llm, recording_env):
    llm = scripted_llm([_plan("I cannot make a plan")])
    env = recording_env({})

    trace = run(task="anything", env=env, llm=llm)

    assert trace.parse_ok is False
    assert trace.terminated_by == "parse_fail"
