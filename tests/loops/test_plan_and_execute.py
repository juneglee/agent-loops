from agent_loops.loops.plan_and_execute import run


def _plan(text):
    return {"tool_calls": None, "text": text}


def test_executes_whole_plan_before_replanning(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan("1. ls[path=a]\n2. ls[path=b]\n3. ls[path=c]"),
            _plan(""),
        ]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="all three", env=env, llm=llm, max_rounds=5)

    assert llm.calls_made == 2
    assert len(env.executed) == 3
    assert trace.terminated_by == "success"


def test_replans_when_work_remains(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan("1. ls[path=a]"),
            _plan("1. ls[path=b]"),
            _plan(""),
        ]
    )
    env = recording_env({"ls": lambda path: path})

    run(task="two", env=env, llm=llm, max_rounds=5)

    assert llm.calls_made == 3
    assert [n for n, _ in env.executed] == ["ls", "ls"]


def test_llm_calls_scale_with_rounds_not_actions(scripted_llm, recording_env):
    plan5 = "\n".join(f"{i}. ls[path=p{i}]" for i in range(1, 6))
    llm = scripted_llm([_plan(plan5), _plan("")])
    env = recording_env({"ls": lambda path: path})

    run(task="five", env=env, llm=llm, max_rounds=5)

    assert llm.calls_made == 2
    assert len(env.executed) == 5


def test_stops_at_max_rounds(scripted_llm, recording_env):
    llm = scripted_llm([_plan("1. ls[path=x]")] * 6)
    env = recording_env({"ls": lambda path: path})

    trace = run(task="forever", env=env, llm=llm, max_rounds=3)

    assert llm.calls_made == 3
    assert trace.terminated_by == "max_steps"


def test_records_parse_failure_when_first_plan_unreadable(scripted_llm, recording_env):
    llm = scripted_llm([_plan("I cannot make a plan")])
    env = recording_env({})

    trace = run(task="anything", env=env, llm=llm, max_rounds=3)

    assert trace.parse_ok is False
    assert trace.terminated_by == "parse_fail"
    assert env.executed == []


def test_observations_are_fed_into_the_replan(scripted_llm, recording_env):
    llm = scripted_llm([_plan("1. ls[path=a]"), _plan("")])
    env = recording_env({"ls": lambda path: "observed-output"})

    run(task="task", env=env, llm=llm, max_rounds=3)

    assert "observed-output" in str(llm.prompts[1])


def test_replanner_sees_past_steps_as_task_result_pairs(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan("1. ls[path=a]\n2. ls[path=b]"),
            _plan(""),
        ]
    )
    env = recording_env({"ls": lambda path: f"contents of {path}"})

    run(task="t", env=env, llm=llm, max_rounds=3)

    replan_prompt = str(llm.prompts[1])
    assert "1. ls[path=a]" in replan_prompt
    assert "ls" in replan_prompt and "'path': 'a'" in replan_prompt
    assert "contents of a" in replan_prompt
