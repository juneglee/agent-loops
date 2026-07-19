from agent_loops.loops.plan_and_act import run


def _plan(text):
    return {"tool_calls": None, "text": text}


def _act(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _final(text="Final: done"):
    return {"tool_calls": None, "text": text}


def _plans(messages):
    return [
        m["content"]["plan"]
        for m in messages
        if isinstance(m.get("content"), dict) and "plan" in m["content"]
    ]


def test_planner_writes_natural_language_and_executor_grounds_a_step(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _plan("1. look at folder a\n2. look at folder b"),
            _act("ls", path="a"),
            _plan("1. look at folder b"),
            _act("ls", path="b"),
            _plan("1. nothing left; report"),
            _final("Final: saw both"),
        ]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="both", env=env, llm=llm, max_steps=10)

    assert trace.terminated_by == "success"
    assert [a for _, a in env.executed] == [{"path": "a"}, {"path": "b"}]
    assert llm.calls_made == 6
    assert llm.kwargs[0].get("want") == "text"
    assert llm.kwargs[1].get("want") != "text"
    assert trace.n_llm_calls == 6 and trace.n_tool_calls == 2


def test_plan_and_act_llm_calls_scale_with_action_count(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan("1. a"),
            _act("ls", path="a"),
            _plan("1. b"),
            _act("ls", path="b"),
            _plan("1. c"),
            _act("ls", path="c"),
            _plan("1. report"),
            _final(),
        ]
    )
    env = recording_env({"ls": lambda path: path})

    run(task="three", env=env, llm=llm, max_steps=10)

    assert llm.calls_made == 8
    assert len(env.executed) == 3


def test_default_budget_is_thirty_steps(scripted_llm, recording_env):
    script = [
        r
        for i in range(31)
        for r in (_plan(f"1. look at {i}"), _act("ls", path=str(i)))
    ]
    llm = scripted_llm(script)
    env = recording_env({"ls": lambda path: path})

    trace = run(task="forever", env=env, llm=llm)

    assert trace.terminated_by == "max_steps"
    assert len(env.executed) == 30
    assert llm.calls_made == 60


def test_plan_and_act_stops_at_max_steps(scripted_llm, recording_env):
    script = [
        r for i in range(5) for r in (_plan(f"1. look at {i}"), _act("ls", path=str(i)))
    ]
    llm = scripted_llm(script)
    env = recording_env({"ls": lambda path: path})

    trace = run(task="forever", env=env, llm=llm, max_steps=3)

    assert trace.terminated_by == "max_steps"
    assert len(env.executed) == 3


def test_executor_sees_whole_plan_and_previous_rounds(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan("1. look at folder a\n2. look at folder b"),
            _act("ls", path="a"),
            _plan("1. look at folder b\n2. report"),
            _act("ls", path="b"),
            _plan("1. report"),
            _final(),
        ]
    )
    env = recording_env({"ls": lambda path: f"contents of {path}"})

    run(task="the original request", env=env, llm=llm, max_steps=10)

    first = str(llm.prompts[1])
    assert "the original request" in first
    assert _plans(llm.prompts[1]) == ["1. look at folder a\n2. look at folder b"]
    assert "This step" not in first
    second = str(llm.prompts[3])
    assert _plans(llm.prompts[3]) == ["1. look at folder b\n2. report"]
    assert "'path': 'a'" in second and "contents of a" in second


def test_replanner_sees_only_the_latest_plan_and_the_actions_taken(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _plan("1. look at folder a\n2. look at folder b"),
            _act("ls", path="a"),
            _plan("1. look at folder b\n2. report"),
            _act("ls", path="b"),
            _plan("1. report"),
            _final(),
        ]
    )
    env = recording_env({"ls": lambda path: f"contents of {path}"})

    run(task="t", env=env, llm=llm, max_steps=10)

    replan1 = str(llm.prompts[2])
    assert _plans(llm.prompts[2]) == ["1. look at folder a\n2. look at folder b"]
    assert "'path': 'a'" in replan1 and "contents of a" in replan1
    replan2 = str(llm.prompts[4])
    assert _plans(llm.prompts[4]) == ["1. look at folder b\n2. report"]
    assert "'path': 'a'" in replan2 and "'path': 'b'" in replan2


def test_executor_final_declaration_ends_the_loop(scripted_llm, recording_env):
    llm = scripted_llm([_plan("1. wrap up"), _final("Final: already done")])
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_steps=5)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 2


def test_planner_cannot_terminate(scripted_llm, recording_env):
    llm = scripted_llm(
        [_plan("1. look at a"), _act("ls", path="a"), _plan("Final: done")]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="t", env=env, llm=llm, max_steps=5)

    assert trace.terminated_by == "parse_fail"
    assert trace.parse_ok is False
    assert len(env.executed) == 1


def test_planner_without_readable_plan_is_parse_fail(scripted_llm, recording_env):
    trace = run(
        task="t",
        env=recording_env({}),
        llm=scripted_llm([_plan("Task completed")]),
        max_steps=5,
    )
    assert trace.terminated_by == "parse_fail"

    trace = run(
        task="t", env=recording_env({}), llm=scripted_llm([_plan("")]), max_steps=5
    )
    assert trace.terminated_by == "parse_fail"


def test_three_unparseable_executor_responses_stop_as_parse_fail(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _plan("1. do something"),
            _plan("I do not know what to do"),
            _plan("1. do something"),
            _plan("still no idea"),
            _plan("1. do something"),
            _plan("nope"),
        ]
    )
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_steps=10)

    assert "I do not know what to do" in str(llm.prompts[2])
    assert trace.terminated_by == "parse_fail"
    assert llm.calls_made == 6


def test_two_unparseable_responses_do_not_stop(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan("1. do something"),
            _plan("hmm"),
            _plan("1. do something"),
            _plan("hmm"),
            _plan("1. do something"),
            _act("ls", path="a"),
            _plan("1. report"),
            _final(),
        ]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="t", env=env, llm=llm, max_steps=10)

    assert trace.terminated_by == "success"
    assert len(env.executed) == 1


def test_five_identical_actions_stop_as_max_steps(scripted_llm, recording_env):
    llm = scripted_llm([_plan("1. look at x"), _act("ls", path="x")] * 6)
    env = recording_env({"ls": lambda path: path})

    trace = run(task="t", env=env, llm=llm, max_steps=30)

    assert trace.terminated_by == "max_steps"
    assert len(env.executed) == 5
    assert llm.calls_made == 10


def test_four_identical_actions_do_not_stop(scripted_llm, recording_env):
    llm = scripted_llm(
        [_plan("1. look at x"), _act("ls", path="x")] * 4
        + [_plan("1. report"), _final()]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="t", env=env, llm=llm, max_steps=30)

    assert trace.terminated_by == "success"
    assert len(env.executed) == 4


def test_only_the_first_tool_call_of_a_round_executes(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan("1. look at a\n2. look at b"),
            {
                "tool_calls": [
                    {"name": "ls", "arguments": {"path": "a"}},
                    {"name": "ls", "arguments": {"path": "b"}},
                ]
            },
            _plan("1. look at b"),
            _final(),
        ]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="t", env=env, llm=llm, max_steps=10)

    assert [a for _, a in env.executed] == [{"path": "a"}]
    assert trace.n_tool_calls == 1
    replan = str(llm.prompts[2])
    assert "ignored" in replan and "'path': 'b'" in replan
