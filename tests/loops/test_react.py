from agent_loops.loops.react import run


def test_react_stops_on_explicit_final_declaration(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}]},
            {"tool_calls": None, "text": "Final: the file is a.txt"},
        ]
    )
    env = recording_env({"ls": lambda path: "a.txt"})

    trace = run(task="list", env=env, llm=llm, max_steps=10)

    assert llm.calls_made == 2
    assert trace.terminated_by == "success"


def test_react_thought_only_step_continues_the_trajectory(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            {"tool_calls": None, "text": "Let me list the directory first"},
            {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}]},
            {"tool_calls": None, "text": "Final: a.txt is there"},
        ]
    )
    env = recording_env({"ls": lambda path: "a.txt"})

    trace = run(task="list", env=env, llm=llm, max_steps=10)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3
    assert [n for n, _ in env.executed] == ["ls"]
    assert "Let me list the directory first" in str(llm.prompts[1])


def test_react_without_final_declaration_runs_to_the_step_budget(
    scripted_llm, recording_env
):
    llm = scripted_llm([{"tool_calls": None, "text": "Hmm, what should I do"}] * 4)
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_steps=3)

    assert trace.terminated_by == "max_steps"
    assert llm.calls_made == 3
    assert trace.n_llm_calls == 3 and trace.n_tool_calls == 0


def test_react_calls_llm_once_per_step(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
            {"tool_calls": [{"name": "ls", "arguments": {"path": "b"}}]},
            {"tool_calls": [{"name": "ls", "arguments": {"path": "c"}}]},
            {"tool_calls": None, "text": "Final: done"},
        ]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="show all three", env=env, llm=llm, max_steps=10)

    assert llm.calls_made == 4
    assert len(trace.steps) == 4


def test_react_transcript_grows_with_observations(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
            {"tool_calls": [{"name": "ls", "arguments": {"path": "b"}}]},
            {"tool_calls": None, "text": "Final: done"},
        ]
    )
    env = recording_env({"ls": lambda path: "observed-" + path})

    run(task="list", env=env, llm=llm, max_steps=10)

    lengths = [len(p) for p in llm.prompts]
    assert lengths == sorted(lengths)
    assert lengths[-1] > lengths[0]


def test_react_stops_at_max_steps(scripted_llm, recording_env):
    llm = scripted_llm(
        [{"tool_calls": [{"name": "ls", "arguments": {"path": "."}}]}] * 5
    )
    env = recording_env({"ls": lambda path: "x"})

    trace = run(task="forever", env=env, llm=llm, max_steps=3)

    assert llm.calls_made == 3
    assert trace.terminated_by == "max_steps"


def test_react_feeds_tool_error_back_as_observation(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "no_such_tool", "arguments": {}}]},
            {"tool_calls": None, "text": "Final: that tool does not exist"},
        ]
    )
    env = recording_env({})

    trace = run(task="anything", env=env, llm=llm, max_steps=5)

    assert trace.steps[0].observation["ok"] is False
    assert llm.calls_made == 2
