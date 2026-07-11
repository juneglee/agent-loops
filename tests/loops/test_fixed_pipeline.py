from agent_loops.loops.fixed_pipeline import run


def _stage(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def test_fixed_pipeline_makes_exactly_three_llm_calls(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _stage("ls", path="."),
            _stage("mv", src="a", dst="b"),
            _stage("ls", path="b"),
        ]
    )
    env = recording_env({"ls": lambda path: path, "mv": lambda src, dst: "moved"})

    trace = run(task="move it", env=env, llm=llm)

    assert llm.calls_made == 3
    assert len(trace.steps) == 3
    assert trace.terminated_by == "success"


def test_fixed_pipeline_does_not_loop_on_failure(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _stage("no_such_tool"),
            _stage("no_such_tool_2"),
            _stage("no_such_tool_3"),
        ]
    )
    env = recording_env({})

    trace = run(task="all fail", env=env, llm=llm)

    assert llm.calls_made == 3
    assert trace.terminated_by == "success"


def test_fixed_pipeline_records_stage_names(scripted_llm, recording_env):
    llm = scripted_llm(
        [_stage("ls", path="."), _stage("ls", path="."), _stage("ls", path=".")]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="anything", env=env, llm=llm)

    assert [s.stage for s in trace.steps] == ["locate", "act", "verify"]
