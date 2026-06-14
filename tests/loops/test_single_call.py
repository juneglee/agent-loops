from agent_loops.loops.single_call import run


def test_single_call_makes_exactly_one_llm_call(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}]},
        ]
    )
    env = recording_env({"ls": lambda path: "a.txt\nb.txt"})

    trace = run(task="list the files", env=env, llm=llm)

    assert trace.n_llm_calls == 1


def test_single_call_does_not_call_llm_again_after_tool_result(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}]},
            {"tool_calls": [{"name": "ls", "arguments": {"path": "/other"}}]},
        ]
    )
    env = recording_env({"ls": lambda path: "a.txt"})

    trace = run(task="list files", env=env, llm=llm)

    assert llm.calls_made == 1
    assert len(trace.steps) == 1


def test_single_call_executes_the_requested_tool(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "ls", "arguments": {"path": "docs"}}]},
        ]
    )
    seen = []
    env = recording_env({"ls": lambda path: seen.append(path) or "ok"})

    run(task="list docs", env=env, llm=llm)

    assert seen == ["docs"]


def test_single_call_records_parse_failure_without_crashing(
    scripted_llm, recording_env
):
    llm = scripted_llm([{"raw": "just a plain text answer", "tool_calls": None}])
    env = recording_env({})

    trace = run(task="anything", env=env, llm=llm)

    assert trace.parse_ok is False
    assert trace.terminated_by == "parse_fail"


def test_single_call_executes_every_returned_call_in_order(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            {
                "tool_calls": [
                    {"name": "ls", "arguments": {"path": "a"}},
                    {"name": "ls", "arguments": {"path": "b"}},
                ]
            }
        ]
    )
    env = recording_env({"ls": lambda path: path})
    trace = run(task="t", env=env, llm=llm)
    assert llm.calls_made == 1
    assert [a["path"] for _, a in env.executed] == ["a", "b"]
    assert trace.n_llm_calls == 1 and trace.n_tool_calls == 2
