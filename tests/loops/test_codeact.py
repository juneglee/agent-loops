from agent_loops.loops.codeact import run


def _code(src):
    return {"tool_calls": [{"name": "execute_code", "arguments": {"code": src}}]}


def _final(text="Final: done"):
    return {"tool_calls": None, "text": text}


def test_codeact_compresses_multiple_tool_uses_into_one_llm_call(
    scripted_llm, recording_env
):
    executed = []

    def execute_code(code):
        executed.extend(["a", "b", "c"])
        return "\n".join(executed)

    llm = scripted_llm([_code("for p in ['a','b','c']: ls(p)"), _final()])
    env = recording_env({"execute_code": execute_code})

    trace = run(task="all three", env=env, llm=llm, max_steps=5)

    assert llm.calls_made == 2
    assert executed == ["a", "b", "c"]
    assert trace.terminated_by == "success"


def test_codeact_uses_stdout_as_observation(scripted_llm, recording_env):
    llm = scripted_llm([_code("print('result')"), _final()])
    env = recording_env({"execute_code": lambda code: "result"})

    trace = run(task="print", env=env, llm=llm, max_steps=5)

    assert trace.steps[0].observation["output"] == "result"


def test_codeact_retries_after_execution_error(scripted_llm, recording_env):
    calls = {"n": 0}

    def execute_code(code):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("NameError: ls is not defined")
        return "recovered"

    llm = scripted_llm(
        [
            _code("ls('a')"),
            _code("from tools import ls; ls('a')"),
            _final(),
        ]
    )
    env = recording_env({"execute_code": execute_code})

    trace = run(task="fix it", env=env, llm=llm, max_steps=5)

    assert calls["n"] == 2
    assert trace.steps[0].observation["ok"] is False
    assert trace.steps[1].observation["ok"] is True


def test_codeact_final_marker_ends_the_episode_as_success(scripted_llm, recording_env):
    llm = scripted_llm([_final("Final: the file is a.txt")])
    env = recording_env({"execute_code": lambda code: ""})

    trace = run(task="which file", env=env, llm=llm, max_steps=5)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 1


def test_codeact_text_without_code_and_without_marker_is_a_thought_and_continues(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            {"tool_calls": None, "text": "I should list the directory first"},
            _code("print(ls())"),
            _final("Final: a.txt"),
        ]
    )
    env = recording_env({"execute_code": lambda code: "a.txt"})

    trace = run(task="list", env=env, llm=llm, max_steps=5)

    assert llm.calls_made == 3
    assert trace.terminated_by == "success"
    assert llm.prompts[1][-1]["role"] == "assistant"
    assert llm.prompts[1][-1]["content"]["text"] == "I should list the directory first"


def test_codeact_thoughts_only_until_budget_ends_in_max_steps_not_success(
    scripted_llm, recording_env
):
    llm = scripted_llm([{"tool_calls": None, "text": "thinking"}] * 3)
    env = recording_env({"execute_code": lambda code: ""})

    trace = run(task="stall", env=env, llm=llm, max_steps=3)

    assert trace.terminated_by == "max_steps"
    assert llm.calls_made == 3


def test_codeact_observation_carries_remaining_step_countdown(
    scripted_llm, recording_env
):
    llm = scripted_llm([_code("print(1)"), _code("print(2)"), _final()])
    env = recording_env({"execute_code": lambda code: "ok"})

    run(task="count", env=env, llm=llm, max_steps=4)

    first = llm.prompts[1][-1]
    second = llm.prompts[2][-1]
    assert first["role"] == "tool"
    assert first["content"]["notice"] == "3 steps remaining"
    assert second["content"]["notice"] == "2 steps remaining"
    assert first["content"]["output"] == "ok"


def test_codeact_rejects_non_code_tool_as_observation_and_continues(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "ls", "arguments": {}}]},
            _code("print(ls())"),
            _final(),
        ]
    )
    env = recording_env(
        {"execute_code": lambda code: "a.txt", "ls": lambda: "should not run"}
    )

    trace = run(task="ls", env=env, llm=llm, max_steps=5)

    assert env.executed == [("execute_code", {"code": "print(ls())"})]
    assert trace.steps[0].stage == "rejected"
    assert trace.steps[0].observation["ok"] is False
    assert trace.terminated_by == "success"
