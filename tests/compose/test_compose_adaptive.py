from agent_loops.compose.adaptive import adaptive
from agent_loops.loops import react, single_call
from tests.conftest import RecordingEnv, ScriptedLLM


def _plan(*items):
    return {"tool_calls": None, "text": "\n".join(f"- {t}" for t in items)}


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def _bad(**_kw):
    raise RuntimeError("no such path")


def test_success_means_no_decomposition_at_all():
    run = adaptive(react, worker_kwargs={"max_steps": 4})
    llm = ScriptedLLM([_tc("ls", path="a"), _txt("Task completed")])
    env = RecordingEnv({"ls": lambda path: f"{path}/file.txt"})

    trace = run(task="list folder a", env=env, llm=llm)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 2


def test_failure_triggers_decomposition_and_the_planner_sees_the_error():
    run = adaptive(react, worker_kwargs={"max_steps": 4})
    llm = ScriptedLLM(
        [
            _tc("bad", path="a"),
            _txt("Task completed"),
            _plan("list folder a"),
            _tc("ls", path="a"),
            _txt("Task completed"),
        ]
    )
    env = RecordingEnv({"ls": lambda path: f"{path}/file.txt", "bad": _bad})

    trace = run(task="list folder a", env=env, llm=llm)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 5
    assert [name for name, _ in env.executed] == ["bad", "ls"]
    assert "no such path" in str(llm.prompts[2])
    assert llm.kwargs[2].get("want") == "text"


def test_worker_that_never_declares_success_counts_as_failure():
    run = adaptive(react, max_depth=1, worker_kwargs={"max_steps": 1})
    llm = ScriptedLLM(
        [
            _tc("ls", path="a"),
            _plan("list folder a"),
            _tc("ls", path="a"),
        ]
    )
    env = RecordingEnv({"ls": lambda path: f"{path}/file.txt"})

    trace = run(task="list folder a", env=env, llm=llm)

    assert trace.terminated_by == "max_depth"


def test_unparseable_decomposition_is_a_parse_failure():
    run = adaptive(single_call)
    llm = ScriptedLLM(
        [
            _tc("bad", path="a"),
            _txt("Hmm... I am not sure."),
        ]
    )
    env = RecordingEnv({"bad": _bad})

    trace = run(task="t", env=env, llm=llm)

    assert trace.terminated_by == "parse_fail" and trace.parse_ok is False
