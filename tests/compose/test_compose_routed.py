from agent_loops.compose.routed import routed
from agent_loops.loops import react, single_call
from tests.conftest import RecordingEnv, ScriptedLLM


def _plan(*items):
    return {"tool_calls": None, "text": "\n".join(f"- {t}" for t in items)}


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def test_simple_verdict_skips_planning_entirely():
    run = routed(react, worker_kwargs={"max_steps": 4})
    llm = ScriptedLLM(
        [
            _txt("simple"),
            _tc("ls", path="a"),
            _txt("Task completed"),
        ]
    )
    env = RecordingEnv({"ls": lambda path: f"{path}/file.txt"})

    trace = run(task="list folder a", env=env, llm=llm)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3
    assert llm.kwargs[0].get("want") == "text"


def test_complex_verdict_takes_the_hierarchical_path():
    run = routed(react, worker_kwargs={"max_steps": 4})
    llm = ScriptedLLM(
        [
            _txt("complex"),
            _plan("list folder a"),
            _tc("ls", path="a"),
            _txt("Task completed"),
            _txt("Final: done"),
        ]
    )
    env = RecordingEnv({"ls": lambda path: f"{path}/file.txt"})

    trace = run(
        task="organize, compress and verify the listing of folder a", env=env, llm=llm
    )

    assert trace.terminated_by == "success"
    assert llm.calls_made == 5
    assert [name for name, _ in env.executed] == ["ls"]


def test_gate_gibberish_is_a_parse_failure_and_worker_never_runs():
    run = routed(react)
    llm = ScriptedLLM([_txt("Well, that is a hard question.")])
    env = RecordingEnv({})

    trace = run(task="t", env=env, llm=llm)

    assert trace.terminated_by == "parse_fail" and trace.parse_ok is False
    assert llm.calls_made == 1 and env.executed == []


def test_worker_can_be_swapped():
    run = routed(single_call)
    llm = ScriptedLLM([_txt("Simple."), _tc("ls", path="a")])
    env = RecordingEnv({"ls": lambda path: "x"})

    trace = run(task="t", env=env, llm=llm)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 2
