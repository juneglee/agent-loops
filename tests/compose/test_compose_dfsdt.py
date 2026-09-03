from agent_loops.compose.hierarchical import hierarchical
from agent_loops.compose.routed import routed
from agent_loops.loops import dfsdt
from tests.conftest import RecordingEnv, ScriptedLLM


def _plan(*items):
    return {"tool_calls": None, "text": "\n".join(f"- {i}" for i in items)}


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def _fail(**_kw):
    raise RuntimeError("no such path")


def test_planner_delegates_each_subtask_to_dfsdt():
    env = RecordingEnv({"ls": lambda path: path})
    llm = ScriptedLLM(
        [
            _plan("list a", "list b"),
            _tc("ls", path="a"),
            _txt("Final: a listed"),
            _tc("ls", path="b"),
            _txt("Final: b listed"),
            _txt("Final: both listed"),
        ]
    )
    run = hierarchical(dfsdt, worker_kwargs={"breadth": 2, "max_calls": 12})
    trace = run(task="list a and b", env=env, llm=llm)

    assert run.NAME == "planner+dfsdt"
    assert env.executed == [("ls", {"path": "a"}), ("ls", {"path": "b"})]
    assert trace.terminated_by == "success"
    assert trace.parse_ok is True
    assert llm.calls_made == 6


def test_give_up_inside_a_subtask_branches_to_a_sibling_within_the_composition():
    env = RecordingEnv({"ls": lambda path: path, "bad": _fail})
    llm = ScriptedLLM(
        [
            _plan("list a"),
            _tc("bad", path="a"),
            _txt("Give up: that path does not exist"),
            _tc("ls", path="a"),
            _txt("Final: a listed"),
            _txt("Final: done"),
        ]
    )
    run = hierarchical(dfsdt, worker_kwargs={"breadth": 2, "max_calls": 12})
    trace = run(task="list a", env=env, llm=llm)

    assert env.executed == [("bad", {"path": "a"}), ("ls", {"path": "a"})]
    assert trace.terminated_by == "success"
    assert any(step.tool_name == "ls" for step in trace.steps)


def test_worker_parse_failure_stays_visible_after_the_planner_finishes():
    env = RecordingEnv({"ls": lambda path: path})
    garbage = _txt("I am not sure what to do here.")
    llm = ScriptedLLM([_plan("list a"), *([garbage] * 12), _txt("Final: done")])
    run = hierarchical(dfsdt, worker_kwargs={"breadth": 2, "max_calls": 12})
    trace = run(task="list a", env=env, llm=llm)

    assert env.executed == []
    assert trace.parse_ok is False
    assert trace.terminated_by == "no_action"
    assert llm.calls_made == 14


def test_routed_simple_verdict_runs_dfsdt_directly():
    env = RecordingEnv({"ls": lambda path: path})
    llm = ScriptedLLM(
        [
            _txt("simple"),
            _tc("ls", path="a"),
            _txt("Final: a listed"),
        ]
    )
    run = routed(dfsdt, worker_kwargs={"breadth": 2, "max_calls": 12})
    trace = run(task="list a", env=env, llm=llm)

    assert run.NAME == "routed+dfsdt"
    assert env.executed == [("ls", {"path": "a"})]
    assert trace.terminated_by == "success"
    assert llm.calls_made == 3


def test_worker_budget_is_the_composition_kwargs_not_the_core_default():
    env = RecordingEnv({"ls": lambda path: path})
    llm = ScriptedLLM(
        [_plan("list a")] + [_tc("ls", path="a")] * 12 + [_txt("Final: done")]
    )
    run = hierarchical(dfsdt, worker_kwargs={"breadth": 2, "max_calls": 3})
    run(task="list a", env=env, llm=llm)

    assert len(env.executed) == 3
