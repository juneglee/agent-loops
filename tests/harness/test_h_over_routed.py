from types import SimpleNamespace

from agent_loops.compose.routed import routed
from agent_loops.harness import apply
from agent_loops.harness.todo import todo
from agent_loops.harness.verifier import verifier
from agent_loops.loops import react


def _stack(run):
    return SimpleNamespace(NAME=run.NAME, run=run)


def _plan(*items):
    return {"tool_calls": None, "text": "\n".join(f"- {i}" for i in items)}


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def _always_violated(env, arguments, observation):
    return "src still exists"


TASKS = [{"task": "list a", "status": "pending"}]


def test_simple_path_keeps_the_todo_tool_without_planning(scripted_llm, recording_env):
    run = apply(_stack(routed(react, worker_kwargs={"max_steps": 6})), [todo])
    llm = scripted_llm(
        [
            _txt("simple"),
            _tc("update_todo", tasks=TASKS),
            _tc("ls", path="a"),
            _txt("Task completed"),
        ]
    )
    env = recording_env({"ls": lambda path: "a/file.txt"})

    trace = run(task="list a", env=env, llm=llm)

    assert run.NAME == "routed+react+todo"
    assert trace.terminated_by == "success"
    assert [n for n, _ in env.executed] == ["ls"]
    assert llm.kwargs[0].get("want") == "text"
    assert "pending" in str(llm.prompts[2])


def test_complex_path_reports_the_verifier_failure_to_the_planner_as_not_ok(
    scripted_llm, recording_env
):
    run = apply(
        _stack(routed(react, worker_kwargs={"max_steps": 6})),
        [verifier({"mv": _always_violated})],
    )
    llm = scripted_llm(
        [
            _txt("complex"),
            _plan("move a to b"),
            _tc("mv", source="a", destination="b"),
            _txt("Task completed"),
            _txt("Final: done"),
        ]
    )
    env = recording_env({"mv": lambda source, destination: "moved"})

    trace = run(task="move a to b", env=env, llm=llm)

    obs = next(s.observation for s in trace.steps if s.tool_name == "mv")
    assert obs["ok"] is False and "src still exists" in obs["error"]
    assert "'ok': False" in str(llm.prompts[4])
