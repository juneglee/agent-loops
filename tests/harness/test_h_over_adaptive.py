from types import SimpleNamespace

from agent_loops.compose.adaptive import adaptive
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


def _violated_once():
    seen = []

    def check(env, arguments, observation):
        if seen:
            return None
        seen.append(True)
        return "src still exists"

    return check


TASKS = [{"task": "move a", "status": "pending"}]


def test_todo_state_survives_the_decomposition_after_a_failed_direct_attempt(
    scripted_llm, recording_env
):
    run = apply(_stack(adaptive(react, worker_kwargs={"max_steps": 6})), [todo])
    llm = scripted_llm(
        [
            _tc("update_todo", tasks=TASKS),
            _txt("Task failed: cannot move"),
            _plan("move a"),
            _tc("mv", source="a", destination="b"),
            _txt("Task completed"),
        ]
    )
    env = recording_env({"mv": lambda source, destination: "moved"})

    trace = run(task="move a", env=env, llm=llm)

    assert run.NAME == "adaptive+react+todo"
    assert trace.terminated_by == "success"
    assert [n for n, _ in env.executed] == ["mv"]
    assert "move a" in str(llm.prompts[2]) and "pending" in str(llm.prompts[2])


def test_verifier_failure_counts_as_a_failed_attempt_and_triggers_decomposition(
    scripted_llm, recording_env
):
    run = apply(
        _stack(adaptive(react, worker_kwargs={"max_steps": 6})),
        [verifier({"mv": _violated_once()})],
    )
    llm = scripted_llm(
        [
            _tc("mv", source="a", destination="b"),
            _txt("Task completed"),
            _plan("move a"),
            _tc("mv", source="a", destination="b"),
            _txt("Task completed"),
        ]
    )
    env = recording_env({"mv": lambda source, destination: "moved"})

    trace = run(task="move a", env=env, llm=llm)

    assert [n for n, _ in env.executed] == ["mv", "mv"]
    assert "src still exists" in str(llm.prompts[2])
    assert llm.calls_made == 5
    assert trace.terminated_by == "success"
