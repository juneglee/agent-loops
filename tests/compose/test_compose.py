from agent_loops.compose.hierarchical import hierarchical
from agent_loops.loops import react, single_call
from tests.conftest import RecordingEnv, ScriptedLLM


def _plan(*items):
    return {"tool_calls": None, "text": "\n".join(f"- {t}" for t in items)}


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def test_planner_delegates_each_subtask_to_the_worker_loop():
    run = hierarchical(react, worker_kwargs={"max_steps": 4})
    llm = ScriptedLLM(
        [
            _plan("list folder a", "list folder b"),
            _tc("ls", path="a"),
            _txt("Task completed"),
            _tc("ls", path="b"),
            _txt("Task completed"),
            _txt("Final: collected everything"),
        ]
    )
    env = RecordingEnv({"ls": lambda path: f"{path}/file.txt"})

    trace = run(task="collect the listings of folders a and b", env=env, llm=llm)

    assert trace.terminated_by == "success"
    assert [name for name, _ in env.executed] == ["ls", "ls"]
    assert llm.calls_made == 6
    assert trace.n_llm_calls == 6
    assert trace.n_tool_calls == 2


def test_replanner_sees_subtask_results_between_rounds():
    run = hierarchical(react, worker_kwargs={"max_steps": 4})
    llm = ScriptedLLM(
        [
            _plan("list folder a"),
            _tc("ls", path="a"),
            _txt("Task completed"),
            _txt("Final: done"),
        ]
    )
    env = RecordingEnv({"ls": lambda path: f"{path}/file.txt"})

    run(task="t", env=env, llm=llm)

    replan_prompt = str(llm.prompts[3])
    assert "list folder a" in replan_prompt
    assert "file.txt" in replan_prompt


def test_worker_can_be_swapped_without_touching_the_planner():
    run = hierarchical(single_call)
    llm = ScriptedLLM(
        [
            _plan("list folder a"),
            _tc("ls", path="a"),
            _txt("Final: done"),
        ]
    )
    env = RecordingEnv({"ls": lambda path: f"{path}/file.txt"})

    trace = run(task="t", env=env, llm=llm)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3


def test_unparseable_first_plan_is_a_parse_failure():
    run = hierarchical(react)
    llm = ScriptedLLM([_txt("Hmm... I am not sure what to do.")])
    trace = run(task="t", env=RecordingEnv({}), llm=llm)
    assert trace.terminated_by == "parse_fail" and trace.parse_ok is False


def test_round_budget_stops_a_planner_that_never_finishes():
    run = hierarchical(single_call, max_rounds=2)
    llm = ScriptedLLM(
        [
            _plan("list folder a"),
            _tc("ls", path="a"),
            _plan("list folder a"),
            _tc("ls", path="a"),
        ]
    )
    trace = run(task="t", env=RecordingEnv({"ls": lambda path: "x"}), llm=llm)
    assert trace.terminated_by == "max_steps"
