from agent_loops.harness import apply
from agent_loops.harness.todo import TODO_TOOL_SCHEMA, todo
from agent_loops.loops import react, single_call


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


TASKS = [
    {"task": "check folder a", "status": "pending"},
    {"task": "write the report", "status": "pending"},
]


def test_update_todo_is_intercepted_and_never_reaches_the_backend(
    scripted_llm, recording_env
):
    run = apply(react, [todo])
    llm = scripted_llm(
        [
            _tc("update_todo", tasks=TASKS),
            _tc("ls", path="a"),
            _txt("Task completed"),
        ]
    )
    env = recording_env({"ls": lambda path: "a/file.txt"})

    trace = run(task="check a and report", env=env, llm=llm)

    assert trace.terminated_by == "success"
    assert [n for n, _ in env.executed] == ["ls"]


def test_current_todo_is_injected_into_every_call_after_update(
    scripted_llm, recording_env
):
    run = apply(react, [todo])
    llm = scripted_llm(
        [
            _tc("update_todo", tasks=TASKS),
            _tc("ls", path="a"),
            _txt("Task completed"),
        ]
    )
    run(task="t", env=recording_env({"ls": lambda path: "x"}), llm=llm)

    assert "check folder a" not in str(llm.prompts[0])
    assert "check folder a" in str(llm.prompts[1])
    assert "check folder a" in str(llm.prompts[2])


def test_second_update_replaces_the_list_deep_agents_style(scripted_llm, recording_env):
    run = apply(react, [todo])
    done = [{"task": "check folder a", "status": "completed"}]
    llm = scripted_llm(
        [
            _tc("update_todo", tasks=TASKS),
            _tc("update_todo", tasks=done),
            _txt("Task completed"),
        ]
    )
    run(task="t", env=recording_env({}), llm=llm)

    injected = [
        m
        for m in llm.prompts[2]
        if isinstance(m, dict) and "Current TODO" in str(m.get("content", ""))
    ][-1]
    block = str(injected["content"])
    assert "completed" in block
    assert "write the report" not in block


def test_state_persists_across_turns_on_the_same_env_and_resets_on_a_new_env(
    scripted_llm, recording_env
):
    run = apply(single_call, [todo])
    env = recording_env({})

    run(task="turn 1", env=env, llm=scripted_llm([_tc("update_todo", tasks=TASKS)]))
    llm2 = scripted_llm([_txt("x")])
    run(task="turn 2", env=env, llm=llm2)
    assert "check folder a" in str(llm2.prompts[0])

    llm3 = scripted_llm([_txt("x")])
    run(task="new case", env=recording_env({}), llm=llm3)
    assert "check folder a" not in str(llm3.prompts[0])


def test_layer_adds_zero_llm_calls_and_leaves_loop_behavior_unchanged(
    scripted_llm, recording_env
):
    script = [_tc("ls", path="a"), _txt("Task completed")]

    bare_llm = scripted_llm(list(script))
    bare = react.run(
        task="t", env=recording_env({"ls": lambda path: "x"}), llm=bare_llm
    )

    layered_llm = scripted_llm(list(script))
    layered = apply(react, [todo])(
        task="t", env=recording_env({"ls": lambda path: "x"}), llm=layered_llm
    )

    assert layered_llm.calls_made == bare_llm.calls_made == 2
    assert layered.terminated_by == bare.terminated_by == "success"


def test_apply_names_the_stack_for_the_registry():
    assert apply(react, [todo]).NAME == "react+todo"
    assert apply(single_call, [todo]).NAME == "single_call+todo"


def test_layer_exposes_the_update_todo_schema_for_runner_wiring():
    fn = TODO_TOOL_SCHEMA["function"]
    assert fn["name"] == "update_todo"
    assert "tasks" in fn["parameters"]["properties"]
