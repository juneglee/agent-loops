from agent_loops.loops.adapt import run


def _exec(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _say(text):
    return {"tool_calls": None, "text": text}


def _decompose(*subtasks, op=None):
    lines = [f"- {s}" for s in subtasks]
    if op:
        lines.append(f"Operator: {op}")
    return {"tool_calls": None, "text": "\n".join(lines)}


def _boom(**_):
    raise RuntimeError("no such file")


DONE = _say("Task completed")
FAILED = _say("Task failed: cannot do this as stated")


def test_adapt_does_not_decompose_on_success(scripted_llm, recording_env):
    llm = scripted_llm([_exec("ls", path="."), DONE])
    env = recording_env({"ls": lambda path: "a.txt"})

    trace = run(task="list", env=env, llm=llm, max_depth=3)

    assert llm.calls_made == 2
    assert trace.terminated_by == "success"
    assert all(k.get("want") != "text" for k in llm.kwargs)


def test_adapt_executor_self_declared_completion_is_success(
    scripted_llm, recording_env
):
    llm = scripted_llm([DONE])
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_depth=3)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 1


def test_adapt_executor_is_multi_step(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _exec("ls", path="a"),
            _exec("ls", path="b"),
            _say("checked both"),
            DONE,
        ]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="both", env=env, llm=llm, max_depth=3, max_steps=5)

    assert trace.terminated_by == "success"
    assert [n for n, _ in env.executed] == ["ls", "ls"]
    assert llm.calls_made == 4


def test_adapt_tool_error_alone_does_not_trigger_decomposition(
    scripted_llm, recording_env
):
    llm = scripted_llm([_exec("bad"), _exec("ls", path="."), DONE])
    env = recording_env({"bad": _boom, "ls": lambda path: "ok"})

    trace = run(task="t", env=env, llm=llm, max_depth=3)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3
    assert all(k.get("want") != "text" for k in llm.kwargs)


def test_adapt_decomposes_only_after_executor_declares_failure(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _exec("bad"),
            FAILED,
            _decompose("sub1", "sub2"),
            _exec("ls", path="a"),
            DONE,
            _exec("ls", path="b"),
            DONE,
        ]
    )
    env = recording_env({"bad": _boom, "ls": lambda path: path})

    trace = run(task="hard task", env=env, llm=llm, max_depth=3)

    assert trace.terminated_by == "success"
    assert [n for n, _ in env.executed] == ["bad", "ls", "ls"]
    assert llm.kwargs[2].get("want") == "text"


def test_adapt_planner_sees_only_the_task_not_the_failed_trajectory(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _exec("bad"),
            FAILED,
            _decompose("sub1"),
            DONE,
        ]
    )
    env = recording_env({"bad": _boom})

    run(task="hard task", env=env, llm=llm, max_depth=3)

    planner_prompt = str(llm.prompts[2])
    assert "hard task" in planner_prompt
    assert "no such file" not in planner_prompt
    assert "cannot do this as stated" not in planner_prompt
    assert llm.prompts[2] != llm.prompts[0]


def test_adapt_propagates_previous_subtask_steps_into_next_subtask(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            FAILED,
            _decompose("A", "B", op="And"),
            _exec("ls", path="a"),
            DONE,
            _exec("ls", path="b"),
            DONE,
        ]
    )
    env = recording_env({"ls": lambda path: f"listed {path}"})

    trace = run(task="t", env=env, llm=llm, max_depth=3)

    assert trace.terminated_by == "success"
    first_of_a = str(llm.prompts[2])
    first_of_b = str(llm.prompts[4])
    assert "carried" not in first_of_a
    assert "carried" in first_of_b
    assert "listed a" in first_of_b
    assert "'tool': 'ls'" in first_of_b


def test_adapt_does_not_propagate_steps_of_a_failed_or_branch(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            FAILED,
            _decompose("A", "B", op="Or"),
            _exec("ls", path="a"),
            _say("Task failed: A does not work"),
            DONE,
        ]
    )
    env = recording_env({"ls": lambda path: f"listed {path}"})

    trace = run(task="t", env=env, llm=llm, max_depth=1)

    assert trace.terminated_by == "success"
    assert "listed a" not in str(llm.prompts[4])


def test_adapt_patience_ends_the_trial_after_consecutive_non_progress(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [_say("hmm"), _exec("bad"), _say("hmm"), _decompose("sub"), DONE]
    )
    env = recording_env({"bad": _boom})

    trace = run(task="t", env=env, llm=llm, max_depth=3, max_steps=10, patience=3)

    assert trace.terminated_by == "success"
    assert llm.kwargs[3].get("want") == "text"
    assert llm.calls_made == 5


def test_adapt_patience_counter_resets_on_progress(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _say("hmm"),
            _say("hmm"),
            _exec("ls", path="."),
            _say("hmm"),
            _say("hmm"),
            DONE,
        ]
    )
    env = recording_env({"ls": lambda path: "a"})

    trace = run(task="t", env=env, llm=llm, max_depth=3, max_steps=10, patience=3)

    assert trace.terminated_by == "success"
    assert all(k.get("want") != "text" for k in llm.kwargs)


def test_adapt_default_patience_is_eight(scripted_llm, recording_env):
    llm = scripted_llm([_say("hmm")] * 7 + [DONE])
    trace = run(task="t", env=recording_env({}), llm=llm, max_depth=3, max_steps=20)
    assert trace.terminated_by == "success"

    llm = scripted_llm([_say("hmm")] * 8 + [_decompose("sub"), DONE])
    trace = run(task="t", env=recording_env({}), llm=llm, max_depth=3, max_steps=20)
    assert trace.terminated_by == "success"
    assert llm.kwargs[8].get("want") == "text"


def test_adapt_executor_step_budget_exhaustion_counts_as_failure(
    scripted_llm, recording_env
):
    llm = scripted_llm([_say("thought 1"), _say("thought 2"), _decompose("sub"), DONE])
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_depth=3, max_steps=2)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 4


def test_adapt_and_operator_requires_every_subtask(scripted_llm, recording_env):
    llm = scripted_llm([FAILED, _decompose("A", "B", op="And"), DONE, DONE])
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_depth=3)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 4


def test_adapt_or_operator_succeeds_when_any_subtask_succeeds(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            FAILED,
            _decompose("way A", "way B", op="Or"),
            _say("Task failed: A does not work"),
            DONE,
        ]
    )
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_depth=1)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 4


def test_adapt_stops_at_max_depth(scripted_llm, recording_env):
    llm = scripted_llm([FAILED, _decompose("sub")] * 4)
    env = recording_env({})

    trace = run(task="keeps failing", env=env, llm=llm, max_depth=2)

    assert trace.terminated_by == "max_depth"


def test_adapt_unreadable_decomposition_is_parse_fail(scripted_llm, recording_env):
    llm = scripted_llm([FAILED, _say("hmm... no idea")])
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_depth=3)

    assert trace.terminated_by == "parse_fail"


class _ProseLLM:
    def __init__(self, n: int, op: str) -> None:
        self.n, self.op, self.calls_made = n, op, 0

    def __call__(self, messages, **kw):
        self.calls_made += 1
        if kw.get("want") == "text":
            lines = [f"- sub {i}" for i in range(self.n)] + [f"Operator: {self.op}"]
            return {"tool_calls": None, "text": "\n".join(lines)}
        return {"tool_calls": None, "text": "still thinking"}


def test_adapt_total_call_budget_bounds_the_or_explosion(recording_env):
    llm = _ProseLLM(5, "Or")

    trace = run(
        task="t",
        env=recording_env({}),
        llm=llm,
        max_depth=3,
        max_steps=10,
        max_calls=30,
    )

    assert llm.calls_made <= 30
    assert trace.terminated_by == "max_steps"


def test_adapt_caps_subtasks_at_five(scripted_llm, recording_env):
    llm = scripted_llm(
        [FAILED, _decompose(*[f"sub{i}" for i in range(8)])] + [DONE] * 5
    )
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_depth=3)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 7


def test_adapt_ignores_bold_heading_lines_in_decomposition(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            FAILED,
            _say("**Step 1: check directory**\n- cd into it\n* ls to check\n**Note**"),
            DONE,
            DONE,
        ]
    )
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_depth=3)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 4


def test_adapt_decomposition_asks_for_self_contained_subtasks():
    from agent_loops.loops.adapt import _DECOMPOSE

    assert "from that line alone" in _DECOMPOSE
    assert "failure" not in _DECOMPOSE
