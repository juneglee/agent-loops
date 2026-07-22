import inspect

from agent_loops.loops.llm_compiler import JOINER_INSTRUCTION, _dependency_waves, run


def _text(text: str) -> dict:
    return {"tool_calls": None, "text": text}


def test_llm_compiler_resolves_dag_waves_and_dependencies(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _text("#E1 = a[]\n#E2 = b[]\n#E3 = c[value=#E1]"),
            _text("Final: done"),
        ]
    )
    env = recording_env(
        {
            "a": lambda: "A-out",
            "b": lambda: "B-out",
            "c": lambda value: value,
        }
    )

    trace = run(task="t", env=env, llm=llm)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 2
    assert trace.n_llm_calls == 2
    assert env.executed[2] == ("c", {"value": "A-out"})
    stages = [step.stage for step in trace.steps if step.tool_name]
    assert stages == ["wave1", "wave1", "wave2"]


def test_llm_compiler_rejects_forward_reference():
    assert _dependency_waves([("#E1", "a", {"value": "#E2"}), ("#E2", "b", {})]) is None


def test_llm_compiler_default_replan_budget_is_one():
    assert inspect.signature(run).parameters["max_replans"].default == 1


def test_llm_compiler_joiner_sees_plan_and_observations(scripted_llm, recording_env):
    llm = scripted_llm([_text("#E1 = a[]"), _text("Final: done")])
    env = recording_env({"a": lambda: "A-out"})

    run(task="t", env=env, llm=llm)

    joiner_prompt = str(llm.prompts[1][-1]["content"])
    assert llm.kwargs[1] == {"want": "text"}
    assert "#E1 = a[]" in joiner_prompt
    assert "Observation: A-out" in joiner_prompt
    assert JOINER_INSTRUCTION in joiner_prompt


def test_llm_compiler_joiner_replan_feeds_thought_to_planner(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _text("#E1 = a[]"),
            _text("Replan: b is still missing"),
            _text("#E2 = b[]"),
            _text("Final: done"),
        ]
    )
    env = recording_env({"a": lambda: "A", "b": lambda: "B"})

    trace = run(task="t", env=env, llm=llm, max_replans=1)

    assert trace.terminated_by == "success"
    assert [name for name, _ in env.executed] == ["a", "b"]
    assert llm.calls_made == 4
    replan_prompt = str(llm.prompts[2])
    assert "b is still missing" in replan_prompt
    assert "#E1 = a[]" in replan_prompt


def test_llm_compiler_replan_verdict_with_budget_exhausted_is_max_steps(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _text("#E1 = a[]"),
            _text("Replan: more needed"),
            _text("#E2 = b[]"),
            _text("Replan: still more"),
        ]
    )
    env = recording_env({"a": lambda: "A", "b": lambda: "B"})

    trace = run(task="t", env=env, llm=llm, max_replans=1)

    assert trace.terminated_by == "max_steps"
    assert llm.calls_made == 4


def test_llm_compiler_unreadable_joiner_output_is_parse_fail(
    scripted_llm, recording_env
):
    llm = scripted_llm([_text("#E1 = a[]"), _text("Hmm, I am not sure.")])
    env = recording_env({"a": lambda: "A"})

    trace = run(task="t", env=env, llm=llm)

    assert trace.terminated_by == "parse_fail"
    assert trace.parse_ok is False


def test_llm_compiler_replan_may_reference_variables_from_earlier_rounds(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _text("#E1 = a[]"),
            _text("Replan: need c"),
            _text("#E2 = c[value=#E1]"),
            _text("Final: done"),
        ]
    )
    env = recording_env({"a": lambda: "A-out", "c": lambda value: f"got {value}"})

    trace = run(task="t", env=env, llm=llm, max_replans=1)

    assert trace.terminated_by == "success"
    assert env.executed == [("a", {}), ("c", {"value": "A-out"})]


def test_llm_compiler_skips_tasks_whose_dependency_failed(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _text("#E1 = bad[]\n#E2 = c[value=#E1]"),
            _text("Replan: bad failed"),
            _text("#E3 = d[]"),
            _text("Final: done"),
        ]
    )
    env = recording_env(
        {
            "bad": lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            "c": lambda value: "never",
            "d": lambda: "D",
        }
    )

    trace = run(task="t", env=env, llm=llm, max_replans=1)

    assert [name for name, _ in env.executed] == ["bad", "d"]
    assert "skipped" in str(llm.prompts[1][-1]["content"])
    assert "skipped" in str(llm.prompts[2])
    assert trace.terminated_by == "success"


def test_llm_compiler_recovers_a_failed_variable_when_replan_reexecutes_it(
    scripted_llm, recording_env
):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first time fails")
        return "A-out"

    llm = scripted_llm(
        [
            _text("#E1 = a[]"),
            _text("Replan: retry a"),
            _text("#E1 = a[]\n#E2 = c[value=#E1]"),
            _text("Final: done"),
        ]
    )
    env = recording_env({"a": flaky, "c": lambda value: f"got {value}"})

    trace = run(task="t", env=env, llm=llm, max_replans=1)

    assert [name for name, _ in env.executed] == ["a", "a", "c"]
    assert env.executed[-1] == ("c", {"value": "A-out"})
    assert trace.terminated_by == "success"
