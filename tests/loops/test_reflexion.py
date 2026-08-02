from agent_loops.loops.reflexion import _REFLECT_INSTRUCTION, _REFLECTION_HEADER, run

_FAIL = {"tool_calls": None, "text": "Task failed: cannot do it"}
_DONE = {"tool_calls": None, "text": "Task completed"}
_LS_A = {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]}
_LS_B = {"tool_calls": [{"name": "ls", "arguments": {"path": "b"}}]}


def _text(prompt) -> str:
    return str(prompt)


def test_reflexion_injects_reflection_into_next_trial_with_header(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _FAIL,
            {"tool_calls": None, "text": "I should have searched the name first"},
            _DONE,
        ]
    )
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_trials=2, max_steps=2)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3
    assert "searched the name first" not in _text(llm.prompts[0])
    injected = llm.prompts[2]
    system_texts = [
        m["content"]
        for m in injected
        if m.get("role") == "system" and isinstance(m["content"], str)
    ]
    assert any(
        s.startswith(_REFLECTION_HEADER)
        and "Reflections:\n- I should have searched the name first" in s
        for s in system_texts
    )


def test_reflexion_reflection_prompt_asks_for_diagnosis_and_new_plan(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _FAIL,
            {"tool_calls": None, "text": "reflection"},
            _DONE,
        ]
    )
    env = recording_env({})

    run(task="t", env=env, llm=llm, max_trials=2, max_steps=2)

    reflection_prompt = _text(llm.prompts[1])
    assert _REFLECT_INSTRUCTION in reflection_prompt
    assert "diagnose a possible reason for failure" in _REFLECT_INSTRUCTION
    assert "devise a new, concise, high level plan" in _REFLECT_INSTRUCTION
    assert llm.kwargs[1].get("want") == "text"


def test_reflexion_injects_only_last_memory_size_reflections(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _FAIL,
            {"tool_calls": None, "text": "reflection one"},
            _FAIL,
            {"tool_calls": None, "text": "reflection two"},
            _FAIL,
            {"tool_calls": None, "text": "reflection three"},
            _FAIL,
            {"tool_calls": None, "text": "reflection four"},
            _DONE,
        ]
    )
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_trials=5, max_steps=1, memory_size=3)

    assert trace.terminated_by == "success"
    fifth_trial_prompt = _text(llm.prompts[8])
    assert "reflection one" not in fifth_trial_prompt
    assert "reflection two" in fifth_trial_prompt
    assert "reflection three" in fifth_trial_prompt
    assert "reflection four" in fifth_trial_prompt
    fourth_trial_prompt = _text(llm.prompts[6])
    assert "reflection one" in fourth_trial_prompt


def test_reflexion_memory_size_one_keeps_only_latest_reflection(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _FAIL,
            {"tool_calls": None, "text": "reflection one"},
            _FAIL,
            {"tool_calls": None, "text": "reflection two"},
            _DONE,
        ]
    )
    env = recording_env({})

    run(task="t", env=env, llm=llm, max_trials=3, max_steps=1, memory_size=1)

    third_trial_prompt = _text(llm.prompts[4])
    assert "reflection one" not in third_trial_prompt
    assert "reflection two" in third_trial_prompt


def test_reflexion_same_action_twice_in_a_row_ends_trial_as_failure(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _LS_A,
            _LS_A,
            {"tool_calls": None, "text": "I kept listing the same path"},
            _DONE,
        ]
    )
    env = recording_env({"ls": lambda path: "a.txt"})

    trace = run(task="t", env=env, llm=llm, max_trials=2, max_steps=10)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 4
    assert env.executed == [("ls", {"path": "a"}), ("ls", {"path": "a"})]
    assert "listing the same path" in _text(llm.prompts[3])


def test_reflexion_repeated_action_on_last_trial_ends_with_max_trials(
    scripted_llm, recording_env
):
    llm = scripted_llm([_LS_A, _LS_A])
    env = recording_env({"ls": lambda path: "a.txt"})

    trace = run(task="t", env=env, llm=llm, max_trials=1, max_steps=10)

    assert trace.terminated_by == "max_trials"
    assert llm.calls_made == 2


def test_reflexion_different_consecutive_actions_do_not_exhaust_trial(
    scripted_llm, recording_env
):
    llm = scripted_llm([_LS_A, _LS_B, _LS_A, _DONE])
    env = recording_env({"ls": lambda path: "x"})

    trace = run(task="t", env=env, llm=llm, max_trials=1, max_steps=10)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 4


def test_reflexion_same_tool_with_different_arguments_is_not_a_repeat(
    scripted_llm, recording_env
):
    llm = scripted_llm([_LS_A, _LS_B, _DONE])
    env = recording_env({"ls": lambda path: "x"})

    trace = run(task="t", env=env, llm=llm, max_trials=1, max_steps=10)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3


def test_reflexion_respects_trial_budget_and_skips_last_reflection(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _FAIL,
            {"tool_calls": None, "text": "reflection 1"},
            _FAIL,
        ]
    )
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_trials=2, max_steps=1)

    assert trace.terminated_by == "max_trials"
    assert llm.calls_made == 3


def test_reflexion_succeeds_first_trial_without_reflection(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}]},
            _DONE,
        ]
    )
    env = recording_env({"ls": lambda path: "a.txt"})

    trace = run(task="t", env=env, llm=llm, max_trials=3, max_steps=5)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 2


def test_reflexion_treats_tool_error_as_failure_even_if_model_claims_completion(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            {"tool_calls": [{"name": "rm", "arguments": {"path": "x"}}]},
            _DONE,
            {"tool_calls": None, "text": "I should have checked the path first"},
            _DONE,
        ]
    )
    env = recording_env(
        {"rm": lambda path: (_ for _ in ()).throw(FileNotFoundError(path))}
    )

    trace = run(task="t", env=env, llm=llm, max_trials=2, max_steps=3)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 4


def test_reflexion_thought_only_step_continues_within_a_trial(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            {"tool_calls": None, "text": "Let me list the directory first"},
            {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}]},
            {"tool_calls": None, "text": "Final: a.txt"},
        ]
    )
    env = recording_env({"ls": lambda path: "a.txt"})

    trace = run(task="t", env=env, llm=llm, max_trials=2, max_steps=5)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3
    assert "list the directory first" in _text(llm.prompts[1])


def test_reflexion_trial_without_final_declaration_fails_at_step_budget(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            {"tool_calls": None, "text": "thought 1"},
            {"tool_calls": None, "text": "thought 2"},
        ]
    )
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, max_trials=1, max_steps=2)

    assert trace.terminated_by == "max_trials"
    assert llm.calls_made == 2


def test_reflexion_calls_env_reset_between_trials(scripted_llm, recording_env):
    class ResettableEnv(recording_env):
        def __init__(self) -> None:
            super().__init__({})
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

    llm = scripted_llm([_FAIL, {"tool_calls": None, "text": "r"}, _DONE])
    env = ResettableEnv()

    run(task="t", env=env, llm=llm, max_trials=2, max_steps=1)

    assert env.resets == 1
