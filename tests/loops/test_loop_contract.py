import pytest

from agent_loops.bench import prompts
from agent_loops.loops import (
    adapt,
    codeact,
    dfsdt,
    fixed_pipeline,
    llm_compiler,
    plan_and_act,
    plan_and_execute,
    plan_and_solve,
    react,
    reflexion,
    rewoo,
    single_call,
)

LOOPS = [
    (single_call, {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}]}),
    (react, {"tool_calls": None, "text": "Final: done"}),
    (rewoo, {"tool_calls": None, "text": "#E1 = ls[path=.]"}),
    (plan_and_solve, {"tool_calls": None, "text": "1. ls[path=.]"}),
    (plan_and_execute, {"tool_calls": None, "text": ""}),
    (plan_and_act, {"tool_calls": None, "text": ""}),
    (adapt, {"tool_calls": None, "text": "Task completed"}),
    (codeact, {"tool_calls": None, "text": "done"}),
    (fixed_pipeline, {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}]}),
]


@pytest.mark.parametrize("module,first", LOOPS, ids=lambda x: getattr(x, "NAME", ""))
def test_loop_sends_shared_system_prompt_first(
    module, first, scripted_llm, recording_env
):
    llm = scripted_llm([first] * 6)
    env = recording_env({"ls": lambda path: "a.txt"})

    module.run(task="task", env=env, llm=llm)

    first_prompt = llm.prompts[0]
    assert first_prompt[0]["role"] == "system"
    assert first_prompt[0]["content"] == prompts.SYSTEM


@pytest.mark.parametrize("module,first", LOOPS, ids=lambda x: getattr(x, "NAME", ""))
def test_loop_includes_the_task_in_the_prompt(
    module, first, scripted_llm, recording_env
):
    llm = scripted_llm([first] * 6)
    env = recording_env({"ls": lambda path: "a.txt"})

    module.run(task="UNIQUE_TASK_STRING", env=env, llm=llm)

    joined = str(llm.prompts[0])
    assert "UNIQUE_TASK_STRING" in joined


@pytest.mark.parametrize("module,first", LOOPS, ids=lambda x: getattr(x, "NAME", ""))
def test_loop_declares_its_name(module, first, scripted_llm, recording_env):
    llm = scripted_llm([first] * 6)
    env = recording_env({"ls": lambda path: "a.txt"})

    trace = module.run(task="task", env=env, llm=llm)

    assert trace.loop == module.NAME
    assert module.NAME in prompts.LOOP_INSTRUCTIONS


_TWO = [
    {"name": "ls", "arguments": {"path": "a"}},
    {"name": "ls", "arguments": {"path": "b"}},
]
_DONE = {"tool_calls": None, "text": "Task completed"}
_TWO_CODE = [
    {"name": "execute_code", "arguments": {"code": "ls(path='a')"}},
    {"name": "execute_code", "arguments": {"code": "ls(path='b')"}},
]

NATIVE_MULTI = [
    (single_call, [{"tool_calls": _TWO}], ["ls", "ls"]),
    (react, [{"tool_calls": _TWO}, _DONE], ["ls", "ls"]),
    (adapt, [{"tool_calls": _TWO}, _DONE], ["ls", "ls"]),
    (reflexion, [{"tool_calls": _TWO}, _DONE], ["ls", "ls"]),
    (dfsdt, [{"tool_calls": _TWO}, _DONE], ["ls", "ls"]),
    (fixed_pipeline, [{"tool_calls": _TWO}, _DONE, _DONE], ["ls", "ls"]),
    (codeact, [{"tool_calls": _TWO_CODE}, _DONE], ["execute_code", "execute_code"]),
]


@pytest.mark.parametrize(
    "module,script,expected", NATIVE_MULTI, ids=[m.NAME for m, _, _ in NATIVE_MULTI]
)
def test_native_loop_executes_every_call_in_a_multi_call_response(
    module, script, expected, scripted_llm, recording_env
):
    llm = scripted_llm(script)
    env = recording_env({"ls": lambda path: path, "execute_code": lambda code: "ok"})

    trace = module.run(task="t", env=env, llm=llm)

    assert [name for name, _ in env.executed] == expected
    assert llm.calls_made == len(script)
    assert trace.n_llm_calls == len(script)


_PRIOR = [
    {"role": "user", "content": "PRIOR_TURN_REQUEST"},
    {"role": "assistant", "content": {"text": "PRIOR_TURN_ANSWER"}},
]

ALL_LOOPS_FIRST = LOOPS + [
    (reflexion, {"tool_calls": None, "text": "Task completed"}),
    (dfsdt, {"tool_calls": None, "text": "Task completed"}),
    (llm_compiler, {"tool_calls": None, "text": "Final: done"}),
]


@pytest.mark.parametrize(
    "module,first", ALL_LOOPS_FIRST, ids=lambda x: getattr(x, "NAME", "")
)
def test_loop_accepts_prior_turns_and_puts_them_before_the_task(
    module, first, scripted_llm, recording_env
):
    llm = scripted_llm([first] + [{"tool_calls": None, "text": "Task completed"}] * 4)
    env = recording_env({"ls": lambda path: path, "execute_code": lambda code: "ok"})

    module.run(task="CURRENT_TASK", env=env, llm=llm, history=list(_PRIOR))

    first_prompt = llm.prompts[0]
    texts = [str(m.get("content")) for m in first_prompt]
    i_prior = next(i for i, t in enumerate(texts) if "PRIOR_TURN_REQUEST" in t)
    i_task = next(i for i, t in enumerate(texts) if "CURRENT_TASK" in t)
    assert i_prior < i_task, (
        f"{module.NAME}: the prior turn comes after the current request"
    )
