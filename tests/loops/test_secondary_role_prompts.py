from agent_loops.bench import prompts
from agent_loops.loops import llm_compiler, reflexion, rewoo


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def _texts(messages):
    return "\n".join(
        m["content"] if isinstance(m["content"], str) else str(m["content"])
        for m in messages
    )


def test_rewoo_solver_gets_only_the_solver_prompt(scripted_llm, recording_env):
    llm = scripted_llm([_txt("Plan: look\n#E1 = ls[path=a]"), _txt("done")])
    rewoo.run(task="t", env=recording_env({"ls": lambda path: "x"}), llm=llm)

    solver = llm.prompts[1]
    assert [m["role"] for m in solver] == ["user"]
    assert solver[0]["content"].startswith(rewoo.SOLVER_PREFIX)
    assert prompts.LOOP_INSTRUCTIONS["rewoo"] not in _texts(solver)
    assert prompts.SYSTEM not in _texts(solver)


def test_reflexion_reflection_gets_only_the_reflect_prompt(scripted_llm, recording_env):
    llm = scripted_llm(
        [_txt("Task failed: no"), _txt("plan better"), _txt("Task completed")]
    )
    reflexion.run(task="t", env=recording_env({}), llm=llm, max_trials=2, max_steps=3)

    reflect = llm.prompts[1]
    assert [m["role"] for m in reflect] == ["user"]
    assert reflect[0]["content"].startswith(reflexion._REFLECT_INSTRUCTION)
    assert reflect[0]["content"].rstrip().endswith("Reflection:")
    assert prompts.LOOP_INSTRUCTIONS["reflexion"] not in _texts(reflect)


def test_llm_compiler_joiner_gets_only_the_joiner_prompt(scripted_llm, recording_env):
    llm = scripted_llm([_txt("#E1 = ls[path=a]"), _txt("Final: done")])
    llm_compiler.run(task="t", env=recording_env({"ls": lambda path: "x"}), llm=llm)

    joiner = llm.prompts[1]
    assert [m["role"] for m in joiner] == ["user"]
    assert llm_compiler.JOINER_INSTRUCTION in joiner[0]["content"]
    assert prompts.LOOP_INSTRUCTIONS["llm_compiler"] not in _texts(joiner)
    assert prompts.SYSTEM not in _texts(joiner)
