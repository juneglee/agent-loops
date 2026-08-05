from agent_loops.loops.dfsdt import DIVERSITY_PROMPT, run

_DIVERSITY_HEAD = DIVERSITY_PROMPT.split("\n")[0]


def _diversity_text(prompt) -> str | None:
    for m in prompt:
        if isinstance(m.get("content"), str) and m["content"].startswith(
            _DIVERSITY_HEAD
        ):
            return m["content"]
    return None


def _fail(**_kw):
    raise RuntimeError("no such file")


def _tc(name, **arguments):
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def test_dfsdt_tool_error_is_an_observation_not_a_branch(scripted_llm, recording_env):
    llm = scripted_llm([_tc("bad", p=1), _tc("good", p=2), _txt("Final: done")])
    env = recording_env({"bad": _fail, "good": lambda p: "ok"})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=6)

    assert trace.terminated_by == "success"
    assert [name for name, _ in env.executed] == ["bad", "good"]
    assert _diversity_text(llm.prompts[1]) is None
    assert "no such file" in str(llm.prompts[1])


def test_dfsdt_generates_sibling_with_diversity_even_without_give_up(
    scripted_llm, recording_env
):
    llm = scripted_llm([_tc("ls", path="a"), _tc("ls", path="b")])
    env = recording_env({"ls": lambda path: f"listing of {path}"})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=10, max_depth=1)

    assert llm.calls_made == 2
    assert [args["path"] for _, args in env.executed] == ["a", "b"]
    shown = _diversity_text(llm.prompts[1])
    assert shown is not None
    assert '"name": "ls"' in shown and '"path": "a"' in shown
    assert '"function_output"' in shown and "listing of a" in shown
    assert "all previous trails failed" in shown
    assert len(llm.prompts[1]) == len(llm.prompts[0]) + 1
    assert trace.terminated_by == "max_depth"


def test_dfsdt_give_up_prunes_back_to_the_parent_state(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _tc("cd", d="x"),
            _txt("Give up: this path is a dead end"),
            _tc("ls", p="."),
            _txt("Final: done"),
        ]
    )
    env = recording_env({"cd": lambda d: "ok", "ls": lambda p: "a"})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=8)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 4
    assert [name for name, _ in env.executed] == ["cd", "ls"]
    shown = _diversity_text(llm.prompts[2])
    assert shown is not None and '"name": "cd"' in shown
    assert _diversity_text(llm.prompts[1]) is None
    assert len(llm.prompts[2]) == len(llm.prompts[0]) + 1


def test_dfsdt_give_up_at_root_ends_the_search_with_the_give_up_text(
    scripted_llm, recording_env
):
    llm = scripted_llm([_txt("Give up: nothing applies")])
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=6)

    assert trace.terminated_by == "give_up"
    assert llm.calls_made == 1
    assert trace.steps[-1].llm_response["finish_type"] == "give_up"
    assert "nothing applies" in trace.steps[-1].llm_response["text"]


def test_dfsdt_depth_limit_prunes_and_parent_tries_next_sibling(
    scripted_llm, recording_env
):
    llm = scripted_llm([_tc(n) for n in "abcdef"])
    env = recording_env({n: (lambda: "ok") for n in "abcdef"})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=10, max_depth=2)

    assert [name for name, _ in env.executed] == list("abcdef")
    assert llm.calls_made == 6
    assert _diversity_text(llm.prompts[2]) is not None
    assert '"name": "b"' in _diversity_text(llm.prompts[2])
    assert "again, your former observation" in _diversity_text(llm.prompts[2])
    assert _diversity_text(llm.prompts[3]) is not None
    assert '"name": "a"' in _diversity_text(llm.prompts[3])
    assert trace.terminated_by == "max_depth"
    assert trace.parse_ok


def test_dfsdt_parse_error_prunes_and_parent_tries_next_sibling(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            {"tool_calls": None, "text": "", "parse_ok": False},
            _tc("ls", path="a"),
            _txt("Final: done"),
        ]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=6)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3
    assert _diversity_text(llm.prompts[1]) is None
    assert len(llm.prompts[1]) == len(llm.prompts[0])


def test_dfsdt_exhaustion_with_only_parse_errors_is_parse_fail(
    scripted_llm, recording_env
):
    llm = scripted_llm([{"tool_calls": None, "text": "", "parse_ok": False}] * 2)
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=6)

    assert trace.terminated_by == "parse_fail"
    assert not trace.parse_ok
    assert llm.calls_made == 2


def test_dfsdt_first_final_answer_terminates_the_whole_search(
    scripted_llm, recording_env
):
    llm = scripted_llm([_tc("ls", path="a"), _txt("Final: done")])
    env = recording_env({"ls": lambda path: path})

    trace = run(task="t", env=env, llm=llm, breadth=3, max_calls=10)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 2
    assert trace.parse_ok


def test_dfsdt_exhaustion_adopts_a_give_up_text_as_the_answer(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [
            _tc("a"),
            _txt("Give up: first attempt failed"),
            _tc("b"),
            _txt("Give up: second attempt failed"),
        ]
    )
    env = recording_env({"a": lambda: "ok", "b": lambda: "ok"})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=10)

    assert trace.terminated_by == "give_up"
    assert llm.calls_made == 4
    assert trace.n_llm_calls == 4
    last = trace.steps[-1].llm_response
    assert last["finish_type"] == "give_up"
    assert last["text"] == "Give up: second attempt failed"


def test_dfsdt_global_call_budget_stops_the_search(scripted_llm, recording_env):
    llm = scripted_llm([_tc("a"), _tc("b"), _tc("c")])
    env = recording_env({n: (lambda: "ok") for n in "abc"})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=2, max_depth=6)

    assert trace.terminated_by == "max_steps"
    assert llm.calls_made == 2
    assert trace.n_llm_calls == 2


def test_dfsdt_thought_only_response_becomes_a_node_and_continues(
    scripted_llm, recording_env
):
    llm = scripted_llm(
        [_txt("let me list first"), _tc("ls", path="a"), _txt("Final: done")]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=6)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3
    assert "let me list first" in str(llm.prompts[1])
    assert _diversity_text(llm.prompts[1]) is None


def test_dfsdt_thought_only_node_consumes_depth(scripted_llm, recording_env):
    llm = scripted_llm([_txt("thinking"), _txt("thinking again")])
    env = recording_env({})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=6, max_depth=1)

    assert llm.calls_made == 2
    assert trace.terminated_by == "max_depth"
    assert _diversity_text(llm.prompts[1]) is None


def test_dfsdt_without_give_up_behaves_like_react(scripted_llm, recording_env):
    llm = scripted_llm([_tc("ls", path="a"), _tc("ls", path="b"), _txt("Final: done")])
    env = recording_env({"ls": lambda path: path})

    trace = run(task="t", env=env, llm=llm, breadth=2, max_calls=6)

    assert trace.terminated_by == "success"
    assert llm.calls_made == 3
    assert trace.n_tool_calls == 2


def test_dfsdt_diversity_message_is_not_inherited_by_children(
    scripted_llm, recording_env
):
    llm = scripted_llm([_tc(n) for n in "abcdef"])
    env = recording_env({n: (lambda: "ok") for n in "abcdef"})

    run(task="t", env=env, llm=llm, breadth=2, max_calls=10, max_depth=2)

    deeper = _diversity_text(llm.prompts[2])
    root_sibling = _diversity_text(llm.prompts[3])
    second_subtree = _diversity_text(llm.prompts[5])
    assert '"name": "b"' in deeper
    assert '"name": "a"' in root_sibling
    assert '"name": "b"' not in root_sibling and '"name": "c"' not in root_sibling
    assert '"name": "e"' in second_subtree
    assert all(f'"name": "{n}"' not in second_subtree for n in "abcd")
    assert _diversity_text(llm.prompts[4]) is None
    assert (
        sum(
            1
            for m in llm.prompts[3]
            if isinstance(m.get("content"), str)
            and m["content"].startswith(_DIVERSITY_HEAD)
        )
        == 1
    )
