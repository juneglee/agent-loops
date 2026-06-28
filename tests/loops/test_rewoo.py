from agent_loops.loops.rewoo import NO_EVIDENCE, SOLVER_PREFIX, SOLVER_SUFFIX, run


def _plan(steps: str):
    return {"tool_calls": None, "text": steps}


def _solver_text(llm) -> str:
    return llm.prompts[1][-1]["content"]


def test_rewoo_makes_two_llm_calls_regardless_of_tool_count(
    scripted_llm, recording_env
):
    plan_5 = "\n".join(f"#E{i} = ls[path=p{i}]" for i in range(1, 6))
    llm = scripted_llm([_plan(plan_5), {"tool_calls": None, "text": "combined result"}])
    env = recording_env({"ls": lambda path: path})

    trace = run(task="show five places", env=env, llm=llm)

    assert llm.calls_made == 2
    assert len(env.executed) == 5
    assert trace.n_llm_calls == 2


def test_rewoo_worker_executes_tools_without_llm(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan("#E1 = ls[path=a]\n#E2 = ls[path=b]"),
            {"tool_calls": None, "text": "done"},
        ]
    )
    env = recording_env({"ls": lambda path: path})

    run(task="both", env=env, llm=llm)

    assert [name for name, _ in env.executed] == ["ls", "ls"]
    assert llm.calls_made == 2


def test_rewoo_substitutes_variable_references(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan("#E1 = ls[path=docs]\n#E2 = cat[path=#E1]"),
            {"tool_calls": None, "text": "done"},
        ]
    )
    env = recording_env(
        {"ls": lambda path: "found.txt", "cat": lambda path: f"content:{path}"}
    )

    run(task="find and read", env=env, llm=llm)

    assert env.executed[1] == ("cat", {"path": "found.txt"})


def test_rewoo_records_parse_failure_when_plan_unreadable(scripted_llm, recording_env):
    llm = scripted_llm(
        [_plan("I cannot make a plan"), {"tool_calls": None, "text": "x"}]
    )
    env = recording_env({"ls": lambda path: path})

    trace = run(task="anything", env=env, llm=llm)

    assert trace.parse_ok is False
    assert trace.terminated_by == "parse_fail"
    assert env.executed == []


def test_rewoo_solver_receives_ordered_plan_evidence_pairs(scripted_llm, recording_env):
    llm = scripted_llm(
        [
            _plan(
                "Plan: list the docs folder\n#E1 = ls[path=docs]\n"
                "Plan: read the file found\n#E2 = cat[path=#E1]"
            ),
            {"tool_calls": None, "text": "done"},
        ]
    )
    env = recording_env(
        {"ls": lambda path: "found.txt", "cat": lambda path: f"content:{path}"}
    )

    run(task="find and read", env=env, llm=llm)

    text = _solver_text(llm)
    assert llm.kwargs[1] == {"want": "text"}
    assert text.startswith(SOLVER_PREFIX)
    assert text.endswith(SOLVER_SUFFIX)
    assert (
        "Plan: list the docs folder #E1 = ls[path='docs']\nEvidence: found.txt\n"
        "Plan: read the file found #E2 = cat[path='found.txt']\nEvidence: content:found.txt\n"
    ) in text
    assert "Task: find and read" in text
    assert text.index("Evidence: found.txt") < text.index("Evidence: content:found.txt")


def test_rewoo_solver_sees_tool_error_text_as_evidence(scripted_llm, recording_env):
    def boom(path):
        raise FileNotFoundError(path)

    llm = scripted_llm(
        [
            _plan("#E1 = cat[path=missing.txt]\n#E2 = ls[path=.]"),
            {"tool_calls": None, "text": "done"},
        ]
    )
    env = recording_env({"cat": boom, "ls": lambda path: "a.txt"})

    run(task="read then list", env=env, llm=llm)

    text = _solver_text(llm)
    assert (
        "Plan: #E1 = cat[path='missing.txt']\nEvidence: FileNotFoundError: missing.txt\n"
        in text
    )
    assert "Plan: #E2 = ls[path='.']\nEvidence: a.txt\n" in text
    assert "Evidence: \n" not in text


def test_rewoo_unknown_tool_evidence_is_not_empty(scripted_llm, recording_env):
    llm = scripted_llm(
        [_plan("#E1 = nope[path=x]"), {"tool_calls": None, "text": "done"}]
    )
    env = recording_env({"ls": lambda path: path})

    run(task="t", env=env, llm=llm)

    text = _solver_text(llm)
    assert "Evidence: unknown tool: nope\n" in text
    assert NO_EVIDENCE not in text
