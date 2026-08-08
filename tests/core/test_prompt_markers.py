from agent_loops.bench import prompts


def _instruction(name: str) -> str:
    return prompts.build_messages(name, task="t")[-1]["content"]


def test_react_family_instructs_the_completion_marker():
    for name in ("react", "reflexion", "dfsdt"):
        assert "Final:" in _instruction(name), name


def test_react_family_allows_thought_only_steps():
    for name in ("react", "reflexion", "dfsdt"):
        assert "thought" in _instruction(name), name


def test_dfsdt_and_adapt_instruct_an_explicit_give_up_form():
    assert "Give up:" in _instruction("dfsdt")
    assert "Task failed:" in _instruction("adapt")
    assert "Task completed" in _instruction("adapt")


def test_plan_and_act_has_a_separate_executor_instruction():
    planner = _instruction("plan_and_act")
    executor = _instruction("plan_and_act_executor")
    assert "Final:" not in planner
    assert "tool names" in planner
    assert "Final:" in executor
    assert planner != executor


def test_rewoo_instructs_the_plan_reasoning_line():
    assert "Plan:" in _instruction("rewoo")


def test_prompt_version_bumped_for_marker_change():
    assert prompts.PROMPT_VERSION == "v1"
