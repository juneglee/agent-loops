from agent_loops.bench import prompts
from agent_loops.loops import (
    adapt,
    codeact,
    fixed_pipeline,
    plan_and_act,
    plan_and_execute,
    plan_and_solve,
    react,
    rewoo,
    single_call,
)

ALL_LOOPS = [
    single_call.NAME,
    react.NAME,
    rewoo.NAME,
    plan_and_solve.NAME,
    plan_and_execute.NAME,
    plan_and_act.NAME,
    adapt.NAME,
    codeact.NAME,
    fixed_pipeline.NAME,
]


def test_every_loop_has_an_instruction():
    missing = [name for name in ALL_LOOPS if name not in prompts.LOOP_INSTRUCTIONS]

    assert missing == []


def test_all_loops_receive_identical_system_prompt():
    systems = {
        name: prompts.build_messages(name, task="any task")[0]["content"]
        for name in ALL_LOOPS
    }

    assert len(set(systems.values())) == 1


def test_system_message_comes_first():
    for name in ALL_LOOPS:
        messages = prompts.build_messages(name, task="task")
        assert messages[0]["role"] == "system"


def test_loop_instructions_differ_between_loops():
    react_msgs = prompts.build_messages("react", task="task")
    rewoo_msgs = prompts.build_messages("rewoo", task="task")

    react_user = react_msgs[-1]["content"]
    rewoo_user = rewoo_msgs[-1]["content"]

    assert react_user != rewoo_user
    assert "#E1" in rewoo_user
    assert "#E1" not in react_user


def test_task_appears_in_the_prompt():
    messages = prompts.build_messages("react", task="tidy up the docs")

    assert any("tidy up the docs" in str(m["content"]) for m in messages)


def test_history_is_appended_after_the_instruction():
    history = [{"role": "tool", "content": {"output": "a.txt"}}]

    messages = prompts.build_messages("react", task="task", history=history)

    assert messages[-1] == history[0]


def test_prompt_version_is_declared():
    assert isinstance(prompts.PROMPT_VERSION, str)
    assert prompts.PROMPT_VERSION


def test_prior_turns_come_before_the_current_task():
    from agent_loops.bench.prompts import build_messages

    prior = [
        {"role": "user", "content": "PRIOR_REQUEST"},
        {"role": "assistant", "content": {"text": "PRIOR_ANSWER"}},
    ]
    messages = build_messages(
        "react",
        "CURRENT_REQUEST",
        history=[{"role": "tool", "content": "OBSERVATION"}],
        prior=prior,
    )

    roles_contents = [(m["role"], str(m["content"])) for m in messages]
    i_prior = next(i for i, (_, c) in enumerate(roles_contents) if "PRIOR_REQUEST" in c)
    i_task = next(
        i for i, (_, c) in enumerate(roles_contents) if "CURRENT_REQUEST" in c
    )
    i_hist = next(i for i, (_, c) in enumerate(roles_contents) if "OBSERVATION" in c)
    assert messages[0]["role"] == "system"
    assert i_prior < i_task < i_hist


def test_codeact_instruction_declares_no_imports_and_virtual_filesystem():
    from agent_loops.bench.core.codeact_setup import code_signatures
    from agent_loops.bench.prompts import LOOP_INSTRUCTIONS

    text = LOOP_INSTRUCTIONS["codeact"] + code_signatures([])
    assert "import" in text
    assert "virtual" in text


def test_demo_tool_schemas_are_openai_shaped():
    schemas = prompts.demo_tool_schemas()

    assert schemas, "with empty tool schemas no loop can use a tool"
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "parameters" in s["function"]
