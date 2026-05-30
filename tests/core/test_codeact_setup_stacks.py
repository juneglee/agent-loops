import pytest

from agent_loops.bench.core.codeact_setup import prepare

CASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ls",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "boolean"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cat",
            "parameters": {
                "type": "object",
                "properties": {"file_name": {"type": "string"}},
                "required": ["file_name"],
            },
        },
    },
]


class _Env:
    def __init__(self):
        self.code_enabled = False

    def enable_code_execution(self):
        self.code_enabled = True


@pytest.mark.parametrize(
    "name",
    [
        "codeact",
        "planner+codeact",
        "adaptive+codeact",
        "routed+codeact",
        "codeact+todo",
        "planner+codeact+todo",
    ],
)
def test_every_stack_containing_codeact_gets_the_code_action_space(name):
    env = _Env()

    tools, extra = prepare(env, CASE_TOOLS, name)

    assert [t["function"]["name"] for t in tools] == ["execute_code"]
    assert extra is not None and "ls" in extra and "cat" in extra
    assert env.code_enabled


@pytest.mark.parametrize(
    "name",
    [
        "react",
        "single_call",
        "planner+react",
        "adaptive+single_call",
        "react+todo",
    ],
)
def test_stacks_without_codeact_are_untouched(name):
    env = _Env()

    tools, extra = prepare(env, CASE_TOOLS, name)

    assert tools is CASE_TOOLS and extra is None
    assert not env.code_enabled
