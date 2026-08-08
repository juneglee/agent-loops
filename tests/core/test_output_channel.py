import pytest

from agent_loops.bench.core.local_llm import LocalLLM
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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "list entries",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

TEXT_PLAN_LOOPS = [rewoo, plan_and_solve, plan_and_execute, plan_and_act]
NATIVE_CALL_LOOPS = [single_call, react, adapt, codeact, fixed_pipeline]


class RecordingLLM(LocalLLM):
    def __init__(self, **kw):
        super().__init__(tools=TOOLS, **kw)
        self.payloads = []

    def __call__(self, messages, **kwargs):
        from agent_loops.bench.core.llm import build_payload

        self.calls_made += 1
        tools = None if kwargs.get("want") == "text" else self.tools
        self.payloads.append(
            build_payload(
                model=self.model,
                messages=[
                    {
                        "role": m.get("role", "user"),
                        "content": str(m.get("content", "")),
                    }
                    for m in messages
                ],
                tools=tools,
            )
        )
        return {"tool_calls": None, "text": "", "parse_ok": True}


@pytest.mark.parametrize("module", TEXT_PLAN_LOOPS, ids=lambda m: m.NAME)
def test_planning_loops_disable_native_tool_calls(module):
    llm = RecordingLLM()
    module.run(task="t", env=_NullEnv(), llm=llm)

    assert llm.payloads, f"{module.NAME}: the LLM was never called"
    assert "tools" not in llm.payloads[0], (
        f"{module.NAME}: tools are open on the planning request; the runtime would intercept the plan"
    )


@pytest.mark.parametrize("module", NATIVE_CALL_LOOPS, ids=lambda m: m.NAME)
def test_action_loops_keep_native_tool_calls(module):
    llm = RecordingLLM()
    module.run(task="t", env=_NullEnv(), llm=llm)

    assert llm.payloads
    assert "tools" in llm.payloads[0], f"{module.NAME}: the native tool path is closed"


class _NullEnv:
    @property
    def tool_names(self) -> list[str]:
        return ["ls"]

    def execute(self, name, arguments):
        return {"ok": True, "error": None, "output": ""}
