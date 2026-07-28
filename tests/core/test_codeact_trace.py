import pytest

from agent_loops.bench.bfcl.runner import _decoded_from_turn_traces
from agent_loops.loops import codeact


class CodeEnv:
    def __init__(self, inner):
        self._inner = inner
        self.executed = []

    def execute(self, name, arguments):
        self.executed.append((name, dict(arguments)))
        if name != "execute_code":
            return {"ok": True, "error": None, "output": ""}
        return {
            "ok": True,
            "error": None,
            "output": "output",
            "calls": [dict(c) for c in self._inner],
        }


def _code(code="pwd()\nls()"):
    return {"tool_calls": [{"name": "execute_code", "arguments": {"code": code}}]}


DONE = {"tool_calls": None, "text": "Final: done"}


def test_inner_calls_appear_in_the_trace(scripted_llm):
    env = CodeEnv(
        [{"name": "pwd", "arguments": {}}, {"name": "ls", "arguments": {"a": True}}]
    )

    trace = codeact.run(
        task="t", env=env, llm=scripted_llm([_code(), DONE]), max_steps=3
    )

    recorded = [(s.tool_name, s.tool_arguments) for s in trace.steps if s.tool_name]
    assert recorded == [("pwd", {}), ("ls", {"a": True})]


def test_execute_code_itself_is_not_sent_to_the_checker(scripted_llm):
    env = CodeEnv([{"name": "pwd", "arguments": {}}])

    trace = codeact.run(
        task="t", env=env, llm=scripted_llm([_code(), DONE]), max_steps=3
    )

    decoded = _decoded_from_turn_traces([trace])
    assert "execute_code" not in str(decoded)
    assert decoded == [[["pwd()"]]]


def test_one_llm_call_can_produce_many_tool_calls(scripted_llm):
    env = CodeEnv([{"name": f"t{i}", "arguments": {}} for i in range(5)])

    trace = codeact.run(
        task="t", env=env, llm=scripted_llm([_code(), DONE]), max_steps=3
    )

    assert trace.n_llm_calls == 2, "one code call plus one finish call"
    assert trace.n_tool_calls == 5


def test_code_that_calls_nothing_records_no_tool_steps(scripted_llm):
    env = CodeEnv([])

    trace = codeact.run(
        task="t", env=env, llm=scripted_llm([_code("x = 1"), DONE]), max_steps=3
    )

    assert trace.n_tool_calls == 0
    assert _decoded_from_turn_traces([trace]) == [[]]


def test_failed_code_still_records_the_calls_that_happened(scripted_llm):

    class FailingEnv(CodeEnv):
        def execute(self, name, arguments):
            self.executed.append((name, dict(arguments)))
            return {
                "ok": False,
                "error": "NameError: boom",
                "output": "",
                "calls": [{"name": "pwd", "arguments": {}}],
            }

    trace = codeact.run(
        task="t", env=FailingEnv([]), llm=scripted_llm([_code(), DONE]), max_steps=3
    )

    assert [s.tool_name for s in trace.steps if s.tool_name] == ["pwd"]


@pytest.mark.integration
def test_end_to_end_with_the_real_backend(scripted_llm):
    from agent_loops.bench.bfcl.env import BFCLEnv
    from agent_loops.bench.bfcl.partition import partition

    env = BFCLEnv(partition()["single_turn_multi_step"][0])
    env.enable_code_execution()

    trace = codeact.run(
        task="t", env=env, max_steps=3, llm=scripted_llm([_code("pwd()\nls()"), DONE])
    )

    assert [s.tool_name for s in trace.steps if s.tool_name] == ["pwd", "ls"]
    assert _decoded_from_turn_traces([trace]) == [[["pwd()"], ["ls()"]]]
