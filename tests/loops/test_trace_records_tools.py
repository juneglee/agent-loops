import pytest

from agent_loops.bench.bfcl.runner import _decoded_from_turn_traces
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
from tests.conftest import RecordingEnv, ScriptedLLM


class CodeRecordingEnv(RecordingEnv):
    def execute(self, name, arguments):
        out = super().execute(name, arguments)
        if name == "execute_code":
            out["calls"] = [{"name": "ls", "arguments": {"path": "a"}}]
        return out

    @property
    def executed_tools(self):
        return [(n, a) for n, a in self.executed if n != "execute_code"] + [
            ("ls", {"path": "a"})
        ] * sum(1 for n, _ in self.executed if n == "execute_code")


CASES = [
    (
        single_call,
        [{"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]}],
        {},
        1,
    ),
    (
        react,
        [
            {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
            {"tool_calls": [{"name": "ls", "arguments": {"path": "b"}}]},
            {"tool_calls": None, "text": "Final: done"},
        ],
        {"max_steps": 5},
        2,
    ),
    (
        rewoo,
        [
            {"tool_calls": None, "text": "#E1 = ls[path=a]\n#E2 = ls[path=b]"},
            {"tool_calls": None, "text": "Final: done"},
        ],
        {},
        2,
    ),
    (
        plan_and_solve,
        [{"tool_calls": None, "text": "1. ls[path=a]\n2. ls[path=b]"}],
        {},
        2,
    ),
    (
        plan_and_execute,
        [
            {"tool_calls": None, "text": "1. ls[path=a]\n2. ls[path=b]"},
            {"tool_calls": None, "text": ""},
        ],
        {"max_rounds": 3},
        2,
    ),
    (
        plan_and_act,
        [
            {"tool_calls": None, "text": "1. look at a"},
            {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
            {"tool_calls": None, "text": "1. look at b"},
            {"tool_calls": [{"name": "ls", "arguments": {"path": "b"}}]},
            {"tool_calls": None, "text": "Final: done"},
        ],
        {"max_steps": 5},
        2,
    ),
    (
        adapt,
        [
            {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
            {"tool_calls": None, "text": "Task completed"},
        ],
        {"max_depth": 2},
        1,
    ),
    (
        codeact,
        [
            {"tool_calls": [{"name": "execute_code", "arguments": {"code": "x"}}]},
            {"tool_calls": None, "text": "Final: done"},
        ],
        {"max_steps": 3},
        1,
    ),
    (
        fixed_pipeline,
        [{"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]}] * 3,
        {},
        3,
    ),
]


def _run(module, script, kwargs):
    tools = {"ls": lambda path: path, "execute_code": lambda code: "ok"}
    env = (CodeRecordingEnv if module.NAME == "codeact" else RecordingEnv)(tools)
    llm = ScriptedLLM(script)
    trace = module.run(task="t", env=env, llm=llm, **kwargs)
    return env, trace


def _expected_tools(env):
    return getattr(env, "executed_tools", None) or env.executed


@pytest.mark.parametrize(
    "module,script,kwargs,expected", CASES, ids=lambda x: getattr(x, "NAME", "")
)
def test_every_executed_tool_appears_in_the_trace(module, script, kwargs, expected):
    env, trace = _run(module, script, kwargs)

    recorded = [s for s in trace.steps if s.tool_name is not None]
    wanted = _expected_tools(env)
    assert len(recorded) == len(wanted), (
        f"{module.NAME}: executed {len(wanted)}, recorded {len(recorded)}"
    )
    assert len(recorded) == expected


@pytest.mark.parametrize(
    "module,script,kwargs,expected", CASES, ids=lambda x: getattr(x, "NAME", "")
)
def test_recorded_calls_match_what_was_executed(module, script, kwargs, expected):
    env, trace = _run(module, script, kwargs)

    recorded = [(s.tool_name, s.tool_arguments) for s in trace.steps if s.tool_name]
    assert recorded == _expected_tools(env)


@pytest.mark.parametrize(
    "module,script,kwargs,expected", CASES, ids=lambda x: getattr(x, "NAME", "")
)
def test_decoded_output_is_not_empty_when_tools_ran(module, script, kwargs, expected):
    env, trace = _run(module, script, kwargs)

    decoded = _decoded_from_turn_traces([trace])
    wanted = _expected_tools(env)
    assert decoded[0], (
        f"{module.NAME}: {len(wanted)} tools executed but the scorer input is empty"
    )
    assert sum(len(step) for step in decoded[0]) == len(wanted)


@pytest.mark.parametrize(
    "module,script,kwargs,expected", CASES, ids=lambda x: getattr(x, "NAME", "")
)
def test_observations_are_recorded(module, script, kwargs, expected):
    _, trace = _run(module, script, kwargs)

    for step in trace.steps:
        if step.tool_name is not None:
            assert step.observation is not None, f"{module.NAME}: observation missing"
