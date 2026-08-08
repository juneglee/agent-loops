from agent_loops.loops import (
    adapt,
    codeact,
    fixed_pipeline,
    plan_and_act,
    plan_and_execute,
)
from tests.conftest import RecordingEnv, ScriptedLLM

GARBAGE = "Hmm... I am not sure what to do."
_ADAPT_FAILED = {"tool_calls": None, "text": "Task failed: the target is missing"}


def _env():
    return RecordingEnv({"ls": lambda path: path, "execute_code": lambda code: "ok"})


class TestParseFailureIsNotSuccess:
    def test_plan_and_act_first_response_unparseable(self):
        trace = plan_and_act.run(
            task="t",
            env=_env(),
            max_steps=5,
            llm=ScriptedLLM([{"tool_calls": None, "text": GARBAGE}]),
        )

        assert trace.parse_ok is False
        assert trace.terminated_by == "parse_fail"

    def test_plan_and_act_executor_final_after_work_is_success(self):
        trace = plan_and_act.run(
            task="t",
            env=_env(),
            max_steps=5,
            llm=ScriptedLLM(
                [
                    {"tool_calls": None, "text": "1. look at a"},
                    {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
                    {"tool_calls": None, "text": "1. report the result"},
                    {"tool_calls": None, "text": "Final: done"},
                ]
            ),
        )

        assert trace.parse_ok is True
        assert trace.terminated_by == "success"

    def test_plan_and_act_empty_replan_is_parse_fail(self):
        trace = plan_and_act.run(
            task="t",
            env=_env(),
            max_steps=5,
            llm=ScriptedLLM(
                [
                    {"tool_calls": None, "text": "1. look at a"},
                    {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
                    {"tool_calls": None, "text": ""},
                ]
            ),
        )

        assert trace.terminated_by == "parse_fail"

    def test_plan_and_execute_second_round_garbage_is_not_silent_success(self):
        trace = plan_and_execute.run(
            task="t",
            env=_env(),
            max_rounds=5,
            llm=ScriptedLLM(
                [
                    {"tool_calls": None, "text": "1. ls[path=a]"},
                    {"tool_calls": None, "text": GARBAGE},
                ]
            ),
        )

        assert trace.terminated_by == "no_plan", (
            "an empty reply and garbage must be told apart"
        )

    def test_plan_and_execute_completion_declaration_after_work_is_success(self):
        trace = plan_and_execute.run(
            task="t",
            env=_env(),
            max_rounds=5,
            llm=ScriptedLLM(
                [
                    {"tool_calls": None, "text": "1. ls[path=a]"},
                    {"tool_calls": None, "text": "Final: checked a"},
                ]
            ),
        )
        assert trace.parse_ok is True
        assert trace.terminated_by == "success"

    def test_plan_and_act_planner_cannot_declare_completion(self):
        trace = plan_and_act.run(
            task="t",
            env=_env(),
            max_steps=5,
            llm=ScriptedLLM([{"tool_calls": None, "text": "Task completed"}]),
        )
        assert trace.terminated_by == "parse_fail"

    def test_plan_and_execute_empty_second_round_is_success(self):
        trace = plan_and_execute.run(
            task="t",
            env=_env(),
            max_rounds=5,
            llm=ScriptedLLM(
                [
                    {"tool_calls": None, "text": "1. ls[path=a]"},
                    {"tool_calls": None, "text": ""},
                ]
            ),
        )

        assert trace.terminated_by == "success"

    def test_fixed_pipeline_that_never_called_a_tool_is_not_success(self):
        env = _env()
        trace = fixed_pipeline.run(
            task="t",
            env=env,
            llm=ScriptedLLM([{"tool_calls": None, "text": GARBAGE}] * 3),
        )

        assert env.executed == []
        assert trace.terminated_by == "no_action"
        assert trace.parse_ok is False

    def test_fixed_pipeline_that_called_tools_completes(self):
        trace = fixed_pipeline.run(
            task="t",
            env=_env(),
            llm=ScriptedLLM(
                [{"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]}] * 3
            ),
        )

        assert trace.terminated_by == "success"
        assert trace.parse_ok is True


class TestCodeActActionSpace:
    def test_non_code_tool_is_not_executed(self):
        env = _env()
        codeact.run(
            task="t",
            env=env,
            max_steps=3,
            llm=ScriptedLLM(
                [
                    {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
                    {"tool_calls": None, "text": "Final: done"},
                ]
            ),
        )

        assert env.executed == [], (
            "a non-code tool was executed; this is indistinguishable from ReAct"
        )

    def test_non_code_tool_gets_an_error_observation_for_self_debug(self):
        env = _env()
        trace = codeact.run(
            task="t",
            env=env,
            max_steps=3,
            llm=ScriptedLLM(
                [
                    {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
                    {
                        "tool_calls": [
                            {"name": "execute_code", "arguments": {"code": "x"}}
                        ]
                    },
                    {"tool_calls": None, "text": "Final: done"},
                ]
            ),
        )

        assert env.executed == [("execute_code", {"code": "x"})]
        rejected = trace.steps[0]
        assert rejected.observation is not None
        assert rejected.observation.get("ok") is False
        assert codeact.CODE_TOOL in str(rejected.observation.get("error", ""))

    def test_code_tool_is_executed(self):
        env = _env()
        codeact.run(
            task="t",
            env=env,
            max_steps=3,
            llm=ScriptedLLM(
                [
                    {
                        "tool_calls": [
                            {"name": "execute_code", "arguments": {"code": "x"}}
                        ]
                    },
                    {"tool_calls": None, "text": "Final: done"},
                ]
            ),
        )

        assert env.executed == [("execute_code", {"code": "x"})]


class TestAdaptDecomposition:
    def _failing_env(self):
        def boom(**_):
            raise RuntimeError("no such file")

        return RecordingEnv({"ls": boom})

    def test_decomposition_prompt_differs_from_execution_prompt(self):
        llm = ScriptedLLM(
            [
                {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
                _ADAPT_FAILED,
                {"tool_calls": None, "text": GARBAGE},
            ]
        )
        adapt.run(task="t", env=self._failing_env(), llm=llm, max_depth=2)

        assert llm.prompts[0] != llm.prompts[2], (
            "asking again with the same input yields the same answer; no decomposition can happen"
        )

    def test_decomposition_prompt_carries_the_failure_observation(self):
        llm = ScriptedLLM(
            [
                {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
                _ADAPT_FAILED,
                {"tool_calls": None, "text": GARBAGE},
            ]
        )
        adapt.run(task="t", env=self._failing_env(), llm=llm, max_depth=2)

        assert "no such file" not in str(llm.prompts[2]), (
            "without knowing why it failed there is no way to split it differently"
        )

    def test_unparseable_decomposition_is_reported_as_such(self):
        trace = adapt.run(
            task="t",
            env=self._failing_env(),
            max_depth=3,
            llm=ScriptedLLM(
                [
                    {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
                    _ADAPT_FAILED,
                    {"tool_calls": None, "text": GARBAGE},
                ]
            ),
        )

        assert trace.terminated_by == "parse_fail"

    def test_depth_budget_exhaustion_is_reported_as_max_depth(self):
        trace = adapt.run(
            task="t",
            env=self._failing_env(),
            max_depth=1,
            llm=ScriptedLLM(
                [
                    _ADAPT_FAILED,
                    {"tool_calls": None, "text": "- subtask A"},
                    _ADAPT_FAILED,
                ]
            ),
        )

        assert trace.terminated_by == "max_depth"

    def test_successful_first_try_does_not_decompose(self):
        llm = ScriptedLLM(
            [
                {"tool_calls": [{"name": "ls", "arguments": {"path": "a"}}]},
                {"tool_calls": None, "text": "Task completed"},
            ]
        )
        trace = adapt.run(task="t", env=_env(), llm=llm, max_depth=3)

        assert llm.calls_made == 2
        assert all(k.get("want") != "text" for k in llm.kwargs)
        assert trace.terminated_by == "success"
