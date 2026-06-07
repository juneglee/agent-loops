from agent_loops.bench.core.runner import CaseResult, summarize


class _LLMStub:
    def __init__(self, calls_made: int, parse_failures: int, quiet_failures: int = 0):
        self.calls_made = calls_made
        self.parse_failures = parse_failures
        self.quiet_failures = quiet_failures


def test_parse_rate_uses_llm_level_failures_not_loop_flags():
    r = CaseResult(case_id="c", loop="single_call")
    r.n_llm_calls = 10
    r.parse_failures = 0
    r.loop_parse_flags = 4

    assert r.parse_rate == 1.0


def test_parse_rate_reflects_actual_parse_failures():
    r = CaseResult(case_id="c", loop="react")
    r.n_llm_calls = 10
    r.parse_failures = 2

    assert r.parse_rate == 0.8


def test_parse_rate_is_zero_when_no_calls_made():
    r = CaseResult(case_id="c", loop="x")

    assert r.parse_rate == 0.0


def test_summary_separates_parse_from_loop_flags():
    a = CaseResult(
        case_id="a", loop="l", n_llm_calls=5, parse_failures=1, loop_parse_flags=3
    )
    b = CaseResult(
        case_id="b", loop="l", n_llm_calls=5, parse_failures=0, loop_parse_flags=0
    )

    s = summarize([a, b])

    assert s["parse_rate"] == 0.9
    assert s["loop_no_call_rate"] == 0.3


def test_duplicate_tool_calls_are_counted_and_summarised():
    from agent_loops.bench.core.runner import (
        CaseResult,
        count_duplicate_calls,
        summarize,
    )

    calls = [
        {"name": "cd", "arguments": {"folder": "a"}},
        {"name": "ls", "arguments": {}},
        {"name": "cd", "arguments": {"folder": "a"}},
        {"name": "ls", "arguments": {}},
        {"name": "cd", "arguments": {"folder": "b"}},
    ]
    assert count_duplicate_calls(calls) == 2

    rows = [
        CaseResult(case_id="x", loop="react", n_duplicate_tool_calls=2),
        CaseResult(case_id="y", loop="react", n_duplicate_tool_calls=0),
    ]
    assert summarize(rows)["mean_duplicate_tool_calls"] == 1.0


def test_success_turns_without_tools_are_counted():
    from agent_loops.bench.core.runner import CaseResult, silent_success, summarize
    from agent_loops.loops.base import Step, Trace

    silent = Trace(
        task="t", loop="react", steps=[Step(llm_response={"text": "Final: done"})]
    )
    worked = Trace(
        task="t",
        loop="react",
        steps=[
            Step(
                llm_response={"tool_calls": [{"name": "ls", "arguments": {}}]},
                tool_name="ls",
            ),
            Step(llm_response={"text": "Final: done"}),
        ],
    )
    assert silent_success(silent) is True
    assert silent_success(worked) is False
    assert (
        silent_success(Trace(task="t", loop="react", terminated_by="max_steps"))
        is False
    )

    rows = [
        CaseResult(case_id="x", loop="react", n_success_turns_without_tools=2),
        CaseResult(case_id="y", loop="react", n_success_turns_without_tools=0),
    ]
    assert summarize(rows)["mean_success_turns_without_tools"] == 1.0
