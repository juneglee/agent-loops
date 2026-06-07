from agent_loops.loops.base import Step, Trace


def _trace_with(n_llm: int, n_tool_only: int) -> Trace:
    trace = Trace(task="t", loop="x")
    for _ in range(n_llm):
        trace.steps.append(Step(llm_response={"text": "..."}))
    for _ in range(n_tool_only):
        trace.steps.append(Step(llm_response=None, tool_name="ls", observation={}))
    return trace


def test_multi_turn_runner_counts_llm_calls_not_steps():
    from agent_loops.bench.core.runner import CaseResult

    result = CaseResult(case_id="c", loop="rewoo")
    for trace in (_trace_with(2, 5), _trace_with(2, 3)):
        result.n_llm_calls += trace.n_llm_calls
        result.n_steps += len(trace.steps)

    assert result.n_llm_calls == 4, (
        "ReWOO makes 2 LLM calls per turn; the 8 tool calls are not LLM calls"
    )
    assert result.n_steps == 12, "step count is the full record including tool steps"


def test_parse_rate_denominator_is_llm_calls():
    from agent_loops.bench.core.runner import CaseResult

    result = CaseResult(case_id="c", loop="rewoo", n_llm_calls=4, parse_failures=1)

    assert result.parse_rate == 0.75


def test_trace_exposes_both_axes_separately():
    trace = _trace_with(2, 5)

    assert trace.n_llm_calls == 2
    assert trace.n_tool_calls == 5
    assert len(trace.steps) == 7
