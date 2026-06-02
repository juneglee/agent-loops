import pytest

from agent_loops.bench.bfcl.single_runner import calls_to_ast_output, run_single_case

pytestmark_integration = pytest.mark.integration


class _ReplayLLM:
    def __init__(self, calls):
        self._calls = calls
        self.calls_made = 0
        self.parse_failures = 0
        self.quiet_failures = 0

    def __call__(self, messages, **_):
        self.calls_made += 1
        return {"tool_calls": self._calls, "text": "", "parse_ok": True}


def test_calls_to_ast_output_uses_name_as_key():
    calls = [{"name": "calc", "arguments": {"base": 10, "height": 5}}]

    assert calls_to_ast_output(calls) == [{"calc": {"base": 10, "height": 5}}]


def test_calls_to_ast_output_handles_multiple_calls():
    calls = [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {}},
    ]

    assert calls_to_ast_output(calls) == [{"a": {"x": 1}}, {"b": {}}]


def test_calls_to_ast_output_of_nothing_is_empty_list():
    assert calls_to_ast_output([]) == []
    assert calls_to_ast_output(None) == []


@pytest.mark.integration
def test_ground_truth_scores_correct_through_single_runner():
    from agent_loops.bench.bfcl.single import load_single_answers, load_single_cases

    cases = load_single_cases("simple")[:20]
    answers = load_single_answers("simple")
    failures = []

    for case in cases:
        gt = answers[case["id"]]
        calls = [
            {
                "name": name.replace(".", "_"),
                "arguments": {k: v[0] for k, v in args.items() if v},
            }
            for entry in gt
            for name, args in entry.items()
        ]

        r = run_single_case(case, _ReplayLLM(calls), "simple")
        if not r.valid:
            failures.append((case["id"], r.error))

    assert len(failures) <= 2, f"{len(failures)}/20 failed: {failures[:3]}"
