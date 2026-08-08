from typing import ClassVar

from scripts.run_single import measure_case


class _FakeLLM:
    made: ClassVar[list] = []

    def __init__(self, tools, base_url=None, model=None, **kwargs):
        self.parse_failures = 0
        _FakeLLM.made.append(self)


def test_each_repeat_gets_a_fresh_llm_and_uncontaminated_counters(monkeypatch):
    import scripts.run_single as mod

    monkeypatch.setattr(mod, "LocalLLM", _FakeLLM)
    monkeypatch.setattr(mod, "single_tools", lambda case: [])
    _FakeLLM.made = []
    seen = []

    def stub_runner(case, llm, category, loop_module=None, loop_kwargs=None):
        llm.parse_failures += 1
        from agent_loops.bench.bfcl.single_runner import SingleResult

        r = SingleResult(case_id=case["id"], loop="react")
        r.parse_failures = llm.parse_failures
        seen.append(llm)
        return r

    rows = measure_case(
        {"id": "c1"},
        None,
        {},
        base_url="http://x/v1",
        model="m",
        category="simple",
        repeats=3,
        _runner=stub_runner,
    )

    assert len(_FakeLLM.made) == 3
    assert len(set(map(id, seen))) == 3
    assert [r.parse_failures for r in rows] == [1, 1, 1]
    assert [r.run_index for r in rows] == [0, 1, 2]
