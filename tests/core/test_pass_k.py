from agent_loops.bench.core.runner import CaseResult, pass_at_k


def _r(case_id, valid, run_index):
    r = CaseResult(case_id=case_id, loop="react")
    r.valid = valid
    r.run_index = run_index
    return r


def test_pass_k_requires_every_repeat_to_succeed():
    rows = [
        _r("c1", True, 0),
        _r("c1", True, 1),
        _r("c1", True, 2),
        _r("c2", True, 0),
        _r("c2", False, 1),
        _r("c2", True, 2),
    ]
    out = pass_at_k(rows)
    assert out["k"] == 3
    assert out["pass_k"] == 0.5
    assert abs(out["accuracy"] - 5 / 6) < 1e-9


def test_single_run_degenerates_to_plain_accuracy():
    rows = [_r("c1", True, 0), _r("c2", False, 0)]
    out = pass_at_k(rows)
    assert out["k"] == 1 and out["pass_k"] == 0.5 and out["accuracy"] == 0.5


def test_single_results_support_repeats_too():
    from agent_loops.bench.bfcl.single_runner import SingleResult

    rows = []
    for cid, valids in (("s1", [True, True]), ("s2", [True, False])):
        for i, v in enumerate(valids):
            r = SingleResult(case_id=cid, loop="react", run_index=i)
            r.valid = v
            rows.append(r)
    out = pass_at_k(rows)
    assert out["k"] == 2 and out["pass_k"] == 0.5 and out["accuracy"] == 0.75
