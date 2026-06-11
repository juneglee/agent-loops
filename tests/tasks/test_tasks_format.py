import json

import pytest

from agent_loops.bench.tasks.format import cell_of, load_tasks, turns_of


def _write(tmp_path, cases):
    (tmp_path / "fx").mkdir(exist_ok=True)
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(cases, ensure_ascii=False))
    return p


def _case(**over):
    base = {
        "id": "a",
        "fixture": "fx",
        "turns": ["t"],
        "gt_calls": [[{"name": "list_dir", "arguments": {}}]],
    }
    base.update(over)
    return base


def test_loads_well_formed_cases(tmp_path):
    cases = load_tasks(_write(tmp_path, [_case()]))
    assert [c["id"] for c in cases] == ["a"]
    assert turns_of(cases[0]) == ["t"]


@pytest.mark.parametrize(
    "broken",
    [
        {"id": "a", "fixture": "fx", "turns": ["t"]},
        _case(turns=["t", "u"]),
        _case(gt_calls=[[{"name": "list_dir"}]]),
        _case(fixture="no_such"),
        _case(gt_calls=[[{"name": "teleport", "arguments": {}}]]),
    ],
)
def test_malformed_case_is_rejected_loudly(tmp_path, broken):
    with pytest.raises(ValueError, match="a"):
        load_tasks(_write(tmp_path, [broken]))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="a"):
        load_tasks(_write(tmp_path, [_case(), _case()]))


def test_cells_follow_the_partition_rule():
    one = [{"name": "list_dir", "arguments": {}}]
    two = one + [{"name": "bash", "arguments": {"command": "ls"}}]
    assert cell_of(_case(gt_calls=[one])) == "single_turn_single_step"
    assert cell_of(_case(gt_calls=[two])) == "single_turn_multi_step"
    assert (
        cell_of(_case(turns=["t", "u"], gt_calls=[one, one]))
        == "multi_turn_single_step"
    )
    assert (
        cell_of(_case(turns=["t", "u"], gt_calls=[one, two])) == "multi_turn_multi_step"
    )


def test_declared_cell_must_match_the_derived_one(tmp_path):
    with pytest.raises(ValueError, match="cell"):
        load_tasks(_write(tmp_path, [_case(cell="multi_turn_multi_step")]))
