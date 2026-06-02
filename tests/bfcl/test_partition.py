import pytest

from agent_loops.bench.bfcl.partition import CELLS, partition, truncate_to_first_turn

pytestmark = pytest.mark.integration


def test_truncate_keeps_only_first_turn():
    case = {"id": "x", "question": [["t0"], ["t1"], ["t2"]], "involved_classes": []}

    out = truncate_to_first_turn(case)

    assert out["question"] == [["t0"]]
    assert case["question"] == [["t0"], ["t1"], ["t2"]], (
        "the original must not be mutated"
    )


EMPTY_IN_FILE_MANAGEMENT = {"multi_turn_single_step"}


def test_partition_produces_exactly_the_four_cells():
    assert set(partition()) == set(CELLS)


def test_non_empty_cells_have_cases():
    cells = partition()

    for name, cases in cells.items():
        if name in EMPTY_IN_FILE_MANAGEMENT:
            continue
        assert cases, f"{name} cell is empty; the classification may be broken"


def test_the_empty_cell_is_still_empty():
    cells = partition()

    for name in EMPTY_IN_FILE_MANAGEMENT:
        assert cells[name] == [], (
            f"{name} now has cases; it is measurable, update the docs"
        )


def test_single_turn_cells_are_truncated_to_one_turn():
    cells = partition()

    for name in ("single_turn_single_step", "single_turn_multi_step"):
        for case in cells[name]:
            assert len(case["question"]) == 1


def test_multi_turn_cells_keep_all_turns():
    cells = partition()

    multi = cells["multi_turn_multi_step"] + cells["multi_turn_single_step"]
    assert any(len(c["question"]) > 1 for c in multi)


def test_every_case_appears_in_exactly_one_cell_per_axis():
    cells = partition()
    st = len(cells["single_turn_single_step"]) + len(cells["single_turn_multi_step"])
    mt = len(cells["multi_turn_single_step"]) + len(cells["multi_turn_multi_step"])

    assert st == mt, "both axes must have the same total"
