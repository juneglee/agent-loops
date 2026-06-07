import json

import pytest

from agent_loops.bench.core.report import collect, format_table, group_key


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _payload(category, cell, all_classes=False, accuracy=0.5, loops=("react",)):
    return {
        "when": "20260828T000000Z",
        "category": category,
        "cell": cell,
        "model": "m",
        "all_classes": all_classes,
        "prompt_version": "v1",
        "n_cases": 4,
        "summaries": {
            name: {
                "n_cases": 4,
                "accuracy": accuracy,
                "parse_rate": 1.0,
                "mean_llm_calls": 2.0,
                "mean_tool_calls": 3.0,
                "mean_seconds": 5.0,
                "quiet_failure_rate": 0.0,
                "truncation_rate": 0.0,
                "loop_no_call_rate": 0.0,
            }
            for name in loops
        },
        "cases": {name: [] for name in loops},
    }


class TestGrouping:
    def test_category_and_scope_form_the_group(self):
        assert group_key(_payload("multi_turn_base", "c", all_classes=False)) == (
            "multi_turn_base",
            "file_management",
        )
        assert group_key(_payload("multi_turn_base", "c", all_classes=True)) == (
            "multi_turn_base",
            "all",
        )

    def test_different_categories_never_merge(self, tmp_path):
        _write(tmp_path, "a.json", _payload("multi_turn_base", "cell1"))
        _write(tmp_path, "b.json", _payload("multi_turn_miss_func", "cell1"))

        groups = collect(tmp_path)

        assert len(groups) == 2, (
            "base and augmented categories differ in difficulty and must not merge"
        )

    def test_different_scopes_never_merge(self, tmp_path):
        _write(tmp_path, "a.json", _payload("multi_turn_base", "c", all_classes=False))
        _write(tmp_path, "b.json", _payload("multi_turn_base", "c", all_classes=True))

        assert len(collect(tmp_path)) == 2, "different tool counts are not comparable"

    def test_same_group_different_cells_stay_separate_columns(self, tmp_path):
        _write(
            tmp_path, "a.json", _payload("multi_turn_base", "single_turn_single_step")
        )
        _write(tmp_path, "b.json", _payload("multi_turn_base", "multi_turn_multi_step"))

        groups = collect(tmp_path)

        assert len(groups) == 1
        cells = groups[("multi_turn_base", "file_management")]
        assert set(cells) == {"single_turn_single_step", "multi_turn_multi_step"}


class TestFreshness:
    def test_newer_run_replaces_older_for_the_same_cell(self, tmp_path):
        old = _payload("multi_turn_base", "c", accuracy=0.1)
        old["when"] = "20260101T000000Z"
        new = _payload("multi_turn_base", "c", accuracy=0.9)
        new["when"] = "20260828T000000Z"
        _write(tmp_path, "old.json", old)
        _write(tmp_path, "new.json", new)

        groups = collect(tmp_path)
        cell = groups[("multi_turn_base", "file_management")]["c"]

        assert cell["summaries"]["react"]["accuracy"] == 0.9, (
            "the stale result must not win"
        )


class TestTable:
    def test_table_lists_every_loop(self):
        cells = {"cellA": _payload("b", "cellA", loops=("react", "rewoo"))}

        text = format_table(("multi_turn_base", "file_management"), cells)

        assert "react" in text
        assert "rewoo" in text

    def test_missing_loop_in_a_cell_is_marked_not_blank(self):
        cells = {
            "cellA": _payload("b", "cellA", loops=("react", "rewoo")),
            "cellB": _payload("b", "cellB", loops=("react",)),
        }

        text = format_table(("multi_turn_base", "file_management"), cells)

        assert "—" in text, "missing measurement marker is absent"

    def test_case_count_is_shown(self):
        cells = {"cellA": _payload("b", "cellA")}

        assert "4" in format_table(("multi_turn_base", "file_management"), cells)


@pytest.mark.parametrize(
    "cell",
    [
        "single_turn_single_step",
        "single_turn_multi_step",
        "multi_turn_single_step",
        "multi_turn_multi_step",
    ],
)
def test_all_four_cells_are_recognized(cell):
    from agent_loops.bench.core.report import CELL_ORDER

    assert cell in CELL_ORDER


class TestPerLoopMerge:
    def test_runs_covering_different_loops_in_the_same_cell_are_merged_per_loop(
        self, tmp_path
    ):
        old = _payload("multi_turn_base", "c", accuracy=0.1, loops=("react", "rewoo"))
        old["when"] = "20260101T000000Z"
        new = _payload("multi_turn_base", "c", accuracy=0.9, loops=("react",))
        new["when"] = "20260828T000000Z"
        _write(tmp_path, "old.json", old)
        _write(tmp_path, "new.json", new)

        cell = collect(tmp_path)[("multi_turn_base", "file_management")]["c"]

        assert cell["summaries"]["react"]["accuracy"] == 0.9
        assert cell["summaries"]["rewoo"]["accuracy"] == 0.1
        assert set(cell["cases"]) == {"react", "rewoo"}
        assert cell["when"] == "20260828T000000Z"
