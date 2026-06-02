from __future__ import annotations

from typing import Any

from agent_loops.bench.bfcl.adapter import load_cases, load_ground_truth

CELLS = (
    "single_turn_single_step",
    "single_turn_multi_step",
    "multi_turn_single_step",
    "multi_turn_multi_step",
)


def truncate_to_first_turn(case: dict[str, Any]) -> dict[str, Any]:
    out = dict(case)
    out["question"] = case["question"][:1]
    return out


FILE_MANAGEMENT = {"GorillaFileSystem"}


def partition(
    category: str = "multi_turn_base",
    only_classes: set[str] | None = FILE_MANAGEMENT,
) -> dict[str, list[dict[str, Any]]]:
    cases = load_cases(category, only_classes=only_classes)
    gt = load_ground_truth(category)
    cells: dict[str, list[dict[str, Any]]] = {c: [] for c in CELLS}

    for case in cases:
        turns = gt.get(case["id"]) or []
        if not turns:
            continue
        first = turns[0]
        cell = (
            "single_turn_multi_step" if len(first) >= 2 else "single_turn_single_step"
        )
        cells[cell].append(truncate_to_first_turn(case))
        cell = (
            "multi_turn_multi_step"
            if any(len(t) >= 2 for t in turns)
            else "multi_turn_single_step"
        )
        cells[cell].append(case)

    return cells


def truncated_ground_truth(
    case_id: str, category: str = "multi_turn_base"
) -> list[list[str]]:
    return load_ground_truth(category)[case_id][:1]
