from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent_loops.tools import schemas

KNOWN_TOOLS = frozenset(s["function"]["name"] for s in schemas()) | {"execute_code"}
CELLS = (
    "single_turn_single_step",
    "single_turn_multi_step",
    "multi_turn_single_step",
    "multi_turn_multi_step",
)


def cell_of(case: dict[str, Any]) -> str:
    multi_turn = len(case["turns"]) > 1
    multi_step = any(len(turn) >= 2 for turn in case["gt_calls"])
    return f"{'multi' if multi_turn else 'single'}_turn_{'multi' if multi_step else 'single'}_step"


def turns_of(case: dict[str, Any]) -> list[str]:
    return list(case["turns"])


def fixture_dir(case: dict[str, Any], base: Path | str) -> Path:
    return (Path(base) / case["fixture"]).resolve()


def _fail(case: Any, reason: str) -> None:
    cid = case.get("id", "?") if isinstance(case, dict) else "?"
    raise ValueError(f"task {cid}: {reason}")


def validate(case: dict[str, Any], base: Path) -> None:
    for key in ("id", "fixture", "turns", "gt_calls"):
        if key not in case:
            _fail(case, f"missing required key: {key}")
    if (
        not isinstance(case["turns"], list)
        or not case["turns"]
        or not all(isinstance(t, str) and t.strip() for t in case["turns"])
    ):
        _fail(case, "turns must be a non-empty list of strings")
    if not isinstance(case["gt_calls"], list) or len(case["gt_calls"]) != len(
        case["turns"]
    ):
        _fail(
            case,
            f"gt_calls length ({len(case.get('gt_calls', []))}) differs from turns length ({len(case['turns'])})",
        )
    for turn in case["gt_calls"]:
        if not isinstance(turn, list):
            _fail(case, "each turn in gt_calls must be a list of calls")
        for call in turn:
            if (
                not isinstance(call, dict)
                or "name" not in call
                or "arguments" not in call
            ):
                _fail(case, f"call lacks name/arguments: {call}")
            if call["name"] not in KNOWN_TOOLS:
                _fail(case, f"unknown tool: {call['name']}")
            if not isinstance(call["arguments"], dict):
                _fail(case, f"arguments must be a dict: {call}")
    if not fixture_dir(case, base).is_dir():
        _fail(case, f"fixture directory does not exist: {case['fixture']}")
    derived = cell_of(case)
    if "cell" in case and case["cell"] != derived:
        _fail(
            case,
            f"declared cell ({case['cell']}) differs from the cell derived from gt_calls ({derived})",
        )
    expect = case.get("expect", {})
    if expect and not isinstance(expect, dict):
        _fail(case, "expect must be a dict")
    for key in ("answer_contains", "ignore"):
        if key in expect and not (
            isinstance(expect[key], list)
            and all(isinstance(x, str) for x in expect[key])
        ):
            _fail(case, f"expect.{key} must be a list of strings")


def load_tasks(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"{path}: must be an array of cases")  # noqa: TRY004
    seen: set[str] = set()
    for case in cases:
        validate(case, path.parent)
        if case["id"] in seen:
            _fail(case, "duplicate id")
        seen.add(case["id"])
        case.setdefault("cell", cell_of(case))
        case.setdefault("expect", {})
        case.setdefault("tags", [])
    return cases


def dataset_revision(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"tasks:{path.parent.name}@{digest}"
