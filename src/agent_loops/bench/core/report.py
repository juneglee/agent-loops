from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CELL_ORDER = (
    "single_turn_single_step",
    "single_turn_multi_step",
    "multi_turn_single_step",
    "multi_turn_multi_step",
)

LOOP_ORDER = (
    "single_call",
    "react",
    "rewoo",
    "plan_and_solve",
    "plan_and_execute",
    "plan_and_act",
    "adapt",
    "codeact",
    "fixed_pipeline",
)

NOT_MEASURED = "—"


def group_key(payload: dict[str, Any]) -> tuple[str, str]:
    scope = "all" if payload.get("all_classes") else "file_management"
    return payload.get("category", "?"), scope


def collect(results_dir: Path | str) -> dict[tuple[str, str], dict[str, dict]]:
    groups: dict[tuple[str, str], dict[str, dict]] = {}

    root = Path(results_dir)
    for path in sorted([*root.glob("*.json"), *root.glob("tasks/*.json")]):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "category" not in payload or "cell" not in payload:
            continue

        cells = groups.setdefault(group_key(payload), {})
        cell = payload["cell"]
        previous = cells.get(cell)
        cells[cell] = payload if previous is None else _merge_cell(previous, payload)

    return groups


def _run_stamp(payload: dict) -> dict:
    return {
        "when": payload.get("when"),
        "prompt_version": payload.get("prompt_version"),
        "layers": payload.get("layers", []),
        "loops": sorted(payload.get("summaries", {})),
    }


def _merge_cell(a: dict, b: dict) -> dict:
    older, newer = sorted((a, b), key=lambda p: p.get("when", ""))
    merged = dict(newer)
    merged["summaries"] = {**older.get("summaries", {}), **newer.get("summaries", {})}
    merged["cases"] = {**older.get("cases", {}), **newer.get("cases", {})}
    merged["runs"] = [
        *older.get("runs", [_run_stamp(older)]),
        *newer.get("runs", [_run_stamp(newer)]),
    ]
    return merged


def format_table(key: tuple[str, str], cells: dict[str, dict]) -> str:
    category, scope = key
    present = [c for c in CELL_ORDER if c in cells] + [
        c for c in cells if c not in CELL_ORDER
    ]

    counts = " | ".join(
        f"{_short(c)}={cells[c].get('n_cases', '?')} cases" for c in present
    )
    model = next(iter(cells.values())).get("model", "?")

    header = f"{'loop':<17}" + "".join(f"{_short(c):>12}" for c in present)
    lines = [
        f"## {category} | {scope}",
        f"model {model} | {counts}",
        "",
        header,
        "-" * len(header),
    ]

    loops = [n for n in LOOP_ORDER if _has(cells, n)]
    loops += sorted({n for c in cells.values() for n in c["summaries"]} - set(loops))

    for name in loops:
        row = f"{name:<17}"
        for cell in present:
            summary = cells[cell]["summaries"].get(name)
            row += (
                f"{NOT_MEASURED:>12}"
                if summary is None
                else f"{summary['accuracy']:>11.1%} "
            )
        lines.append(row)

    return "\n".join(lines)


def format_cost(key: tuple[str, str], cells: dict[str, dict]) -> str:
    present = [c for c in CELL_ORDER if c in cells]
    header = f"{'loop':<17}" + "".join(f"{_short(c):>12}" for c in present)
    lines = [
        f"### LLM calls (mean) — {key[0]} | {key[1]}",
        "",
        header,
        "-" * len(header),
    ]

    for name in [n for n in LOOP_ORDER if _has(cells, n)]:
        row = f"{name:<17}"
        for cell in present:
            summary = cells[cell]["summaries"].get(name)
            row += (
                f"{NOT_MEASURED:>12}"
                if summary is None
                else f"{summary['mean_llm_calls']:>12.1f}"
            )
        lines.append(row)

    return "\n".join(lines)


def _has(cells: dict[str, dict], loop: str) -> bool:
    return any(loop in c["summaries"] for c in cells.values())


def _short(cell: str) -> str:
    return (
        cell.replace("single_turn", "1T")
        .replace("multi_turn", "NT")
        .replace("single_step", "1S")
        .replace("multi_step", "NS")
        .replace("_", "")
    )


def main() -> int:
    groups = collect("results")
    if not groups:
        print("No results to read in results/.")
        return 1

    for key in sorted(groups):
        print()
        print(format_table(key, groups[key]))
        print()
        print(format_cost(key, groups[key]))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
